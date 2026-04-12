from __future__ import annotations

import json
import logging
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    ExperimentCell,
    ModelConfig,
    load_benchmarks,
    load_models,
    load_runtimes,
)
from .runner import BenchmarkRunner, RunRecord
from .reporters.export import export_csv, export_json
from .reporters.summary import render_markdown_summary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestratorConfig:
    """Configuration for the experiment orchestrator."""

    # Paths
    checkpoint_path: Path
    results_dir: Path

    # Config file paths
    runtimes_yaml: Path
    benchmarks_yaml: Path
    models_yaml: Path

    # Server binary
    server_binary: Path

    # Legacy single-model entrypoint. Kept for backward compatibility.
    model_id: str = ""
    # Mixed-model entrypoint. When set, each listed model participates in the run.
    model_ids: tuple[str, ...] = ()

    # Execution settings
    num_gpus: int = 1
    seed: int = 42
    request_timeout: float = 120.0
    max_retries: int = 2
    max_tokens: int = 256
    server_host: str = "127.0.0.1"
    base_port: int = 8080
    parallel_requests: int = 4


@dataclass(frozen=True)
class _ExecutionLane:
    """One long-lived llama-server lane used by the orchestrator."""

    name: str
    model_config: ModelConfig
    gpu_id: int | None
    port: int
    parallel_requests: int


class Orchestrator:
    """High-level experiment scheduler with resume and multi-GPU support.

    The orchestrator builds the full experiment matrix, delegates execution to
    ``BenchmarkRunner``, and manages checkpointing so runs can be resumed after
    interruption.

    In single-model mode, parallel execution shards cells round-robin across
    GPU lanes.

    In mixed-model mode, each model can be pinned to its own GPU lane via the
    ``gpu_id`` / ``port`` / ``parallel_requests`` fields in ``models.yaml``.
    This is the intended path for:

    - Qwen3-VL-2B-Thinking on GPU 0 / RTX 5070 Ti
    - Qwen3-VL-2B-Instruct on GPU 1 / RTX 4060 Ti
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self._shutdown_event = threading.Event()
        self._original_sigint: Any = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, *, parallel: bool = False, resume: bool = True) -> list[RunRecord]:
        """Execute the configured experiment matrix."""
        self._install_signal_handler()

        try:
            return self._run_matrix(parallel=parallel, resume=resume)
        finally:
            self._restore_signal_handler()

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _run_matrix(self, *, parallel: bool, resume: bool) -> list[RunRecord]:
        runtimes = load_runtimes(self.config.runtimes_yaml)
        benchmarks = load_benchmarks(self.config.benchmarks_yaml)
        loaded_models = load_models(self.config.models_yaml)

        target_models = self._resolve_target_models(loaded_models)
        models_by_id = {model.id: model for model in target_models}
        allow_legacy_cell_ids = len(target_models) == 1

        matrix = self._build_matrix(runtimes, benchmarks, target_models)
        total = len(matrix)
        logger.info(
            "Experiment matrix: %d cells (%d models x %d runtimes x %d benchmarks)",
            total,
            len(target_models),
            len(runtimes),
            len(benchmarks),
        )

        completed_ids = self._load_completed(resume)
        logger.info(
            "Completed cells (resume=%s): %d/%d",
            resume,
            len(completed_ids),
            total,
        )

        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        runs_dir = self.config.results_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        all_records: list[RunRecord] = []

        for cell in matrix:
            if not self._cell_completed(cell, completed_ids, allow_legacy_cell_ids):
                continue
            record_path = self._find_record_path(runs_dir, cell, allow_legacy_cell_ids)
            if record_path is None:
                continue
            record = self._load_record(record_path)
            if record is None:
                continue
            if not record.model_id:
                record.model_id = cell.model_id
            all_records.append(record)

        pending = [
            cell
            for cell in matrix
            if not self._cell_completed(cell, completed_ids, allow_legacy_cell_ids)
        ]
        logger.info("Pending cells: %d", len(pending))

        if not pending:
            logger.info("All cells already completed. Nothing to do.")
            self._generate_report(all_records)
            return all_records

        lanes = self._build_parallel_lanes(target_models) if parallel else []
        use_parallel = len(lanes) > 1

        if parallel and not use_parallel:
            logger.info(
                "Parallel execution requested but only one execution lane is "
                "available. Falling back to sequential mode."
            )

        if use_parallel:
            new_records = self._run_parallel(
                pending=pending,
                models_by_id=models_by_id,
                lanes=lanes,
                runs_dir=runs_dir,
                completed_ids=completed_ids,
                total=total,
            )
        else:
            new_records = self._run_sequential(
                pending=pending,
                models_by_id=models_by_id,
                runs_dir=runs_dir,
                completed_ids=completed_ids,
                total=total,
            )

        all_records.extend(new_records)
        self._generate_report(all_records)
        return all_records

    def _run_sequential(
        self,
        *,
        pending: list[ExperimentCell],
        models_by_id: dict[str, ModelConfig],
        runs_dir: Path,
        completed_ids: set[str],
        total: int,
    ) -> list[RunRecord]:
        """Run cells one at a time, honoring any per-model GPU pinning."""
        records: list[RunRecord] = []
        done_count = len(completed_ids)
        runner_cache: dict[tuple[int, int], BenchmarkRunner] = {}

        for i, cell in enumerate(pending):
            if self._shutdown_event.is_set():
                logger.warning("Shutdown requested. Stopping after %d cells.", i)
                break

            model_config = self._resolve_model_for_cell(cell, models_by_id)
            gpu_id, port, parallel_requests = self._resolve_sequential_exec_params(model_config)
            runner = self._get_runner(
                runner_cache,
                port=port,
                parallel_requests=parallel_requests,
            )

            done_count += 1
            logger.info(
                "Running cell %d/%d: %s x %s (model=%s, gpu=%s, port=%d, parallel=%d)",
                done_count,
                total,
                cell.runtime.id,
                cell.benchmark.id,
                model_config.id,
                gpu_id,
                port,
                parallel_requests,
            )

            record = runner.run_cell(
                cell,
                model_config,
                gpu_id=gpu_id,
                port=port,
                seed=self.config.seed,
                parallel_requests=parallel_requests,
            )
            if not record.model_id:
                record.model_id = model_config.id
            records.append(record)

            self._save_record(record, runs_dir / f"{record.cell_id}.json")
            completed_ids.add(record.cell_id)
            self._save_checkpoint(completed_ids)

            logger.info(
                "Cell %d/%d done: %s x %s -> status=%s, score=%.4f, model=%s",
                done_count,
                total,
                cell.runtime.id,
                cell.benchmark.id,
                record.status,
                record.score or 0.0,
                record.model_id,
            )

        return records

    def _run_parallel(
        self,
        *,
        pending: list[ExperimentCell],
        models_by_id: dict[str, ModelConfig],
        lanes: list[_ExecutionLane],
        runs_dir: Path,
        completed_ids: set[str],
        total: int,
    ) -> list[RunRecord]:
        """Run pending cells across dedicated execution lanes."""
        records: list[RunRecord] = []
        done_count = len(completed_ids)
        lock = threading.Lock()

        runners = {
            lane.name: BenchmarkRunner(
                server_binary=self.config.server_binary,
                default_host=self.config.server_host,
                default_port=lane.port,
                request_timeout=self.config.request_timeout,
                max_retries=self.config.max_retries,
                max_tokens=self.config.max_tokens,
                parallel_requests=lane.parallel_requests,
            )
            for lane in lanes
        }

        lane_queues = self._assign_cells_to_lanes(pending, lanes)

        def _run_lane_queue(lane: _ExecutionLane) -> list[RunRecord]:
            lane_records: list[RunRecord] = []
            runner = runners[lane.name]

            for cell in lane_queues[lane.name]:
                if self._shutdown_event.is_set():
                    break

                model_config = self._resolve_model_for_cell(cell, models_by_id)
                record = runner.run_cell(
                    cell,
                    model_config,
                    gpu_id=lane.gpu_id,
                    port=lane.port,
                    seed=self.config.seed,
                    parallel_requests=lane.parallel_requests,
                )
                if not record.model_id:
                    record.model_id = model_config.id

                with lock:
                    records.append(record)
                    self._save_record(record, runs_dir / f"{record.cell_id}.json")
                    completed_ids.add(record.cell_id)
                    self._save_checkpoint(completed_ids)

                    nonlocal done_count
                    done_count += 1
                    logger.info(
                        "Cell %d/%d done: %s x %s -> status=%s, score=%.4f, "
                        "model=%s, gpu=%s",
                        done_count,
                        total,
                        cell.runtime.id,
                        cell.benchmark.id,
                        record.status,
                        record.score or 0.0,
                        record.model_id,
                        lane.gpu_id,
                    )

                lane_records.append(record)

            return lane_records

        with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
            futures = [pool.submit(_run_lane_queue, lane) for lane in lanes]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    logger.error("Execution lane raised an exception: %s", exc)

        return records

    # ------------------------------------------------------------------
    # Lane planning
    # ------------------------------------------------------------------

    def _resolve_target_models(
        self, loaded_models: dict[str, ModelConfig]
    ) -> list[ModelConfig]:
        if self.config.model_ids:
            model_ids = list(self.config.model_ids)
        elif self.config.model_id:
            model_ids = [self.config.model_id]
        else:
            raise ValueError(
                "No model configured. Set OrchestratorConfig.model_id or "
                "OrchestratorConfig.model_ids."
            )

        resolved: list[ModelConfig] = []
        seen: set[str] = set()
        for model_id in model_ids:
            if model_id in seen:
                continue
            if model_id not in loaded_models:
                raise ValueError(
                    f"Model '{model_id}' not found in models.yaml. "
                    f"Available: {sorted(loaded_models.keys())}"
                )
            resolved.append(loaded_models[model_id])
            seen.add(model_id)
        return resolved

    def _build_matrix(
        self,
        runtimes: list[Any],
        benchmarks: list[Any],
        target_models: list[ModelConfig],
    ) -> list[ExperimentCell]:
        matrix: list[ExperimentCell] = []
        for model in target_models:
            for runtime in runtimes:
                for benchmark in benchmarks:
                    matrix.append(
                        ExperimentCell(
                            runtime=runtime,
                            benchmark=benchmark,
                            model_id=model.id,
                        )
                    )
        return matrix

    def _build_parallel_lanes(
        self, target_models: list[ModelConfig]
    ) -> list[_ExecutionLane]:
        if len(target_models) > 1:
            lanes: list[_ExecutionLane] = []
            seen_gpu_ids: set[int] = set()
            seen_ports: set[int] = set()

            for model in target_models:
                if model.gpu_id is None:
                    raise ValueError(
                        "Mixed-model parallel execution requires gpu_id for each "
                        "model in models.yaml."
                    )
                gpu_id = model.gpu_id
                port = model.port if model.port is not None else self.config.base_port + gpu_id
                parallel_requests = (
                    model.parallel_requests
                    if model.parallel_requests is not None
                    else self.config.parallel_requests
                )

                if gpu_id in seen_gpu_ids:
                    raise ValueError(
                        f"Duplicate gpu_id={gpu_id} in mixed-model lane config."
                    )
                if port in seen_ports:
                    raise ValueError(
                        f"Duplicate port={port} in mixed-model lane config."
                    )

                seen_gpu_ids.add(gpu_id)
                seen_ports.add(port)
                lanes.append(
                    _ExecutionLane(
                        name=model.id,
                        model_config=model,
                        gpu_id=gpu_id,
                        port=port,
                        parallel_requests=parallel_requests,
                    )
                )

            return lanes

        lane_count = max(1, self.config.num_gpus)
        if lane_count < 2:
            return []

        model = target_models[0]
        parallel_requests = (
            model.parallel_requests
            if model.parallel_requests is not None
            else self.config.parallel_requests
        )

        return [
            _ExecutionLane(
                name=f"gpu{gpu_id}",
                model_config=model,
                gpu_id=gpu_id,
                port=self.config.base_port + gpu_id,
                parallel_requests=parallel_requests,
            )
            for gpu_id in range(lane_count)
        ]

    def _assign_cells_to_lanes(
        self, pending: list[ExperimentCell], lanes: list[_ExecutionLane]
    ) -> dict[str, list[ExperimentCell]]:
        lane_queues = {lane.name: [] for lane in lanes}

        if len({lane.model_config.id for lane in lanes}) > 1:
            lane_by_model = {lane.model_config.id: lane for lane in lanes}
            for cell in pending:
                lane = lane_by_model.get(cell.model_id)
                if lane is None:
                    raise ValueError(
                        f"No execution lane configured for model '{cell.model_id}'."
                    )
                lane_queues[lane.name].append(cell)
            return lane_queues

        lane_names = [lane.name for lane in lanes]
        for idx, cell in enumerate(pending):
            lane_queues[lane_names[idx % len(lane_names)]].append(cell)
        return lane_queues

    def _resolve_sequential_exec_params(
        self, model_config: ModelConfig
    ) -> tuple[int | None, int, int]:
        gpu_id = model_config.gpu_id
        if model_config.port is not None:
            port = model_config.port
        elif gpu_id is not None:
            port = self.config.base_port + gpu_id
        else:
            port = self.config.base_port
        parallel_requests = (
            model_config.parallel_requests
            if model_config.parallel_requests is not None
            else self.config.parallel_requests
        )
        return gpu_id, port, parallel_requests

    def _resolve_model_for_cell(
        self, cell: ExperimentCell, models_by_id: dict[str, ModelConfig]
    ) -> ModelConfig:
        if cell.model_id:
            return models_by_id[cell.model_id]
        if len(models_by_id) == 1:
            return next(iter(models_by_id.values()))
        raise ValueError(
            "ExperimentCell.model_id is empty in a multi-model orchestrator run."
        )

    def _get_runner(
        self,
        cache: dict[tuple[int, int], BenchmarkRunner],
        *,
        port: int,
        parallel_requests: int,
    ) -> BenchmarkRunner:
        key = (port, parallel_requests)
        runner = cache.get(key)
        if runner is not None:
            return runner

        runner = BenchmarkRunner(
            server_binary=self.config.server_binary,
            default_host=self.config.server_host,
            default_port=port,
            request_timeout=self.config.request_timeout,
            max_retries=self.config.max_retries,
            max_tokens=self.config.max_tokens,
            parallel_requests=parallel_requests,
        )
        cache[key] = runner
        return runner

    # ------------------------------------------------------------------
    # Checkpoint / record helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cell_id(model_id: str, runtime_id: str, benchmark_id: str) -> str:
        if model_id:
            return f"{model_id}_{runtime_id}_{benchmark_id}"
        return f"{runtime_id}_{benchmark_id}"

    def _cell_id_for_cell(self, cell: ExperimentCell) -> str:
        return self._make_cell_id(cell.model_id, cell.runtime.id, cell.benchmark.id)

    def _candidate_cell_ids(
        self, cell: ExperimentCell, allow_legacy: bool
    ) -> tuple[str, ...]:
        ids = [self._cell_id_for_cell(cell)]
        if allow_legacy:
            ids.append(self._make_cell_id("", cell.runtime.id, cell.benchmark.id))
        return tuple(ids)

    def _cell_completed(
        self, cell: ExperimentCell, completed_ids: set[str], allow_legacy: bool
    ) -> bool:
        return any(
            candidate in completed_ids
            for candidate in self._candidate_cell_ids(cell, allow_legacy)
        )

    def _find_record_path(
        self, runs_dir: Path, cell: ExperimentCell, allow_legacy: bool
    ) -> Path | None:
        for candidate in self._candidate_cell_ids(cell, allow_legacy):
            path = runs_dir / f"{candidate}.json"
            if path.exists():
                return path
        return None

    def _load_completed(self, resume: bool) -> set[str]:
        """Load the set of completed cell IDs from checkpoint and result files."""
        completed: set[str] = set()
        if not resume:
            return completed

        cp = self.config.checkpoint_path
        if cp.exists():
            try:
                data = json.loads(cp.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    completed.update(str(item) for item in data)
                elif isinstance(data, dict) and "completed" in data:
                    completed.update(str(item) for item in data["completed"])
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read checkpoint %s: %s", cp, exc)

        runs_dir = self.config.results_dir / "runs"
        if runs_dir.is_dir():
            for result_file in runs_dir.glob("*.json"):
                if result_file.name == "checkpoint.json":
                    continue
                completed.add(result_file.stem)

        return completed

    def _save_checkpoint(self, completed_ids: set[str]) -> None:
        """Persist the set of completed cell IDs to the checkpoint file."""
        cp = self.config.checkpoint_path
        cp.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "completed": sorted(completed_ids),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_completed": len(completed_ids),
        }
        try:
            cp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to write checkpoint: %s", exc)

    def _save_record(self, record: RunRecord, path: Path) -> None:
        """Save a single RunRecord as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(
                json.dumps(record.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to save record to %s: %s", path, exc)

    def _load_record(self, path: Path) -> RunRecord | None:
        """Load a RunRecord from a JSON file, returning None on failure."""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))

            from .runner import SampleResult

            sample_results = []
            for sr in data.get("sample_results", []):
                if isinstance(sr, dict):
                    sample_results.append(SampleResult(**sr))
                else:
                    sample_results.append(sr)

            return RunRecord(
                runtime_id=data["runtime_id"],
                benchmark_id=data["benchmark_id"],
                status=data["status"],
                model_id=data.get("model_id", ""),
                score=data.get("score"),
                n_samples=data.get("n_samples", 0),
                n_succeeded=data.get("n_succeeded", 0),
                n_failed=data.get("n_failed", 0),
                wall_time_seconds=data.get("wall_time_seconds", 0.0),
                sample_results=sample_results,
                notes=data.get("notes", ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError, OSError) as exc:
            logger.warning("Failed to load record from %s: %s", path, exc)
            return None

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _generate_report(self, records: list[RunRecord]) -> None:
        """Generate summary outputs for single-model and mixed-model runs."""
        reports_dir = self.config.results_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        summary_dicts = [
            {
                "model_id": r.model_id,
                "runtime_id": r.runtime_id,
                "benchmark_id": r.benchmark_id,
                "status": r.status,
                "score": r.score,
                "n_samples": r.n_samples,
                "n_succeeded": r.n_succeeded,
                "n_failed": r.n_failed,
                "wall_time_seconds": round(r.wall_time_seconds, 1),
                "notes": r.notes,
            }
            for r in records
        ]

        export_json(summary_dicts, reports_dir / "results_summary.json")
        export_csv(summary_dicts, reports_dir / "results_summary.csv")

        md = render_markdown_summary(summary_dicts)
        md_path = reports_dir / "results_summary.md"
        md_path.write_text(md, encoding="utf-8")

        logger.info("Report generated: %d records -> %s", len(records), reports_dir)

        model_ids = sorted({row["model_id"] for row in summary_dicts if row.get("model_id")})
        if len(model_ids) > 1:
            logger.info(
                "Mixed-model summary detected. Generating per-model reports "
                "and skipping aggregate charts to avoid model collisions."
            )
            self._generate_per_model_reports(summary_dicts, reports_dir, model_ids)
            return

        try:
            import pandas as pd
            from .reporters.charts import generate_all_charts

            results_df = pd.DataFrame(summary_dicts)
            generate_all_charts(results_df, output_dir=reports_dir)
            logger.info("Charts generated in %s", reports_dir)
        except Exception as exc:
            logger.warning("Chart generation failed: %s", exc)

    def _generate_per_model_reports(
        self,
        summary_dicts: list[dict[str, Any]],
        reports_dir: Path,
        model_ids: list[str],
    ) -> None:
        try:
            import pandas as pd
            from .reporters.charts import generate_all_charts
        except Exception as exc:
            logger.warning("Per-model chart imports failed: %s", exc)
            return

        for model_id in model_ids:
            model_rows = [
                row for row in summary_dicts if row.get("model_id") == model_id
            ]
            model_dir = reports_dir / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            export_json(model_rows, model_dir / "results_summary.json")
            export_csv(model_rows, model_dir / "results_summary.csv")
            (model_dir / "results_summary.md").write_text(
                render_markdown_summary(model_rows),
                encoding="utf-8",
            )

            try:
                generate_all_charts(pd.DataFrame(model_rows), output_dir=model_dir)
                logger.info("Per-model charts generated in %s", model_dir)
            except Exception as exc:
                logger.warning(
                    "Chart generation failed for model %s: %s", model_id, exc
                )

    # ------------------------------------------------------------------
    # Signal handling for graceful shutdown
    # ------------------------------------------------------------------

    def _install_signal_handler(self) -> None:
        """Install a SIGINT handler that sets the shutdown event."""
        self._original_sigint = signal.getsignal(signal.SIGINT)

        def _handler(signum: int, frame: Any) -> None:
            if self._shutdown_event.is_set():
                logger.warning("Second interrupt received. Force exiting.")
                sys.exit(1)
            logger.warning(
                "Interrupt received. Finishing current cell(s) and saving checkpoint..."
            )
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, _handler)

    def _restore_signal_handler(self) -> None:
        """Restore the original SIGINT handler."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
            self._original_sigint = None
