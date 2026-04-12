from __future__ import annotations

import json
import logging
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    ExperimentCell,
    ModelConfig,
    build_matrix,
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

    # Model to use (key into models.yaml)
    model_id: str

    # Execution settings
    num_gpus: int = 1
    seed: int = 42
    request_timeout: float = 120.0
    max_retries: int = 2
    max_tokens: int = 256
    server_host: str = "127.0.0.1"
    base_port: int = 8080
    parallel_requests: int = 4


class Orchestrator:
    """High-level experiment scheduler with resume and multi-GPU support.

    The orchestrator builds the full experiment matrix (runtimes x benchmarks),
    iterates over each cell, delegates execution to ``BenchmarkRunner``, and
    manages checkpointing so runs can be resumed after interruption.

    In dual-GPU mode, two cells run in parallel on GPU 0 and GPU 1 with
    separate server instances on different ports.
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self._shutdown_event = threading.Event()
        self._original_sigint: Any = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, *, parallel: bool = False, resume: bool = True) -> list[RunRecord]:
        """Execute the full experiment matrix.

        Args:
            parallel: If True and ``config.num_gpus >= 2``, run two cells
                simultaneously on GPU 0 and GPU 1.
            resume: If True, skip cells whose result files already exist.

        Returns:
            List of RunRecord objects for all executed (and resumed) cells.
        """
        # Install signal handler for graceful shutdown
        self._install_signal_handler()

        try:
            return self._run_matrix(parallel=parallel, resume=resume)
        finally:
            self._restore_signal_handler()

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _run_matrix(
        self, *, parallel: bool, resume: bool
    ) -> list[RunRecord]:
        # 1. Load configuration
        runtimes = load_runtimes(self.config.runtimes_yaml)
        benchmarks = load_benchmarks(self.config.benchmarks_yaml)
        models = load_models(self.config.models_yaml)

        if self.config.model_id not in models:
            raise ValueError(
                f"Model '{self.config.model_id}' not found in models.yaml. "
                f"Available: {sorted(models.keys())}"
            )
        model_config = models[self.config.model_id]

        # 2. Build experiment matrix
        matrix = build_matrix(runtimes, benchmarks)
        total = len(matrix)
        logger.info("Experiment matrix: %d cells (%d runtimes x %d benchmarks)",
                     total, len(runtimes), len(benchmarks))

        # 3. Load checkpoint / determine completed cells
        completed_ids = self._load_completed(resume)
        logger.info("Completed cells (resume=%s): %d/%d", resume, len(completed_ids), total)

        # 4. Ensure results directory exists
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        runs_dir = self.config.results_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        # 5. Collect all records (both resumed and new)
        all_records: list[RunRecord] = []

        # Load existing records for completed cells
        for cell in matrix:
            cell_id = f"{cell.runtime.id}_{cell.benchmark.id}"
            if cell_id in completed_ids:
                record = self._load_record(runs_dir / f"{cell_id}.json")
                if record is not None:
                    all_records.append(record)

        # 6. Filter to pending cells
        pending = [
            cell for cell in matrix
            if f"{cell.runtime.id}_{cell.benchmark.id}" not in completed_ids
        ]
        logger.info("Pending cells: %d", len(pending))

        if not pending:
            logger.info("All cells already completed. Nothing to do.")
            self._generate_report(all_records)
            return all_records

        # 7. Execute pending cells
        use_parallel = parallel and self.config.num_gpus >= 2
        if use_parallel:
            new_records = self._run_parallel(
                pending, model_config, runs_dir, completed_ids, total
            )
        else:
            new_records = self._run_sequential(
                pending, model_config, runs_dir, completed_ids, total
            )

        all_records.extend(new_records)

        # 8. Generate summary report
        self._generate_report(all_records)

        return all_records

    def _run_sequential(
        self,
        pending: list[ExperimentCell],
        model_config: ModelConfig,
        runs_dir: Path,
        completed_ids: set[str],
        total: int,
    ) -> list[RunRecord]:
        """Run cells one at a time on a single GPU."""
        runner = BenchmarkRunner(
            server_binary=self.config.server_binary,
            default_host=self.config.server_host,
            default_port=self.config.base_port,
            request_timeout=self.config.request_timeout,
            max_retries=self.config.max_retries,
            max_tokens=self.config.max_tokens,
            parallel_requests=self.config.parallel_requests,
        )

        records: list[RunRecord] = []
        done_count = len(completed_ids)

        for i, cell in enumerate(pending):
            if self._shutdown_event.is_set():
                logger.warning("Shutdown requested. Stopping after %d cells.", i)
                break

            cell_id = f"{cell.runtime.id}_{cell.benchmark.id}"
            done_count += 1
            logger.info(
                "Running cell %d/%d: %s x %s",
                done_count, total, cell.runtime.id, cell.benchmark.id,
            )

            record = runner.run_cell(
                cell, model_config, seed=self.config.seed
            )
            records.append(record)

            # Save individual result
            self._save_record(record, runs_dir / f"{cell_id}.json")

            # Update checkpoint
            completed_ids.add(cell_id)
            self._save_checkpoint(completed_ids)

            logger.info(
                "Cell %d/%d done: %s x %s -> status=%s, score=%.4f",
                done_count,
                total,
                cell.runtime.id,
                cell.benchmark.id,
                record.status,
                record.score or 0.0,
            )

        return records

    def _run_parallel(
        self,
        pending: list[ExperimentCell],
        model_config: ModelConfig,
        runs_dir: Path,
        completed_ids: set[str],
        total: int,
    ) -> list[RunRecord]:
        """Run cells in parallel across two GPUs.

        GPU 0 uses port ``base_port``, GPU 1 uses ``base_port + 1``.
        A ThreadPoolExecutor with 2 workers processes cells from the
        pending queue.
        """
        records: list[RunRecord] = []
        done_count = len(completed_ids)
        lock = threading.Lock()

        # Create per-GPU runners with different ports
        runner_gpu0 = BenchmarkRunner(
            server_binary=self.config.server_binary,
            default_host=self.config.server_host,
            default_port=self.config.base_port,
            request_timeout=self.config.request_timeout,
            max_retries=self.config.max_retries,
            max_tokens=self.config.max_tokens,
            parallel_requests=self.config.parallel_requests,
        )
        runner_gpu1 = BenchmarkRunner(
            server_binary=self.config.server_binary,
            default_host=self.config.server_host,
            default_port=self.config.base_port + 1,
            request_timeout=self.config.request_timeout,
            max_retries=self.config.max_retries,
            max_tokens=self.config.max_tokens,
            parallel_requests=self.config.parallel_requests,
        )

        def _execute_cell(
            cell: ExperimentCell, gpu_id: int, runner: BenchmarkRunner
        ) -> RunRecord:
            port = self.config.base_port + gpu_id
            return runner.run_cell(
                cell,
                model_config,
                gpu_id=gpu_id,
                port=port,
                seed=self.config.seed,
            )

        # Split pending cells into two queues: one per GPU.
        # Each GPU processes its queue sequentially (only one server
        # can run on a given port at a time).
        gpu0_cells = [c for i, c in enumerate(pending) if i % 2 == 0]
        gpu1_cells = [c for i, c in enumerate(pending) if i % 2 == 1]

        def _run_gpu_queue(
            cells: list[ExperimentCell], gpu_id: int, runner: BenchmarkRunner
        ) -> list[RunRecord]:
            gpu_records: list[RunRecord] = []
            for cell in cells:
                if self._shutdown_event.is_set():
                    break
                port = self.config.base_port + gpu_id
                record = runner.run_cell(
                    cell, model_config, gpu_id=gpu_id, port=port,
                    seed=self.config.seed,
                )
                cell_id = f"{cell.runtime.id}_{cell.benchmark.id}"
                with lock:
                    records.append(record)
                    self._save_record(record, runs_dir / f"{cell_id}.json")
                    completed_ids.add(cell_id)
                    self._save_checkpoint(completed_ids)
                    nonlocal done_count
                    done_count += 1
                    logger.info(
                        "Cell %d/%d done: %s x %s -> status=%s, score=%.4f",
                        done_count, total,
                        cell.runtime.id, cell.benchmark.id,
                        record.status, record.score or 0.0,
                    )
                gpu_records.append(record)
            return gpu_records

        with ThreadPoolExecutor(max_workers=2) as pool:
            f0 = pool.submit(_run_gpu_queue, gpu0_cells, 0, runner_gpu0)
            f1 = pool.submit(_run_gpu_queue, gpu1_cells, 1, runner_gpu1)

            for future in as_completed([f0, f1]):
                try:
                    future.result()
                except Exception as exc:
                    logger.error("GPU queue raised an exception: %s", exc)

        return records

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _load_completed(self, resume: bool) -> set[str]:
        """Load the set of completed cell IDs from checkpoint and result files.

        When ``resume`` is True, the method checks both the checkpoint JSON
        and the existence of individual result files. A cell is considered
        complete if its result file exists in ``results/runs/``.
        """
        completed: set[str] = set()
        if not resume:
            return completed

        # Read checkpoint file
        cp = self.config.checkpoint_path
        if cp.exists():
            try:
                data = json.loads(cp.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    completed.update(data)
                elif isinstance(data, dict) and "completed" in data:
                    completed.update(data["completed"])
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read checkpoint %s: %s", cp, exc)

        # Also scan result files in case checkpoint is out of sync
        runs_dir = self.config.results_dir / "runs"
        if runs_dir.is_dir():
            for result_file in runs_dir.glob("*.json"):
                if result_file.name == "checkpoint.json":
                    continue
                cell_id = result_file.stem
                completed.add(cell_id)

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
            # Reconstruct RunRecord from dict; sample_results may be dicts
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
        """Generate summary outputs: JSON, CSV, and markdown."""
        reports_dir = self.config.results_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Simplified records for summary (without per-sample data)
        summary_dicts = [
            {
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

        # Export JSON and CSV
        export_json(summary_dicts, reports_dir / "results_summary.json")
        export_csv(summary_dicts, reports_dir / "results_summary.csv")

        # Markdown summary
        md = render_markdown_summary(summary_dicts)
        md_path = reports_dir / "results_summary.md"
        md_path.write_text(md, encoding="utf-8")

        logger.info(
            "Report generated: %d records -> %s",
            len(records),
            reports_dir,
        )

        # Generate charts from the summary data
        try:
            import pandas as pd
            from .reporters.charts import generate_all_charts

            results_df = pd.DataFrame(summary_dicts)
            generate_all_charts(results_df, output_dir=reports_dir)
            logger.info("Charts generated in %s", reports_dir)
        except Exception as e:
            logger.warning("Chart generation failed: %s", e)

    # ------------------------------------------------------------------
    # Signal handling for graceful shutdown
    # ------------------------------------------------------------------

    def _install_signal_handler(self) -> None:
        """Install a SIGINT handler that sets the shutdown event."""
        self._original_sigint = signal.getsignal(signal.SIGINT)

        def _handler(signum: int, frame: Any) -> None:
            if self._shutdown_event.is_set():
                # Second Ctrl+C: force exit
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
