#!/usr/bin/env python3
"""Patch run: re-run failed cells from the initial vlm_final_n30 run.

Only re-runs cells that had 'error' status (port bind / pkill race issues).
TQ server_crash cells are kept as-is (genuine CUDA kernel crashes).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from tq_bench.env import default_server_binary, prepend_ld_library_path, project_root_from

_PROJECT = project_root_from(__file__)
_BUILD_BIN = _PROJECT / "llama.cpp" / "build" / "bin"
prepend_ld_library_path(_BUILD_BIN)

from tq_bench.config import (
    BenchmarkConfig,
    ExperimentCell,
    load_benchmarks,
    load_models,
    load_runtimes,
)
from tq_bench.runner import BenchmarkRunner, RunRecord

BENCH_DIR = Path(__file__).resolve().parent
SERVER_BINARY = default_server_binary(_PROJECT)
EXISTING_PATH = _PROJECT / "results" / "runs" / "vlm_final_n30.json"
OUTPUT_PATH = _PROJECT / "results" / "runs" / "vlm_final_n30.json"

RUNTIME_IDS = ["baseline", "lcpp-kv-4", "tq-4", "tq-K4V3", "tq-3"]
BENCHMARK_IDS = ["ai2d", "mmmu", "mathvista"]
MODEL_ID = "qwen3_vl_2b_instruct"
N_SAMPLES = 30
PORT = 8080

OFFICIAL_SCORES = {
    "ai2d": 0.804,
    "mmmu": 0.614,
    "mathvista": 0.736,
}

TQ_RUNTIME_IDS = {"tq-4", "tq-K4V3", "tq-3", "tq-2", "tq-2h", "tq-3h"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vlm_patch")


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(_PROJECT),
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def wait_port_free(port: int, timeout: float = 15.0) -> None:
    """Block until no process is listening on port."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ["lsof", "-t", "-i", f"TCP:{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=3,
            )
            pids = [p.strip() for p in result.stdout.strip().split() if p.strip().isdigit()]
            if not pids:
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        time.sleep(1.0)
    logger.warning("Port %d still occupied after %.0fs wait", port, timeout)


def force_kill_port(port: int) -> None:
    """Kill anything on port and wait for it to be free."""
    try:
        result = subprocess.run(
            ["lsof", "-t", "-i", f"TCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [int(p) for p in result.stdout.strip().split() if p.strip().isdigit()]
        for pid in pids:
            if pid == os.getpid():
                continue
            logger.info("Killing pid=%d on port %d", pid, port)
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        if pids:
            time.sleep(1.5)
            for pid in pids:
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    wait_port_free(port, timeout=10.0)


def main():
    t_start = time.monotonic()

    # Load existing results
    with open(EXISTING_PATH) as f:
        existing = json.load(f)

    # Build a map of existing records: (model_id, runtime_id, benchmark_id) -> record
    existing_map: dict[tuple[str, str, str], dict] = {}
    for rec in existing.get("records", []):
        key = (
            rec.get("model_id", MODEL_ID),
            rec["runtime_id"],
            rec["benchmark_id"],
        )
        existing_map[key] = rec

    # Identify cells that need re-running (status == "error")
    cells_to_rerun = []
    for key, rec in existing_map.items():
        if rec["status"] == "error":
            cells_to_rerun.append(key)

    logger.info("Cells to re-run (status=error): %s", cells_to_rerun)

    if not cells_to_rerun:
        logger.info("No cells to re-run. All done.")
        return

    # Load configs
    runtimes_all = load_runtimes(BENCH_DIR / "configs" / "runtimes.yaml")
    benchmarks_all = load_benchmarks(BENCH_DIR / "configs" / "benchmarks.yaml")
    models = load_models(BENCH_DIR / "configs" / "models.yaml")
    runtimes_map = {rt.id: rt for rt in runtimes_all}
    benchmarks_map = {bm.id: bm for bm in benchmarks_all}
    model_config = models[MODEL_ID]
    port = model_config.port or PORT
    gpu_id = model_config.gpu_id

    # Create runner with parallel=1 for safety
    runner = BenchmarkRunner(
        server_binary=SERVER_BINARY,
        default_port=port,
        request_timeout=180.0,
        max_retries=2,
        parallel_requests=1,
    )

    new_records: dict[tuple[str, str, str], dict] = {}

    for i, (_, rt_id, bm_id) in enumerate(cells_to_rerun):
        rt = runtimes_map.get(rt_id)
        bm_orig = benchmarks_map.get(bm_id)
        if not rt or not bm_orig:
            logger.warning("Missing config for %s x %s, skipping", rt_id, bm_id)
            continue

        is_tq = rt_id in TQ_RUNTIME_IDS
        n_parallel = 1  # Use 1 for all re-runs to be safe

        bm = BenchmarkConfig(
            id=bm_orig.id,
            task_type=bm_orig.task_type,
            sample_count=N_SAMPLES,
            metric=bm_orig.metric,
            max_tokens=bm_orig.max_tokens,
        )
        cell = ExperimentCell(runtime=rt, benchmark=bm, model_id=model_config.id)

        logger.info("")
        logger.info("=" * 60)
        logger.info("[%d/%d] RE-RUN: %s x %s (bits=%s, K=%s, V=%s, parallel=%d)",
                    i + 1, len(cells_to_rerun), rt.id, bm.id,
                    rt.bits, rt.cache_type_k, rt.cache_type_v, n_parallel)
        logger.info("=" * 60)

        force_kill_port(port)

        try:
            record = runner.run_cell(
                cell,
                model_config,
                gpu_id=gpu_id,
                port=port,
                seed=42,
                parallel_requests=n_parallel,
            )
        except Exception as exc:
            logger.error("Cell %s x %s failed: %s", rt_id, bm_id, exc)
            record = RunRecord(
                runtime_id=rt_id,
                benchmark_id=bm_id,
                status="error",
                model_id=model_config.id,
                score=0.0,
                notes=f"Exception: {exc}",
            )

        rec_dict = record.to_dict()
        rec_dict["runtime_config"] = {
            "cache_type_k": rt.cache_type_k,
            "cache_type_v": rt.cache_type_v,
            "bits": rt.bits,
            "method": rt.method,
        }
        new_records[(model_config.id, rt_id, bm_id)] = rec_dict

        logger.info("  => status=%s, score=%.4f, time=%.1fs, ok=%d/%d",
                    record.status, record.score or 0,
                    record.wall_time_seconds,
                    record.n_succeeded, record.n_samples)

    # Merge new records into existing results
    updated_records = []
    for rec in existing.get("records", []):
        key = (
            rec.get("model_id", MODEL_ID),
            rec["runtime_id"],
            rec["benchmark_id"],
        )
        if key in new_records:
            updated_records.append(new_records[key])
            logger.info(
                "Replaced %s :: %s x %s: %s -> %s",
                key[0], key[1], key[2], rec["status"], new_records[key]["status"]
            )
        else:
            updated_records.append(rec)

    # Rebuild scores summary
    scores_table: dict[str, dict[str, float | None]] = {}
    for rec in updated_records:
        rt_id = rec["runtime_id"]
        bm_id = rec["benchmark_id"]
        if rt_id not in scores_table:
            scores_table[rt_id] = {}
        scores_table[rt_id][bm_id] = rec.get("score")

    total_time = time.monotonic() - t_start
    existing["meta"]["timestamp_patch"] = datetime.now(timezone.utc).isoformat()
    existing["meta"]["patch_total_seconds"] = round(total_time, 1)
    existing["meta"]["patch_cells_rerun"] = [f"{k[0]}x{k[1]}" for k in cells_to_rerun]
    existing["meta"]["gpu_id"] = gpu_id
    existing["meta"]["port"] = port
    existing["scores_summary"] = scores_table
    existing["records"] = updated_records

    with open(OUTPUT_PATH, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    logger.info("Patched results saved to %s", OUTPUT_PATH)

    # Print summary
    runtimes = [runtimes_map[rid] for rid in RUNTIME_IDS if rid in runtimes_map]
    print("\n" + "=" * 75)
    print("VLM BENCHMARK RESULTS (n=%d, %s) - PATCHED" % (N_SAMPLES, model_config.description))
    print("=" * 75)

    header = f"{'Runtime':<14} {'Bits':<7} {'Status':<12}"
    for bid in BENCHMARK_IDS:
        header += f" {bid:>10}"
    header += f" {'Avg':>8}"
    print(header)
    print("-" * 75)

    for rt in runtimes:
        row_scores = []
        # Determine worst status for this runtime
        statuses = set()
        for bid in BENCHMARK_IDS:
            for rec in updated_records:
                if rec["runtime_id"] == rt.id and rec["benchmark_id"] == bid:
                    statuses.add(rec["status"])
        worst = "ok"
        if "server_crash" in statuses:
            worst = "crash"
        elif "error" in statuses:
            worst = "error"

        line = f"{rt.id:<14} {rt.bits:<7} {worst:<12}"
        for bid in BENCHMARK_IDS:
            s = scores_table.get(rt.id, {}).get(bid)
            if s is not None:
                line += f" {s:>10.3f}"
                row_scores.append(s)
            else:
                line += f" {'N/A':>10}"
        if row_scores:
            avg = sum(row_scores) / len(row_scores)
            line += f" {avg:>8.3f}"
        else:
            line += f" {'N/A':>8}"
        print(line)

    print("-" * 75)
    ref_line = f"{'Official Ref':<14} {'---':<7} {'---':<12}"
    ref_scores = []
    for bid in BENCHMARK_IDS:
        s = OFFICIAL_SCORES.get(bid)
        if s is not None:
            ref_line += f" {s:>10.3f}"
            ref_scores.append(s)
        else:
            ref_line += f" {'N/A':>10}"
    ref_avg = sum(ref_scores) / len(ref_scores) if ref_scores else 0
    ref_line += f" {ref_avg:>8.3f}"
    print(ref_line)
    print("=" * 75)

    # Delta from baseline
    baseline_scores = scores_table.get("baseline", {})
    if any(v is not None for v in baseline_scores.values()):
        print("\nDelta from baseline:")
        print(f"{'Runtime':<14} {'Bits':<7}", end="")
        for bid in BENCHMARK_IDS:
            print(f" {bid:>10}", end="")
        print(f" {'Avg':>8}")
        print("-" * 75)
        for rt in runtimes:
            if rt.id == "baseline":
                continue
            line = f"{rt.id:<14} {rt.bits:<7}"
            deltas = []
            for bid in BENCHMARK_IDS:
                s = scores_table.get(rt.id, {}).get(bid)
                b = baseline_scores.get(bid)
                if s is not None and b is not None and b > 0:
                    delta = s - b
                    pct = (delta / b) * 100
                    line += f" {pct:>+9.1f}%"
                    deltas.append(pct)
                else:
                    line += f" {'N/A':>10}"
            if deltas:
                avg_pct = sum(deltas) / len(deltas)
                line += f" {avg_pct:>+7.1f}%"
            print(line)
        print()

    print(f"Patch wall time: {total_time:.0f}s")
    print(f"Results saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
