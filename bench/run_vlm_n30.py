#!/usr/bin/env python3
"""VLM benchmark runner: 5 runtimes x 3 VLM benchmarks x n=30.

Runtimes (MSE-only, no prod/QJL):
  - baseline (f16/f16)
  - lcpp-kv-4 (q4_0/q4_0)
  - tq-4 (turbo4/turbo4)
  - tq-K4V3 (turbo4/turbo3)
  - tq-3 (turbo3/turbo3)

Benchmarks:
  - ai2d (option_match)
  - mmmu (option_match)
  - mathvista (mathvista_match)

Official reference scores (Qwen3-VL-2B-Instruct):
  AI2D=0.804, MMMU=0.614, MathVista=0.736

Run 2: TQ runtimes use parallel=1 to avoid concurrency-related crashes.
       Removed external pkill to prevent race with server start in run_cell.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure LD_LIBRARY_PATH includes the llama.cpp build dir
_BUILD_BIN = "/home/kch3dri4n/lab/TQ_LLAMACPP_VSION_BENCH/llama.cpp/build/bin"
os.environ.setdefault("LD_LIBRARY_PATH", _BUILD_BIN)

from tq_bench.config import (
    BenchmarkConfig,
    ExperimentCell,
    load_benchmarks,
    load_models,
    load_runtimes,
)
from tq_bench.runner import BenchmarkRunner, RunRecord

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BENCH_DIR = Path("/home/kch3dri4n/lab/TQ_LLAMACPP_VSION_BENCH/bench")
SERVER_BINARY = Path("/home/kch3dri4n/lab/TQ_LLAMACPP_VSION_BENCH/llama.cpp/build/bin/llama-server")
OUTPUT_PATH = Path("/home/kch3dri4n/lab/TQ_LLAMACPP_VSION_BENCH/results/runs/vlm_final_n30.json")

RUNTIME_IDS = ["baseline", "lcpp-kv-4", "tq-4", "tq-K4V3", "tq-3"]
BENCHMARK_IDS = ["ai2d", "mmmu", "mathvista"]
MODEL_ID = "qwen3_vl_2b_instruct"
N_SAMPLES = 30
PORT = 8080
# baseline/lcpp use parallel=4; TQ runtimes use parallel=1 to avoid
# concurrency-related CUDA crashes observed in run 1
PARALLEL_SAFE = 4
PARALLEL_TQ = 1

OFFICIAL_SCORES = {
    "ai2d": 0.804,
    "mmmu": 0.614,
    "mathvista": 0.736,
}

# TQ runtimes that need single-request mode
TQ_RUNTIME_IDS = {"tq-4", "tq-K4V3", "tq-3", "tq-2", "tq-2h", "tq-3h"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vlm_bench")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd="/home/kch3dri4n/lab/TQ_LLAMACPP_VSION_BENCH",
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def wait_port_free(port: int, timeout: float = 15.0) -> None:
    """Block until no process is listening on *port* (or timeout).

    Uses lsof to check if any process has the port in LISTEN state.
    Falls back to socket connect check.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Check with lsof first (most reliable)
        try:
            result = subprocess.run(
                ["lsof", "-t", "-i", f"TCP:{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=3,
            )
            pids = [p.strip() for p in result.stdout.strip().split() if p.strip().isdigit()]
            if not pids:
                return  # Port is free
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        time.sleep(1.0)
    logger.warning("Port %d still occupied after %.0fs wait", port, timeout)


def force_kill_port(port: int) -> None:
    """Kill anything on *port* and wait for it to be free."""
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
            time.sleep(1.0)
            # Check if still alive, escalate to SIGKILL
            for pid in pids:
                try:
                    os.kill(pid, 0)  # Check if alive
                    os.kill(pid, signal.SIGKILL)
                    logger.info("SIGKILL pid=%d", pid)
                except OSError:
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    wait_port_free(port, timeout=10.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_global_start = time.monotonic()

    # Load configs
    runtimes_all = load_runtimes(BENCH_DIR / "configs" / "runtimes.yaml")
    benchmarks_all = load_benchmarks(BENCH_DIR / "configs" / "benchmarks.yaml")
    models = load_models(BENCH_DIR / "configs" / "models.yaml")

    runtimes_map = {rt.id: rt for rt in runtimes_all}
    benchmarks_map = {bm.id: bm for bm in benchmarks_all}
    model_config = models[MODEL_ID]
    port = model_config.port or PORT
    gpu_id = model_config.gpu_id

    # Filter to requested runtimes and benchmarks
    runtimes = [runtimes_map[rid] for rid in RUNTIME_IDS if rid in runtimes_map]
    benchmarks = [benchmarks_map[bid] for bid in BENCHMARK_IDS if bid in benchmarks_map]

    missing_rt = [rid for rid in RUNTIME_IDS if rid not in runtimes_map]
    missing_bm = [bid for bid in BENCHMARK_IDS if bid not in benchmarks_map]
    if missing_rt:
        logger.warning("Missing runtimes: %s", missing_rt)
    if missing_bm:
        logger.warning("Missing benchmarks: %s", missing_bm)

    logger.info("=" * 70)
    logger.info("VLM Benchmark: %d runtimes x %d benchmarks x n=%d",
                len(runtimes), len(benchmarks), N_SAMPLES)
    logger.info("Runtimes: %s", [rt.id for rt in runtimes])
    logger.info("Benchmarks: %s", [bm.id for bm in benchmarks])
    logger.info("Model: %s", model_config.id)
    logger.info("GPU: %s  Port: %d", gpu_id, port)
    logger.info("Server: %s", SERVER_BINARY)
    logger.info("=" * 70)

    # Create runner with safe defaults
    runner = BenchmarkRunner(
        server_binary=SERVER_BINARY,
        default_port=port,
        request_timeout=180.0,
        max_retries=2,
        parallel_requests=PARALLEL_SAFE,
    )

    # Run all cells
    all_records: list[dict] = []
    scores_table: dict[str, dict[str, float | None]] = {
        rt.id: {} for rt in runtimes
    }

    total_cells = len(runtimes) * len(benchmarks)
    cell_num = 0

    for rt in runtimes:
        # Determine parallel level: TQ runtimes use 1 to avoid CUDA crashes
        is_tq = rt.id in TQ_RUNTIME_IDS
        n_parallel = PARALLEL_TQ if is_tq else PARALLEL_SAFE

        for bm_orig in benchmarks:
            cell_num += 1

            # Override sample_count to N_SAMPLES
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
            logger.info("[%d/%d] %s x %s (bits=%s, K=%s, V=%s, parallel=%d)",
                        cell_num, total_cells, rt.id, bm.id,
                        rt.bits, rt.cache_type_k, rt.cache_type_v,
                        n_parallel)
            logger.info("=" * 60)

            # Ensure port is free before starting the cell.
            # run_cell() does its own _kill_existing_on_port() but the
            # 0.5s wait there is not always sufficient.
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
                logger.error("Cell %s x %s failed with exception: %s",
                             rt.id, bm.id, exc)
                record = RunRecord(
                    runtime_id=rt.id,
                    benchmark_id=bm.id,
                    status="error",
                    model_id=model_config.id,
                    score=0.0,
                    notes=f"Exception: {exc}",
                )

            scores_table[rt.id][bm.id] = record.score
            rec_dict = record.to_dict()
            rec_dict["runtime_config"] = {
                "cache_type_k": rt.cache_type_k,
                "cache_type_v": rt.cache_type_v,
                "bits": rt.bits,
                "method": rt.method,
            }
            all_records.append(rec_dict)

            logger.info("  => status=%s, score=%.4f, time=%.1fs, ok=%d/%d",
                        record.status, record.score or 0,
                        record.wall_time_seconds,
                        record.n_succeeded, record.n_samples)

    total_time = time.monotonic() - t_global_start

    # Build output JSON
    output = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
            "model": model_config.id,
            "model_description": model_config.description,
            "gpu_id": gpu_id,
            "port": port,
            "runtimes_tested": RUNTIME_IDS,
            "benchmarks_tested": BENCHMARK_IDS,
            "n_samples": N_SAMPLES,
            "parallel_safe": PARALLEL_SAFE,
            "parallel_tq": PARALLEL_TQ,
            "total_wall_time_seconds": round(total_time, 1),
            "official_reference_scores": OFFICIAL_SCORES,
        },
        "scores_summary": scores_table,
        "records": all_records,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved to %s", OUTPUT_PATH)

    # Print summary table
    print("\n" + "=" * 75)
    print("VLM BENCHMARK RESULTS (n=%d, %s)" % (N_SAMPLES, model_config.description))
    print("=" * 75)

    header = f"{'Runtime':<14} {'Bits':<7}"
    for bid in BENCHMARK_IDS:
        header += f" {bid:>10}"
    header += f" {'Avg':>8}"
    print(header)
    print("-" * 75)

    for rt in runtimes:
        row_scores = []
        line = f"{rt.id:<14} {rt.bits:<7}"
        for bid in BENCHMARK_IDS:
            s = scores_table[rt.id].get(bid)
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
    ref_line = f"{'Official Ref':<14} {'---':<7}"
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
    print(f"Total wall time: {total_time/60:.1f} minutes")
    print(f"Results saved: {OUTPUT_PATH}")
    print()

    # Print delta from baseline
    baseline_scores = scores_table.get("baseline", {})
    if any(baseline_scores.get(bid) is not None for bid in BENCHMARK_IDS):
        print("Delta from baseline:")
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
                s = scores_table[rt.id].get(bid)
                b = baseline_scores.get(bid)
                if s is not None and b is not None and b > 0:
                    delta = s - b
                    pct = (delta / b) * 100
                    line += f" {pct:>+9.1f}%"
                    deltas.append(delta)
                else:
                    line += f" {'N/A':>10}"
            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                avg_base = sum(baseline_scores.get(bid, 0) for bid in BENCHMARK_IDS) / len(BENCHMARK_IDS)
                if avg_base > 0:
                    avg_pct = (avg_delta / avg_base) * 100
                    line += f" {avg_pct:>+7.1f}%"
            print(line)
        print()


if __name__ == "__main__":
    main()
