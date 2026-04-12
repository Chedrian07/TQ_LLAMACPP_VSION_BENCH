#!/usr/bin/env python3
"""TQ-VLM-Bench main runner.

Usage:
  # Quick smoke (10 samples, baseline only, 3 VLM benchmarks)
  uv run python run_bench.py --num 10

  # Full P0 run (100 samples, all runtimes, 3 VLM parity benchmarks)
  uv run python run_bench.py --num 100

  # Specific runtimes
  uv run python run_bench.py --num 50 --runtimes baseline lcpp-kv-4 tq-4

  # Specific benchmarks
  uv run python run_bench.py --num 30 --benchmarks ai2d mmmu

  # All 11 benchmarks × all runtimes
  uv run python run_bench.py --num 100 --benchmarks all --runtimes all

  # Thinking model
  uv run python run_bench.py --num 30 --model qwen3_vl_2b_thinking

Runtimes are grouped into tiers:
  core    = baseline, lcpp-kv-8, lcpp-kv-4, tq-4, tq-K4V3
  tq-all  = core + tq-2, tq-2h, tq-3, tq-3h, tq-K4V2, tq-K3V2
  prod    = tqp-3, tqp-4, tqp-5
  all     = tq-all + prod  (lcpp-kv-2 excluded: upstream unsupported)

Benchmarks are grouped:
  p0      = ai2d, mmmu, mathvista  (parity evaluators, official comparison)
  vlm     = all 8 VLM benchmarks
  text    = mmlu, commonsenseqa, hellaswag
  all     = vlm + text

P0 benchmarks use parity metrics; all others use existing (approximate) metrics.
"""

from __future__ import annotations

import argparse
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

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_PROJECT = Path("/home/kch3dri4n/lab/TQ_LLAMACPP_VSION_BENCH")
_BUILD_BIN = str(_PROJECT / "llama.cpp/build/bin")
os.environ.setdefault("LD_LIBRARY_PATH", _BUILD_BIN)

BENCH_DIR = _PROJECT / "bench"
SERVER_BINARY = _PROJECT / "llama.cpp/build/bin/llama-server"
RESULTS_DIR = _PROJECT / "results/runs"

# ---------------------------------------------------------------------------
# Runtime / benchmark groups
# ---------------------------------------------------------------------------

RUNTIME_GROUPS: dict[str, list[str]] = {
    "core": ["baseline", "lcpp-kv-8", "lcpp-kv-4", "tq-4", "tq-K4V3"],
    "tq-all": [
        "baseline", "lcpp-kv-8", "lcpp-kv-4",
        "tq-2", "tq-2h", "tq-3", "tq-3h", "tq-4",
        "tq-K4V2", "tq-K4V3", "tq-K3V2",
    ],
    "prod": ["tqp-3", "tqp-4", "tqp-5"],
    "all": [
        "baseline", "lcpp-kv-8", "lcpp-kv-4",
        "tq-2", "tq-2h", "tq-3", "tq-3h", "tq-4",
        "tq-K4V2", "tq-K4V3", "tq-K3V2",
        "tqp-3", "tqp-4", "tqp-5",
    ],
}

BENCHMARK_GROUPS: dict[str, list[str]] = {
    "p0": ["ai2d", "mmmu", "mathvista"],
    "vlm": ["ai2d", "chartqa", "chartqapro", "docvqa", "mathvista", "mmmu",
            "ocrbench_v2", "textvqa"],
    "text": ["mmlu", "commonsenseqa", "hellaswag"],
    "all": ["ai2d", "chartqa", "chartqapro", "docvqa", "mathvista", "mmmu",
            "ocrbench_v2", "textvqa", "mmlu", "commonsenseqa", "hellaswag"],
}

# P0 benchmarks use parity metrics for official comparison
PARITY_METRICS: dict[str, str] = {
    "ai2d": "option_match",         # already near-parity
    "mmmu": "mmmu_official",
    "mathvista": "mathvista_official",
}

OFFICIAL_SCORES: dict[str, float] = {
    "ai2d": 0.804,
    "mmmu": 0.614,
    "mathvista": 0.736,
}

# TQ/prod runtimes: parallel=1 to avoid CUDA concurrency crashes
TQ_RUNTIME_PREFIXES = ("tq-", "tqp-")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_bench")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def force_kill_port(port: int) -> None:
    """Kill any process on port and wait until free."""
    try:
        result = subprocess.run(
            ["lsof", "-t", "-i", f"TCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [int(p) for p in result.stdout.strip().split() if p.strip().isdigit()]
        for pid in pids:
            if pid == os.getpid():
                continue
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
    time.sleep(0.5)


def get_git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5,
                           cwd=str(_PROJECT))
        return r.stdout.strip()
    except Exception:
        return "unknown"


def resolve_ids(args_list: list[str], groups: dict[str, list[str]]) -> list[str]:
    """Expand group names and flatten into an ordered unique list."""
    seen: set[str] = set()
    result: list[str] = []
    for item in args_list:
        expanded = groups.get(item, [item])
        for x in expanded:
            if x not in seen:
                seen.add(x)
                result.append(x)
    return result


def make_cell_key(model_id: str, runtime_id: str, benchmark_id: str) -> str:
    if model_id:
        return f"{model_id}:{runtime_id}:{benchmark_id}"
    return f"{runtime_id}:{benchmark_id}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TQ-VLM-Bench runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--num", type=int, required=True,
                        help="Number of samples per benchmark")
    parser.add_argument("--runtimes", nargs="+", default=["core"],
                        help="Runtime IDs or group names (core/tq-all/prod/all)")
    parser.add_argument("--benchmarks", nargs="+", default=["p0"],
                        help="Benchmark IDs or group names (p0/vlm/text/all)")
    parser.add_argument("--model", default="qwen3_vl_2b_instruct",
                        help="Model ID from models.yaml")
    parser.add_argument("--port", type=int, default=None,
                        help="Override server port (default: from models.yaml)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (auto-generated if omitted)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from existing JSON (skip completed cells)")
    args = parser.parse_args()

    from tq_bench.config import (
        BenchmarkConfig, ExperimentCell,
        load_benchmarks, load_models, load_runtimes,
    )
    from tq_bench.runner import BenchmarkRunner

    # Load configs
    runtimes_all = load_runtimes(BENCH_DIR / "configs/runtimes.yaml")
    benchmarks_all = load_benchmarks(BENCH_DIR / "configs/benchmarks.yaml")
    models = load_models(BENCH_DIR / "configs/models.yaml")

    runtimes_map = {rt.id: rt for rt in runtimes_all}
    benchmarks_map = {bm.id: bm for bm in benchmarks_all}

    if args.model not in models:
        logger.error("Unknown model '%s'. Available: %s", args.model, list(models.keys()))
        sys.exit(1)
    model_config = models[args.model]

    # Resolve port from model config or CLI
    port = args.port or model_config.port or 8080
    gpu_id = model_config.gpu_id

    # Resolve runtime and benchmark lists
    runtime_ids = resolve_ids(args.runtimes, RUNTIME_GROUPS)
    benchmark_ids = resolve_ids(args.benchmarks, BENCHMARK_GROUPS)

    runtimes = [runtimes_map[r] for r in runtime_ids if r in runtimes_map]
    benchmarks_raw = [benchmarks_map[b] for b in benchmark_ids if b in benchmarks_map]

    missing_rt = [r for r in runtime_ids if r not in runtimes_map]
    missing_bm = [b for b in benchmark_ids if b not in benchmarks_map]
    if missing_rt:
        logger.warning("Skipped unknown runtimes: %s", missing_rt)
    if missing_bm:
        logger.warning("Skipped unknown benchmarks: %s", missing_bm)

    if not runtimes or not benchmarks_raw:
        logger.error("No valid runtimes or benchmarks. Exiting.")
        sys.exit(1)

    # Build benchmark configs with sample count and metric overrides
    benchmarks: list[BenchmarkConfig] = []
    for bm in benchmarks_raw:
        metric = PARITY_METRICS.get(bm.id, bm.metric)
        benchmarks.append(BenchmarkConfig(
            id=bm.id,
            task_type=bm.task_type,
            sample_count=args.num,
            metric=metric,
            max_tokens=bm.max_tokens,
            parity_mode=bm.id in PARITY_METRICS,
            parity_metric=PARITY_METRICS.get(bm.id),
            parity_sample_count=bm.parity_sample_count,
        ))

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%m%d_%H%M")
        rt_tag = args.runtimes[0] if len(args.runtimes) == 1 else f"{len(runtimes)}rt"
        bm_tag = args.benchmarks[0] if len(args.benchmarks) == 1 else f"{len(benchmarks)}bm"
        output_path = RESULTS_DIR / f"bench_{rt_tag}_{bm_tag}_n{args.num}_{ts}.json"

    # Resume: load existing records to skip
    completed_keys: set[str] = set()
    resumed_records: list[dict] = []
    if args.resume:
        try:
            with open(args.resume) as f:
                prev = json.load(f)
            for rec in prev.get("records", []):
                if rec.get("status") == "ok":
                    key = make_cell_key(
                        rec.get("model_id", model_config.id),
                        rec["runtime_id"],
                        rec["benchmark_id"],
                    )
                    completed_keys.add(key)
                    resumed_records.append(rec)
            logger.info("Resumed %d completed cells from %s", len(completed_keys), args.resume)
        except Exception as e:
            logger.warning("Could not resume from %s: %s", args.resume, e)

    # Print plan
    total_cells = len(runtimes) * len(benchmarks)
    skip_count = sum(
        1
        for rt in runtimes
        for bm in benchmarks
        if make_cell_key(model_config.id, rt.id, bm.id) in completed_keys
    )

    logger.info("=" * 70)
    logger.info("TQ-VLM-Bench")
    logger.info("  Model:      %s (%s)", model_config.id, model_config.reasoning_mode)
    logger.info("  GPU:        %s", gpu_id)
    logger.info("  Runtimes:   %s", [rt.id for rt in runtimes])
    logger.info("  Benchmarks: %s", [bm.id for bm in benchmarks])
    logger.info("  Samples:    %d per benchmark", args.num)
    logger.info("  Cells:      %d total, %d to run, %d resumed",
                total_cells, total_cells - skip_count, skip_count)
    logger.info("  Port:       %d", port)
    logger.info("  Output:     %s", output_path)
    logger.info("=" * 70)

    # Create runner
    base_parallel = model_config.parallel_requests or 4
    runner = BenchmarkRunner(
        server_binary=SERVER_BINARY,
        default_port=port,
        request_timeout=180.0,
        max_retries=2,
        parallel_requests=base_parallel,
    )

    # Run matrix
    t_global = time.monotonic()
    all_records: list[dict] = list(resumed_records)
    scores: dict[str, dict[str, float | None]] = {rt.id: {} for rt in runtimes}

    # Fill in resumed scores
    for rec in resumed_records:
        if rec.get("model_id", model_config.id) == model_config.id:
            scores.setdefault(rec["runtime_id"], {})[rec["benchmark_id"]] = rec.get("score", 0)

    cell_num = 0
    for rt in runtimes:
        is_tq = any(rt.id.startswith(p) for p in TQ_RUNTIME_PREFIXES)
        n_parallel = 1 if is_tq else base_parallel

        for bm in benchmarks:
            cell_num += 1
            cell_key = make_cell_key(model_config.id, rt.id, bm.id)

            if cell_key in completed_keys:
                logger.info("[%d/%d] SKIP (resumed): %s × %s",
                            cell_num, total_cells, rt.id, bm.id)
                continue

            cell = ExperimentCell(runtime=rt, benchmark=bm, model_id=model_config.id)

            logger.info("")
            logger.info("=" * 60)
            logger.info("[%d/%d] %s × %s  (bits=%s, metric=%s, parallel=%d)",
                        cell_num, total_cells, rt.id, bm.id,
                        rt.bits, bm.metric, n_parallel)
            logger.info("=" * 60)

            force_kill_port(port)

            try:
                record = runner.run_cell(
                    cell, model_config,
                    gpu_id=gpu_id,
                    port=port, seed=args.seed,
                    parallel_requests=n_parallel,
                )
            except Exception as exc:
                logger.error("FAILED: %s × %s — %s", rt.id, bm.id, exc)
                from tq_bench.runner import RunRecord
                record = RunRecord(
                    runtime_id=rt.id,
                    benchmark_id=bm.id,
                    status="error",
                    model_id=model_config.id,
                    score=0.0,
                    notes=f"Exception: {exc}",
                )

            scores[rt.id][bm.id] = record.score
            rec_dict = record.to_dict()
            rec_dict["runtime_config"] = {
                "cache_type_k": rt.cache_type_k,
                "cache_type_v": rt.cache_type_v,
                "bits": rt.bits,
                "method": rt.method,
            }
            rec_dict["metric_used"] = bm.metric
            rec_dict["is_parity"] = bm.id in PARITY_METRICS
            all_records.append(rec_dict)

            logger.info("  => status=%s  score=%.3f  time=%.0fs  ok=%d/%d",
                        record.status, record.score or 0,
                        record.wall_time_seconds,
                        record.n_succeeded, record.n_samples)

            # Incremental save
            _save_output(output_path, args, model_config, port, runtimes, benchmarks,
                         scores, all_records, time.monotonic() - t_global)

    total_time = time.monotonic() - t_global

    # Final save
    _save_output(output_path, args, model_config, port, runtimes, benchmarks,
                 scores, all_records, total_time)

    # Print summary
    _print_summary(runtimes, benchmarks, scores, args.num, model_config, total_time)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _save_output(path, args, model_config, port, runtimes, benchmarks,
                 scores, records, wall_time):
    output = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
            "model": model_config.id,
            "reasoning_mode": model_config.reasoning_mode,
            "gpu_id": model_config.gpu_id,
            "port": port,
            "n_samples": args.num,
            "seed": args.seed,
            "runtimes": [rt.id for rt in runtimes],
            "benchmarks": [bm.id for bm in benchmarks],
            "parity_metrics": {bm.id: bm.metric for bm in benchmarks if bm.id in PARITY_METRICS},
            "official_reference": OFFICIAL_SCORES,
            "total_wall_time": round(wall_time, 1),
        },
        "scores": scores,
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)


def _print_summary(runtimes, benchmarks, scores, n_samples, model_config, total_time):
    bm_ids = [bm.id for bm in benchmarks]

    print(f"\n{'=' * 90}")
    print(f"TQ-VLM-Bench  (n={n_samples}, {model_config.id})")
    print(f"{'=' * 90}")

    # Header
    header = f"{'Runtime':<14} {'Bits':<7}"
    for bid in bm_ids:
        w = max(len(bid), 8)
        header += f" {bid:>{w}}"
    header += f" {'Avg':>8}"
    print(header)
    print("-" * 90)

    # Baseline row for delta computation
    baseline_scores = scores.get("baseline", {})

    for rt in runtimes:
        row = f"{rt.id:<14} {rt.bits:<7}"
        vals = []
        for bid in bm_ids:
            s = scores.get(rt.id, {}).get(bid)
            w = max(len(bid), 8)
            if s is None:
                row += f" {'---':>{w}}"
            else:
                vals.append(s)
                # Show delta from baseline
                bl = baseline_scores.get(bid)
                if rt.id == "baseline":
                    row += f" {s:>{w}.3f}"
                elif bl is not None and bl > 0:
                    delta = s - bl
                    row += f" {s:>{w}.3f}"
                else:
                    row += f" {s:>{w}.3f}"

        avg = sum(vals) / len(vals) if vals else 0
        row += f" {avg:>8.3f}"
        print(row)

    # Delta row
    if "baseline" in {rt.id for rt in runtimes} and len(runtimes) > 1:
        print("-" * 90)
        print("Delta from baseline:")
        for rt in runtimes:
            if rt.id == "baseline":
                continue
            row = f"  {rt.id:<12} {'':7}"
            deltas = []
            for bid in bm_ids:
                s = scores.get(rt.id, {}).get(bid)
                bl = baseline_scores.get(bid)
                w = max(len(bid), 8)
                if s is not None and bl is not None:
                    d = s - bl
                    deltas.append(d)
                    sign = "+" if d >= 0 else ""
                    row += f" {sign}{d:>{w-1}.3f}"
                else:
                    row += f" {'---':>{w}}"
            avg_d = sum(deltas) / len(deltas) if deltas else 0
            sign = "+" if avg_d >= 0 else ""
            row += f" {sign}{avg_d:>7.3f}"
            print(row)

    # Official comparison (P0 only)
    p0_in_run = [bid for bid in bm_ids if bid in OFFICIAL_SCORES]
    if p0_in_run and "baseline" in {rt.id for rt in runtimes}:
        print("-" * 90)
        print("Official comparison (baseline):")
        for bid in p0_in_run:
            ours = baseline_scores.get(bid)
            off = OFFICIAL_SCORES[bid]
            if ours is not None:
                gap = ours - off
                sign = "+" if gap >= 0 else ""
                status = "OK" if abs(gap) <= 0.05 else "GAP"
                print(f"  {bid:<12} ours={ours:.3f}  official={off:.3f}  "
                      f"delta={sign}{gap:.3f}  [{status}]")

    print("-" * 90)
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print()


if __name__ == "__main__":
    main()
