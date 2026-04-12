#!/usr/bin/env python3
"""Smoke test: run a small subset of the benchmark matrix to verify everything works.

Usage:
    python3 smoke_test.py                        # default: all configured models, baseline + tq-3 + tq-4, commonsenseqa, n=3
    python3 smoke_test.py --runtime baseline      # single runtime
    python3 smoke_test.py --runtime baseline --runtime tq-4
    python3 smoke_test.py --benchmark mmlu --n 5
    python3 smoke_test.py --model qwen3_vl_2b_instruct
    python3 smoke_test.py --model qwen3_vl_2b_instruct --gpu 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

# Ensure bench/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from tq_bench.config import (
    BenchmarkConfig,
    ExperimentCell,
    ModelConfig,
    RuntimeConfig,
    load_benchmarks,
    load_models,
    load_runtimes,
)
from tq_bench.runner import BenchmarkRunner, RunRecord

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_BENCH_DIR = Path(__file__).parent
_PROJECT_ROOT = _BENCH_DIR.parent
_DEFAULT_BINARY = _PROJECT_ROOT / "llama.cpp" / "build" / "bin" / "llama-server"
DEFAULT_RUNTIMES = ["baseline", "tq-3", "tq-4"]
DEFAULT_BENCHMARKS = ["commonsenseqa"]
DEFAULT_N_SAMPLES = 3
DEFAULT_PORT = 8080


def _setup_ld_library_path() -> None:
    """Ensure CUDA shared libs from the llama.cpp build tree are findable."""
    cuda_lib_dir = str(_PROJECT_ROOT / "llama.cpp" / "build" / "bin")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if cuda_lib_dir not in existing:
        os.environ["LD_LIBRARY_PATH"] = (
            f"{cuda_lib_dir}:{existing}" if existing else cuda_lib_dir
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TQ-VLM-Bench smoke test")
    parser.add_argument(
        "--runtime",
        action="append",
        dest="runtimes",
        help="Runtime ID(s) to test (repeatable). Default: baseline, tq-3, tq-4",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        dest="benchmarks",
        help="Benchmark ID(s) to test (repeatable). Default: commonsenseqa",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of samples per cell (default: {DEFAULT_N_SAMPLES})",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID to pin the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port for llama-server (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=_DEFAULT_BINARY,
        help="Path to llama-server binary",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model ID(s) from models.yaml (repeatable). Default: all configured models.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    _setup_ld_library_path()

    # ---- Load configs -------------------------------------------------------
    configs_dir = _BENCH_DIR / "configs"
    all_runtimes = {r.id: r for r in load_runtimes(configs_dir / "runtimes.yaml")}
    all_benchmarks = {b.id: b for b in load_benchmarks(configs_dir / "benchmarks.yaml")}
    all_models = load_models(configs_dir / "models.yaml")

    runtime_ids = args.runtimes or DEFAULT_RUNTIMES
    benchmark_ids = args.benchmarks or DEFAULT_BENCHMARKS
    model_ids = args.models or sorted(all_models.keys())

    # Validate IDs
    for rid in runtime_ids:
        if rid not in all_runtimes:
            print(f"ERROR: Unknown runtime '{rid}'. Available: {sorted(all_runtimes)}")
            return 1
    for bid in benchmark_ids:
        if bid not in all_benchmarks:
            print(f"ERROR: Unknown benchmark '{bid}'. Available: {sorted(all_benchmarks)}")
            return 1
    for mid in model_ids:
        if mid not in all_models:
            print(f"ERROR: Unknown model '{mid}'. Available: {sorted(all_models)}")
            return 1
    if len(model_ids) > 1 and (args.gpu is not None or args.port != DEFAULT_PORT):
        print("ERROR: --gpu/--port overrides only work with a single selected model.")
        return 1

    selected_models = [all_models[mid] for mid in model_ids]
    n_samples = args.n

    print(f"Smoke test configuration:")
    print(f"  Binary:     {args.binary}")
    print(f"  Models:     {[m.id for m in selected_models]}")
    print(f"  Runtimes:   {runtime_ids}")
    print(f"  Benchmarks: {benchmark_ids}")
    print(f"  Samples:    {n_samples}")
    if len(selected_models) == 1:
        model = selected_models[0]
        print(f"  GPU:        {args.gpu if args.gpu is not None else model.gpu_id}")
        print(
            f"  Port:       "
            f"{args.port if args.port != DEFAULT_PORT else (model.port or DEFAULT_PORT)}"
        )
    else:
        print("  GPU/Port:   per-model from configs/models.yaml")
    print()

    # ---- Run cells -----------------------------------------------------------
    results: list[RunRecord] = []
    total_cells = len(selected_models) * len(runtime_ids) * len(benchmark_ids)
    runner_cache: dict[tuple[int, int], BenchmarkRunner] = {}

    cell_idx = 0
    for model_config in selected_models:
        model_gpu = args.gpu if len(selected_models) == 1 else model_config.gpu_id
        model_port = (
            args.port
            if len(selected_models) == 1 and args.port != DEFAULT_PORT
            else (model_config.port or DEFAULT_PORT)
        )
        model_parallel = model_config.parallel_requests or 4

        runner_key = (model_port, model_parallel)
        runner = runner_cache.get(runner_key)
        if runner is None:
            runner = BenchmarkRunner(
                server_binary=args.binary,
                default_port=model_port,
                request_timeout=120.0,
                max_retries=2,
                max_tokens=256,
                parallel_requests=model_parallel,
            )
            runner_cache[runner_key] = runner

        for rid, bid in ((r, b) for r in runtime_ids for b in benchmark_ids):
            cell_idx += 1
            rt = all_runtimes[rid]
            bm = all_benchmarks[bid]

            bm_smoke = replace(bm, sample_count=n_samples)
            cell = ExperimentCell(
                runtime=rt,
                benchmark=bm_smoke,
                model_id=model_config.id,
            )

            print(f"\n{'=' * 60}")
            print(
                f"[{cell_idx}/{total_cells}] "
                f"{model_config.id} :: {rid} x {bid} (n={n_samples})"
            )
            print(
                f"  cache-type-k: {rt.cache_type_k}  "
                f"cache-type-v: {rt.cache_type_v}  bits: {rt.bits}"
            )
            print(
                f"  gpu={model_gpu}  port={model_port}  parallel={model_parallel}"
            )
            print(f"{'=' * 60}")

            record = runner.run_cell(
                cell,
                model_config,
                gpu_id=model_gpu,
                port=model_port,
                parallel_requests=model_parallel,
            )
            results.append(record)

            if record.status == "ok":
                print(
                    f"  -> PASS  score={record.score:.3f}  "
                    f"time={record.wall_time_seconds:.1f}s"
                )
            else:
                print(
                    f"  -> {record.status.upper()}  score={record.score:.3f}  "
                    f"time={record.wall_time_seconds:.1f}s"
                )
                if record.notes:
                    print(f"     {record.notes}")

    # ---- Summary table -------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SMOKE TEST RESULTS")
    print(f"{'=' * 70}")
    print(
        f"{'Model':>24s}  {'Runtime':>14s}  {'Benchmark':>15s}  "
        f"{'Status':>13s}  {'Score':>6s}  {'Time':>7s}  {'Ok/N':>5s}"
    )
    print(
        f"{'-' * 24}  {'-' * 14}  {'-' * 15}  "
        f"{'-' * 13}  {'-' * 6}  {'-' * 7}  {'-' * 5}"
    )

    all_pass = True
    for r in results:
        status_str = "PASS" if r.status == "ok" else r.status.upper()
        if r.status != "ok":
            all_pass = False
        print(
            f"{r.model_id:>24s}  {r.runtime_id:>14s}  {r.benchmark_id:>15s}  {status_str:>13s}  "
            f"{r.score:6.3f}  {r.wall_time_seconds:6.1f}s  "
            f"{r.n_succeeded}/{r.n_samples}"
        )

    print(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME FAILURES'}")

    # ---- Save results to JSON ------------------------------------------------
    results_dir = _PROJECT_ROOT / "results" / "smoke"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"smoke_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
