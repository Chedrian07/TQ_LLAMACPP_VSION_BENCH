#!/usr/bin/env python3
"""Parity evaluator smoke test: baseline × {AI2D, MMMU, MathVista} × n=10.

Runs each benchmark once with the PARITY metric (which needs metadata from
the runner), then re-scores raw predictions with the existing (approximate)
metric for side-by-side comparison.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
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
from tq_bench.evaluators import get_evaluator
from tq_bench.runner import BenchmarkRunner

BENCH_DIR = Path(__file__).resolve().parent
SERVER_BINARY = default_server_binary(_PROJECT)
OUTPUT_PATH = _PROJECT / "results" / "runs" / "parity_smoke_n10.json"

RUNTIME_ID = "baseline"
MODEL_ID = "qwen3_vl_2b_instruct"
N_SAMPLES = 10
PORT = 8080

OFFICIAL = {"ai2d": 0.804, "mmmu": 0.614, "mathvista": 0.736}

# (benchmark_id, parity_metric_for_runner, existing_metric_for_rescore)
CELLS = [
    ("ai2d",      "option_match",       "option_match"),
    ("mmmu",      "mmmu_official",       "option_match"),
    ("mathvista", "mathvista_official",  "mathvista_match"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logger = logging.getLogger("parity_smoke")


def force_kill_port(port: int) -> None:
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
    time.sleep(1.0)


def main():
    t0 = time.monotonic()

    runtimes_all = load_runtimes(BENCH_DIR / "configs" / "runtimes.yaml")
    benchmarks_all = load_benchmarks(BENCH_DIR / "configs" / "benchmarks.yaml")
    models = load_models(BENCH_DIR / "configs" / "models.yaml")

    runtimes_map = {rt.id: rt for rt in runtimes_all}
    benchmarks_map = {bm.id: bm for bm in benchmarks_all}
    model_config = models[MODEL_ID]
    rt = runtimes_map[RUNTIME_ID]
    port = model_config.port or PORT
    gpu_id = model_config.gpu_id
    parallel_requests = model_config.parallel_requests or 4

    logger.info("=" * 65)
    logger.info("PARITY SMOKE: baseline × 3 VLM benchmarks × n=%d", N_SAMPLES)
    logger.info("Model=%s gpu=%s port=%d parallel=%d", model_config.id, gpu_id, port, parallel_requests)
    logger.info("=" * 65)

    runner = BenchmarkRunner(
        server_binary=SERVER_BINARY,
        default_port=port,
        request_timeout=180.0,
        max_retries=2,
        parallel_requests=parallel_requests,
    )

    results = []

    for bm_id, parity_metric, existing_metric in CELLS:
        bm_orig = benchmarks_map[bm_id]

        # Run with PARITY metric so the runner passes metadata to the evaluator
        bm = BenchmarkConfig(
            id=bm_orig.id,
            task_type=bm_orig.task_type,
            sample_count=N_SAMPLES,
            metric=parity_metric,       # <-- parity evaluator
            max_tokens=bm_orig.max_tokens,
        )
        cell = ExperimentCell(runtime=rt, benchmark=bm, model_id=model_config.id)

        logger.info("")
        logger.info("-" * 55)
        logger.info("[%s] metric=%s, n=%d", bm_id, parity_metric, N_SAMPLES)
        logger.info("-" * 55)

        force_kill_port(port)

        try:
            record = runner.run_cell(
                cell,
                model_config,
                gpu_id=gpu_id,
                port=port,
                seed=42,
                parallel_requests=parallel_requests,
            )
        except Exception as exc:
            logger.error("FAILED: %s — %s", bm_id, exc)
            results.append({"benchmark": bm_id, "status": "error", "error": str(exc)})
            continue

        # Parity scores come from run_cell
        parity_scores = [sr.score for sr in record.sample_results]
        parity_avg = sum(parity_scores) / len(parity_scores) if parity_scores else 0.0

        # Re-score with EXISTING metric (doesn't need metadata)
        existing_ev = get_evaluator(existing_metric)
        existing_scores = []
        for sr in record.sample_results:
            try:
                es = existing_ev.score(sr.prediction, sr.reference)
            except Exception:
                es = 0.0
            existing_scores.append(es)
        existing_avg = sum(existing_scores) / len(existing_scores) if existing_scores else 0.0

        # Per-sample details
        samples = []
        for sr, es, ps in zip(record.sample_results, existing_scores, parity_scores):
            samples.append({
                "id": sr.sample_id,
                "pred": (sr.prediction or "")[:300],
                "ref": str(sr.reference)[:150],
                "existing": es,
                "parity": ps,
                "match": "=" if abs(es - ps) < 1e-6 else "DIFF",
            })

        results.append({
            "benchmark": bm_id,
            "status": record.status,
            "n_ok": record.n_succeeded,
            "n_total": record.n_samples,
            "existing_metric": existing_metric,
            "existing_score": round(existing_avg, 4),
            "parity_metric": parity_metric,
            "parity_score": round(parity_avg, 4),
            "official": OFFICIAL.get(bm_id),
            "wall_time": round(record.wall_time_seconds, 1),
            "samples": samples,
        })

        logger.info("  existing (%s):  %.3f", existing_metric, existing_avg)
        logger.info("  parity   (%s): %.3f", parity_metric, parity_avg)
        logger.info("  official:            %.3f", OFFICIAL.get(bm_id, 0))

        # Show per-sample diffs
        diffs = [(s["id"], s["existing"], s["parity"])
                 for s in samples if s["match"] == "DIFF"]
        if diffs:
            logger.info("  Divergences (%d/%d):", len(diffs), len(samples))
            for sid, e, p in diffs:
                logger.info("    %s: existing=%.2f parity=%.2f", sid, e, p)
        else:
            logger.info("  No divergences — evaluators agree on all %d samples", len(samples))

    total_time = time.monotonic() - t0

    output = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model_config.id,
            "runtime": RUNTIME_ID,
            "gpu_id": gpu_id,
            "port": port,
            "parallel_requests": parallel_requests,
            "n_samples": N_SAMPLES,
            "total_wall_time": round(total_time, 1),
        },
        "results": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Summary table
    print("\n" + "=" * 80)
    print(f"PARITY SMOKE TEST  (n={N_SAMPLES}, baseline f16, {model_config.description})")
    print("=" * 80)
    print(f"{'Benchmark':<12} {'Existing (approx)':<22} {'Parity (official)':<27} {'Official':>8} {'Time':>6}")
    print("-" * 80)
    for r in results:
        if r["status"] == "error":
            print(f"{r['benchmark']:<12} ERROR: {r.get('error', '')[:55]}")
            continue
        ex = f"{r['existing_metric']}: {r['existing_score']:.3f}"
        pa = f"{r['parity_metric']}: {r['parity_score']:.3f}"
        off = r.get("official", 0) or 0
        print(f"{r['benchmark']:<12} {ex:<22} {pa:<27} {off:>7.3f} {r['wall_time']:>5.0f}s")
    print("-" * 80)
    print(f"Total: {total_time:.0f}s")
    print(f"\nResults: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
