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

  # Single model
  uv run python run_bench.py --num 30 --model qwen3_vl_2b_thinking

  # Single model on a specific GPU
  uv run python run_bench.py --num 30 --model qwen3_vl_2b_instruct --gpu 0

  # Single model with an explicit request parallelism
  uv run python run_bench.py --num 30 --model qwen3_vl_2b_instruct --parallel 2

  # Apply a quantized model variant
  uv run python run_bench.py --num 30 --model qwen3_vl_2b_instruct --model-quant q4_k_m

  # Mixed-model dual-GPU run (default when --model is omitted)
  uv run python run_bench.py --num 50 --runtimes tq-all

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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from tq_bench.config import ExecutionProfile
from tq_bench.env import default_server_binary, prepend_ld_library_path, project_root_from

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_PROJECT = project_root_from(__file__)
_DEFAULT_BUILD_BIN = _PROJECT / "llama.cpp" / "build" / "bin"
prepend_ld_library_path(_DEFAULT_BUILD_BIN)

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_SERVER_BINARY = default_server_binary(_PROJECT)
DEFAULT_KV_DUMP_BINARY = _DEFAULT_BUILD_BIN / "llama-kv-dump"
RESULTS_DIR = _PROJECT / "results/runs"
LOGS_DIR = _PROJECT / "logs"
PROFILES_YAML = BENCH_DIR / "configs/profiles.yaml"

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

# TQ/prod runtimes default to parallel=1 to avoid CUDA concurrency crashes.
# An explicit CLI --parallel override is allowed to bypass this default.
TQ_RUNTIME_PREFIXES = ("tq-", "tqp-")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s(%(threadName)s): %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy HF dataset HTTP chatter (hundreds of HEAD/GET per load)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
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


def _apply_profile(model, profile: ExecutionProfile) -> object:
    updates = {}
    if profile.gpu_id is not None:
        updates["gpu_id"] = profile.gpu_id
    if profile.port is not None:
        updates["port"] = profile.port
    if profile.parallel_requests is not None:
        updates["parallel_requests"] = profile.parallel_requests
    if not updates:
        return model
    return replace(model, **updates)


def _resolve_path_override(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


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
    parser.add_argument("--model", action="append", dest="models", default=None,
                        help="Model ID from models.yaml (repeatable). Default: all configured models.")
    parser.add_argument("--profile", default="local",
                        help="Execution profile ID from configs/profiles.yaml (default: local).")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Override GPU device id for a single selected model.")
    parser.add_argument("--parallel", type=int, default=None,
                        help="Override parallel request count for a single selected model.")
    parser.add_argument("--model-quant", choices=["bf16", "q8_0", "q4_k_m"], default="bf16",
                        help="Select which model weight quantization variant to use.")
    parser.add_argument("--port", type=int, default=None,
                        help="Override server port (default: from models.yaml)")
    parser.add_argument("--server-binary", type=str, default=None,
                        help="Path to llama-server when using a non-default build directory.")
    parser.add_argument("--kv-dump-binary", type=str, default=None,
                        help="Path to llama-kv-dump when using a non-default build directory.")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory for auto-generated output JSONs.")
    parser.add_argument("--slot-save-path", type=str, default=None,
                        help="Override llama-server --slot-save-path.")
    parser.add_argument("--cache-ram", type=int, default=None,
                        help="Override llama-server --cache-ram.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override llama-server -b.")
    parser.add_argument("--ubatch-size", type=int, default=None,
                        help="Override llama-server -ub.")
    parser.add_argument("--n-gpu-layers", type=int, default=None,
                        help="Override llama-server -ngl.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (auto-generated if omitted)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from existing JSON (skip completed cells)")
    parser.add_argument("--logs-dir", type=str, default=None,
                        help="Override logs directory (default: PROJECT_ROOT/logs). "
                             "Use this to persist KV dumps/analysis to Google Drive on Colab.")
    parser.add_argument("--kv-dump", action="store_true", default=False,
                        help="After benchmarks, extract KV dumps + run analysis for each runtime")
    parser.add_argument("--kv-dump-image", type=str, default=None,
                        help="Image path for VLM KV dump probe (default: first image from ai2d dataset)")
    parser.add_argument("--kv-dump-prompt", type=str, default=None,
                        help="Prompt for KV dump probe (default: 'Describe this image in detail.')")
    args = parser.parse_args()

    # Allow overriding LOGS_DIR from the command line (e.g. to persist KV
    # dumps/analysis to Google Drive on Colab).
    global LOGS_DIR  # noqa: PLW0603
    if args.logs_dir is not None:
        LOGS_DIR = Path(args.logs_dir).expanduser().resolve()
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    from tq_bench.config import (
        BenchmarkConfig, ExperimentCell,
        load_benchmarks, load_models, load_profiles, load_runtimes,
    )
    from tq_bench.runner import BenchmarkRunner, RunRecord

    # Load configs
    runtimes_all = load_runtimes(BENCH_DIR / "configs/runtimes.yaml")
    benchmarks_all = load_benchmarks(BENCH_DIR / "configs/benchmarks.yaml")
    models = load_models(BENCH_DIR / "configs/models.yaml")
    profiles = load_profiles(PROFILES_YAML)
    if args.profile not in profiles:
        logger.error("Unknown profile '%s'. Available: %s", args.profile, sorted(profiles.keys()))
        sys.exit(1)
    profile = profiles[args.profile]

    runtimes_map = {rt.id: rt for rt in runtimes_all}
    benchmarks_map = {bm.id: bm for bm in benchmarks_all}

    selected_model_ids = args.models or list(models.keys())
    selected_models = []
    seen_models: set[str] = set()
    for model_id in selected_model_ids:
        if model_id in seen_models:
            continue
        if model_id not in models:
            logger.error("Unknown model '%s'. Available: %s", model_id, list(models.keys()))
            sys.exit(1)
        selected_models.append(_apply_profile(models[model_id], profile))
        seen_models.add(model_id)

    if args.model_quant != "bf16":
        resolved_models = []
        for model in selected_models:
            quant_key = args.model_quant.lower()
            quant_path = model.quantized_model_paths.get(quant_key)
            if quant_path is None:
                logger.error(
                    "Model '%s' does not define a '%s' quantized path in models.yaml.",
                    model.id,
                    args.model_quant,
                )
                sys.exit(1)
            if not quant_path.exists():
                logger.error(
                    "Quantized model file not found for '%s' (%s): %s",
                    model.id,
                    args.model_quant,
                    quant_path,
                )
                sys.exit(1)
            resolved_models.append(
                replace(
                    model,
                    model_path=quant_path,
                    model_quantization=quant_key,
                    description=f"{model.description} [{args.model_quant}]",
                )
            )
        selected_models = resolved_models

    if len(selected_models) > 1 and args.port is not None:
        logger.error("--port override is only valid for a single selected model.")
        sys.exit(1)
    if len(selected_models) > 1 and args.gpu is not None:
        logger.error("--gpu override is only valid for a single selected model. Use --model MODEL --gpu N.")
        sys.exit(1)
    if len(selected_models) > 1 and args.parallel is not None:
        logger.error("--parallel override is only valid for a single selected model. Use --model MODEL --parallel N.")
        sys.exit(1)
    if (
        len(selected_models) > 1
        and args.profile != "local"
        and any(
            value is not None
            for value in (profile.gpu_id, profile.port, profile.parallel_requests)
        )
    ):
        logger.error(
            "--profile %s applies single-model execution overrides. Select one --model for this profile.",
            args.profile,
        )
        sys.exit(1)
    if len(selected_models) == 1 and args.gpu is not None:
        selected_models[0] = replace(selected_models[0], gpu_id=args.gpu)
    if len(selected_models) == 1 and args.port is not None:
        selected_models[0] = replace(selected_models[0], port=args.port)
    if args.parallel is not None:
        if args.parallel < 1:
            logger.error("--parallel must be >= 1.")
            sys.exit(1)
        selected_models[0] = replace(selected_models[0], parallel_requests=args.parallel)

    server_binary = _resolve_path_override(args.server_binary) or DEFAULT_SERVER_BINARY
    kv_dump_binary = _resolve_path_override(args.kv_dump_binary) or DEFAULT_KV_DUMP_BINARY

    if not server_binary.exists():
        logger.error(
            "llama-server binary not found: %s. Build llama.cpp or pass --server-binary.",
            server_binary,
        )
        sys.exit(1)
    if args.kv_dump and not kv_dump_binary.exists():
        logger.error(
            "llama-kv-dump binary not found: %s. Build llama-kv-dump or pass --kv-dump-binary.",
            kv_dump_binary,
        )
        sys.exit(1)

    prepend_ld_library_path(server_binary.parent)
    prepend_ld_library_path(kv_dump_binary.parent)

    results_dir = _resolve_path_override(args.results_dir) or profile.results_dir or RESULTS_DIR
    slot_save_path = _resolve_path_override(args.slot_save_path) or profile.slot_save_path or Path("./kvcache")
    cache_ram = (
        args.cache_ram
        if args.cache_ram is not None
        else profile.cache_ram if profile.cache_ram is not None else 16384
    )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else profile.batch_size if profile.batch_size is not None else 512
    )
    ubatch_size = (
        args.ubatch_size
        if args.ubatch_size is not None
        else profile.ubatch_size if profile.ubatch_size is not None else 512
    )
    n_gpu_layers = (
        args.n_gpu_layers
        if args.n_gpu_layers is not None
        else profile.n_gpu_layers if profile.n_gpu_layers is not None else 99
    )
    no_warmup = profile.no_warmup if profile.no_warmup is not None else True
    no_mmap = profile.no_mmap if profile.no_mmap is not None else True

    if cache_ram < 0:
        logger.error("--cache-ram must be >= 0.")
        sys.exit(1)
    if batch_size < 1:
        logger.error("--batch-size must be >= 1.")
        sys.exit(1)
    if ubatch_size < 1:
        logger.error("--ubatch-size must be >= 1.")
        sys.exit(1)
    server_meta = {
        "server_binary": str(server_binary),
        "kv_dump_binary": str(kv_dump_binary),
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "n_gpu_layers": n_gpu_layers,
        "cache_ram": cache_ram,
        "slot_save_path": str(slot_save_path),
        "no_warmup": no_warmup,
        "no_mmap": no_mmap,
    }

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
        if len(selected_models) == 1:
            model_tag = selected_models[0].id
            if selected_models[0].model_quantization != "bf16":
                model_tag += f"_{selected_models[0].model_quantization}"
        else:
            model_tag = f"{len(selected_models)}models"
        output_path = results_dir / f"bench_{model_tag}_{rt_tag}_{bm_tag}_n{args.num}_{ts}.json"

    # Resume: load existing records to skip
    completed_keys: set[str] = set()
    resumed_records: list[dict] = []
    selected_model_id_set = {model.id for model in selected_models}
    if args.resume:
        try:
            with open(args.resume) as f:
                prev = json.load(f)
            for rec in prev.get("records", []):
                rec_model_id = rec.get("model_id")
                if rec_model_id is None and len(selected_models) == 1:
                    rec_model_id = selected_models[0].id
                if rec_model_id not in selected_model_id_set:
                    continue
                if rec.get("status") == "ok":
                    key = make_cell_key(
                        rec_model_id or "",
                        rec["runtime_id"],
                        rec["benchmark_id"],
                    )
                    completed_keys.add(key)
                rec["model_id"] = rec_model_id
                resumed_records.append(rec)
            logger.info("Resumed %d completed cells from %s", len(completed_keys), args.resume)
        except Exception as e:
            logger.warning("Could not resume from %s: %s", args.resume, e)

    # Print plan
    total_cells = len(selected_models) * len(runtimes) * len(benchmarks)
    skip_count = sum(
        1
        for model in selected_models
        for rt in runtimes
        for bm in benchmarks
        if make_cell_key(model.id, rt.id, bm.id) in completed_keys
    )

    logger.info("=" * 70)
    logger.info("TQ-VLM-Bench")
    logger.info("  Profile:    %s", args.profile)
    logger.info(
        "  Models:     %s",
        [
            (
                f"{m.id}(quant={m.model_quantization}, mode={m.reasoning_mode}, gpu={m.gpu_id}, "
                f"port={m.port or 8080}, parallel={m.parallel_requests or 4}, "
                f"temp={m.temperature if m.temperature is not None else 0.0}, "
                f"top_p={m.top_p}, top_k={m.top_k}, min_p={m.min_p}, "
                f"repeat_penalty={m.repeat_penalty}, presence_penalty={m.presence_penalty}, "
                f"sampling_seed={m.sampling_seed})"
            )
            for m in selected_models
        ],
    )
    logger.info(
        "  Server:     binary=%s kv_dump=%s batch=%d ubatch=%d ngl=%d cache_ram=%d slot_save_path=%s "
        "no_warmup=%s no_mmap=%s",
        server_binary,
        kv_dump_binary,
        batch_size,
        ubatch_size,
        n_gpu_layers,
        cache_ram,
        slot_save_path,
        no_warmup,
        no_mmap,
    )
    logger.info("  Runtimes:   %s", [rt.id for rt in runtimes])
    logger.info("  Benchmarks: %s", [bm.id for bm in benchmarks])
    logger.info("  Samples:    %d per benchmark", args.num)
    logger.info("  Cells:      %d total, %d to run, %d resumed",
                total_cells, total_cells - skip_count, skip_count)
    logger.info("  Output:     %s", output_path)
    logger.info("=" * 70)

    # Run matrix
    t_global = time.monotonic()
    all_records: list[dict] = list(resumed_records)
    scores: dict[str, dict[str, dict[str, float | None]]] = {
        model.id: {rt.id: {} for rt in runtimes}
        for model in selected_models
    }
    statuses: dict[str, dict[str, dict[str, str | None]]] = {
        model.id: {rt.id: {} for rt in runtimes}
        for model in selected_models
    }

    # Fill in resumed scores
    for rec in resumed_records:
        rec_model_id = rec.get("model_id")
        if rec_model_id in scores:
            score_val = rec.get("score") if rec.get("status") == "ok" else None
            scores[rec_model_id].setdefault(rec["runtime_id"], {})[rec["benchmark_id"]] = score_val
            statuses[rec_model_id].setdefault(rec["runtime_id"], {})[rec["benchmark_id"]] = rec.get("status")

    state_lock = threading.Lock()
    progress_counter = {"done": len(completed_keys)}

    # KV dump: resolve probe sample once (reused across all runtimes)
    kv_probe_image_path: Path | None = None
    kv_probe_prompt: str = "Describe this image in detail."
    if args.kv_dump:
        kv_probe_prompt = args.kv_dump_prompt or kv_probe_prompt
        if args.kv_dump_image:
            kv_probe_image_path = Path(args.kv_dump_image).expanduser().resolve()
        else:
            kv_probe_image_path = _resolve_kv_probe_image()
        if kv_probe_image_path is None:
            kv_probe_prompt = "Explain the concept of quantization in neural networks."
            logger.info("KV dump: no image available, using text-only probe.")
        else:
            logger.info("KV dump: probe image=%s", kv_probe_image_path)

    kv_ts = datetime.now().strftime("%m%d_%H%M")
    kv_dump_paths: dict[str, dict[str, Path]] = {}  # model_id -> {rt_id -> dump_dir}
    analysis_futures: list = []
    analysis_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kv-analysis") if args.kv_dump else None

    def run_model_queue(lane_idx, model_config):
        threading.current_thread().name = f"lane-{model_config.id}"

        model_port = model_config.port or 8080
        model_gpu_id = model_config.gpu_id
        base_parallel = model_config.parallel_requests or 4
        runner = BenchmarkRunner(
            server_binary=server_binary,
            default_port=model_port,
            request_timeout=model_config.request_timeout or 180.0,
            max_retries=2,
            parallel_requests=base_parallel,
            batch_size=batch_size,
            ubatch_size=ubatch_size,
            n_gpu_layers=n_gpu_layers,
            cache_ram=cache_ram,
            slot_save_path=slot_save_path,
            no_warmup=no_warmup,
            no_mmap=no_mmap,
        )

        for rt in runtimes:
            is_tq = any(rt.id.startswith(p) for p in TQ_RUNTIME_PREFIXES)
            n_parallel = 1 if (is_tq and args.parallel is None) else base_parallel

            for bm in benchmarks:
                cell_key = make_cell_key(model_config.id, rt.id, bm.id)

                if cell_key in completed_keys:
                    with state_lock:
                        progress_counter["done"] += 1
                        logger.info(
                            "[%d/%d] SKIP (resumed): %s :: %s × %s",
                            progress_counter["done"], total_cells,
                            model_config.id, rt.id, bm.id,
                        )
                    continue

                cell = ExperimentCell(
                    runtime=rt,
                    benchmark=bm,
                    model_id=model_config.id,
                )

                logger.info("")
                logger.info("=" * 60)
                logger.info(
                    "[lane=%s gpu=%s port=%d] %s × %s  (bits=%s, metric=%s, parallel=%d)",
                    model_config.id,
                    model_gpu_id,
                    model_port,
                    rt.id,
                    bm.id,
                    rt.bits,
                    bm.metric,
                    n_parallel,
                )
                logger.info("=" * 60)

                force_kill_port(model_port)

                try:
                    record = runner.run_cell(
                        cell,
                        model_config,
                        gpu_id=model_gpu_id,
                        port=model_port,
                        seed=args.seed,
                        parallel_requests=n_parallel,
                        progress_position=lane_idx,
                    )
                except Exception as exc:
                    logger.error(
                        "FAILED: %s :: %s × %s — %s",
                        model_config.id,
                        rt.id,
                        bm.id,
                        exc,
                    )
                    record = RunRecord(
                        runtime_id=rt.id,
                        benchmark_id=bm.id,
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
                rec_dict["metric_used"] = bm.metric
                rec_dict["is_parity"] = bm.id in PARITY_METRICS

                with state_lock:
                    completed_keys.add(cell_key)
                    scores[model_config.id][rt.id][bm.id] = (
                        record.score if record.status == "ok" else None
                    )
                    statuses[model_config.id][rt.id][bm.id] = record.status
                    all_records.append(rec_dict)
                    progress_counter["done"] += 1
                    logger.info(
                        "[%d/%d] %s :: %s × %s => status=%s  score=%.3f  time=%.0fs  ok=%d/%d",
                        progress_counter["done"],
                        total_cells,
                        model_config.id,
                        rt.id,
                        bm.id,
                        record.status,
                        record.score or 0,
                        record.wall_time_seconds,
                        record.n_succeeded,
                        record.n_samples,
                    )
                    _save_output(
                        output_path,
                        args,
                        profile,
                        selected_models,
                        runtimes,
                        benchmarks,
                        scores,
                        statuses,
                        all_records,
                        server_meta,
                        time.monotonic() - t_global,
                    )

            # -- KV dump for this runtime (server is stopped, GPU is free) --
            if args.kv_dump and kv_dump_binary.exists():
                _inline_kv_dump(
                    rt=rt,
                    model_config=model_config,
                    gpu_id=model_gpu_id,
                    kv_ts=kv_ts,
                    probe_prompt=kv_probe_prompt,
                    probe_image=kv_probe_image_path,
                    kv_dump_paths=kv_dump_paths,
                    analysis_futures=analysis_futures,
                    analysis_pool=analysis_pool,
                    state_lock=state_lock,
                    n_gpu_layers_=n_gpu_layers,
                    kv_dump_binary=kv_dump_binary,
                )

    max_workers = len(selected_models) if len(selected_models) > 1 else 1
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(run_model_queue, lane_idx, model)
            for lane_idx, model in enumerate(selected_models)
        ]
        for future in as_completed(futures):
            future.result()

    total_time = time.monotonic() - t_global

    # Final save
    _save_output(
        output_path,
        args,
        profile,
        selected_models,
        runtimes,
        benchmarks,
        scores,
        statuses,
        all_records,
        server_meta,
        total_time,
    )

    # Print summary
    _print_summary(selected_models, runtimes, benchmarks, scores, statuses, args.num, total_time)

    # Wait for background KV analysis threads to finish
    if analysis_futures:
        logger.info("Waiting for %d background KV analysis tasks...", len(analysis_futures))
        for fut in analysis_futures:
            try:
                fut.result(timeout=120)
            except Exception as exc:
                logger.warning("Background KV analysis error: %s", exc)
        logger.info("All KV analysis tasks complete.")
    if analysis_pool is not None:
        analysis_pool.shutdown(wait=False)

    if args.kv_dump and kv_dump_paths:
        logger.info("KV dump results saved to: %s", LOGS_DIR / "kv_analysis")
        for model_id, rt_paths in kv_dump_paths.items():
            for rt_id, path in rt_paths.items():
                logger.info("  %s/%s → %s", model_id, rt_id, path)


# ---------------------------------------------------------------------------
# KV dump (inline, per-runtime)
# ---------------------------------------------------------------------------

def _resolve_kv_probe_image() -> Path | None:
    """Try to grab the first image from ai2d dataset as a default probe."""
    try:
        from tq_bench.datasets.vlm import AI2DDataset
        ds = AI2DDataset()
        ds.load(1, seed=42)
        sample = next(ds.iter_samples())
        image = sample.get("image")
        if image is not None:
            import tempfile
            tmp = Path(tempfile.mkdtemp()) / "kv_dump_probe.jpg"
            image.save(str(tmp), format="JPEG", quality=90)
            return tmp
    except Exception as exc:
        logger.debug("Could not load default probe image from ai2d: %s", exc)
    return None


def _run_analysis_background(
    baseline_path: Path,
    quant_path: Path,
    rt_id: str,
    analysis_dir: Path,
    bits: str,
):
    """Run KV comparative analysis (CPU-only, safe for background thread)."""
    try:
        from tq_bench.kv_analysis.report import generate_full_report

        bits_by_run: dict[str, int] = {}
        try:
            bits_str = bits.split("/")[0] if "/" in str(bits) else str(bits)
            bits_by_run[rt_id] = int(float(bits_str))
        except (ValueError, TypeError):
            pass

        generate_full_report(
            baseline_dump=baseline_path,
            quant_dumps={rt_id: quant_path},
            output_dir=analysis_dir,
            bits_by_run=bits_by_run if bits_by_run else None,
        )
        logger.info("KV analysis done: %s → %s", rt_id, analysis_dir)
    except Exception:
        logger.exception("KV analysis failed for %s (non-fatal)", rt_id)


def _inline_kv_dump(
    *,
    rt,
    model_config,
    gpu_id,
    kv_ts,
    probe_prompt,
    probe_image,
    kv_dump_paths,
    analysis_futures,
    analysis_pool,
    state_lock,
    n_gpu_layers_,
    kv_dump_binary,
):
    """Run llama-kv-dump for one runtime, then kick off analysis in background."""
    from tq_bench.kv_dump_runner import KVDumpConfig, run_kv_dump

    dump_dir = LOGS_DIR / "kv_analysis" / f"{model_config.id}_{kv_ts}" / "dumps" / rt.id
    config = KVDumpConfig(
        kv_dump_binary=kv_dump_binary,
        model_config=model_config,
        runtime=rt,
        output_dir=dump_dir,
        prompt=probe_prompt,
        image_path=probe_image,
        gpu_id=gpu_id,
        n_gpu_layers=n_gpu_layers_,
    )

    logger.info("[KV-DUMP] %s (cache_k=%s, cache_v=%s) ...",
                rt.id, rt.cache_type_k, rt.cache_type_v)

    if not run_kv_dump(config):
        logger.warning("[KV-DUMP] Failed for %s, skipping", rt.id)
        return

    with state_lock:
        kv_dump_paths.setdefault(model_config.id, {})[rt.id] = dump_dir

    # If this is not the baseline, and we have a baseline dump, run analysis in background
    baseline_path = kv_dump_paths.get(model_config.id, {}).get("baseline")
    if baseline_path and rt.id != "baseline" and analysis_pool is not None:
        analysis_dir = (
            LOGS_DIR / "kv_analysis" / f"{model_config.id}_{kv_ts}" / "analysis" / rt.id
        )
        fut = analysis_pool.submit(
            _run_analysis_background,
            baseline_path, dump_dir, rt.id, analysis_dir, rt.bits,
        )
        with state_lock:
            analysis_futures.append(fut)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _save_output(path, args, profile, model_configs, runtimes, benchmarks,
                 scores, statuses, records, server_meta, wall_time):
    models_meta = [
        {
            "id": model.id,
            "reasoning_mode": model.reasoning_mode,
            "gpu_id": model.gpu_id,
            "port": model.port,
            "parallel_requests": model.parallel_requests,
            "model_quantization": model.model_quantization,
            "model_path": str(model.model_path),
            "sampling_seed": model.sampling_seed,
            "temperature": model.temperature,
            "top_k": model.top_k,
            "top_p": model.top_p,
            "min_p": model.min_p,
            "repeat_penalty": model.repeat_penalty,
            "presence_penalty": model.presence_penalty,
            "frequency_penalty": model.frequency_penalty,
        }
        for model in model_configs
    ]
    output = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
            "profile": profile.id,
            "models": models_meta,
            "n_samples": args.num,
            "seed": args.seed,
            "runtimes": [rt.id for rt in runtimes],
            "benchmarks": [bm.id for bm in benchmarks],
            "parity_metrics": {bm.id: bm.metric for bm in benchmarks if bm.id in PARITY_METRICS},
            "official_reference": OFFICIAL_SCORES,
            "server_launch": server_meta,
            "total_wall_time": round(wall_time, 1),
        },
        "scores": scores,
        "statuses": statuses,
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)


def _print_summary(model_configs, runtimes, benchmarks, scores, statuses, n_samples, total_time):
    bm_ids = [bm.id for bm in benchmarks]

    for model_config in model_configs:
        model_scores = scores.get(model_config.id, {})
        model_statuses = statuses.get(model_config.id, {})
        print(f"\n{'=' * 90}")
        print(
            f"TQ-VLM-Bench  (n={n_samples}, {model_config.id}, "
            f"gpu={model_config.gpu_id}, port={model_config.port})"
        )
        print(f"{'=' * 90}")

        header = f"{'Runtime':<14} {'Bits':<7}"
        for bid in bm_ids:
            w = max(len(bid), 8)
            header += f" {bid:>{w}}"
        header += f" {'Avg':>8}"
        print(header)
        print("-" * 90)

        baseline_scores = model_scores.get("baseline", {})
        baseline_statuses = model_statuses.get("baseline", {})

        for rt in runtimes:
            row = f"{rt.id:<14} {rt.bits:<7}"
            vals = []
            for bid in bm_ids:
                s = model_scores.get(rt.id, {}).get(bid)
                status = model_statuses.get(rt.id, {}).get(bid)
                w = max(len(bid), 8)
                if s is None:
                    label = {
                        "server_crash": "CRASH",
                        "error": "ERROR",
                        "fail": "FAIL",
                    }.get(status, "---")
                    row += f" {label:>{w}}"
                else:
                    vals.append(s)
                    row += f" {s:>{w}.3f}"

            avg = sum(vals) / len(vals) if vals else 0
            row += f" {avg:>8.3f}"
            print(row)

        if "baseline" in {rt.id for rt in runtimes} and len(runtimes) > 1:
            print("-" * 90)
            print("Delta from baseline:")
            for rt in runtimes:
                if rt.id == "baseline":
                    continue
                row = f"  {rt.id:<12} {'':7}"
                deltas = []
                for bid in bm_ids:
                    s = model_scores.get(rt.id, {}).get(bid)
                    bl = baseline_scores.get(bid)
                    status = model_statuses.get(rt.id, {}).get(bid)
                    w = max(len(bid), 8)
                    if s is not None and bl is not None:
                        d = s - bl
                        deltas.append(d)
                        sign = "+" if d >= 0 else ""
                        row += f" {sign}{d:>{w-1}.3f}"
                    else:
                        label = {
                            "server_crash": "CRASH",
                            "error": "ERROR",
                            "fail": "FAIL",
                        }.get(status, "---")
                        row += f" {label:>{w}}"
                avg_d = sum(deltas) / len(deltas) if deltas else 0
                sign = "+" if avg_d >= 0 else ""
                row += f" {sign}{avg_d:>7.3f}" if deltas else f" {'---':>8}"
                print(row)

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
                    print(
                        f"  {bid:<12} ours={ours:.3f}  official={off:.3f}  "
                        f"delta={sign}{gap:.3f}  [{status}]"
                    )
                else:
                    baseline_status = baseline_statuses.get(bid)
                    print(
                        f"  {bid:<12} ours={baseline_status or '---':>5}  official={off:.3f}  [NO SCORE]"
                    )

    print("-" * 90)
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print()


if __name__ == "__main__":
    main()
