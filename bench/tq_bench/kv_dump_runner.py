"""KV dump extraction + analysis runner.

Wraps the C++ ``llama-kv-dump`` binary and the Python
``kv_analysis`` pipeline into a single callable that:

1. Runs ``llama-kv-dump`` for a given runtime config → produces
   ``K_layer_*.bin`` / ``V_layer_*.bin`` / ``meta.json``
2. Runs the Python analysis (distribution, outliers, quant_error,
   rotation, attention) against a baseline dump
3. Saves all outputs to a structured directory

Designed to be called from ``run_bench.py`` after benchmark cells
complete for each runtime, so the same model/GPU/port config is reused.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ModelConfig, RuntimeConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Default VLM prompt + image for KV dump probes.
# This prompt exercises both vision and text tokens.
_DEFAULT_VLM_PROMPT = "Describe this image in detail."
_DEFAULT_TEXT_PROMPT = "Explain the concept of quantization in neural networks."


@dataclass
class KVDumpConfig:
    """Configuration for a single KV dump extraction."""

    kv_dump_binary: Path
    model_config: ModelConfig
    runtime: RuntimeConfig
    output_dir: Path
    prompt: str = _DEFAULT_VLM_PROMPT
    image_path: Path | None = None
    gpu_id: int | None = None
    ctx_size: int = 4096
    n_gpu_layers: int = 99
    disable_attn_rot: bool = True  # LLAMA_ATTN_ROT_DISABLE=1


# ---------------------------------------------------------------------------
# KV dump extraction (C++)
# ---------------------------------------------------------------------------


def run_kv_dump(config: KVDumpConfig) -> bool:
    """Run llama-kv-dump and return True on success.

    The output directory will contain:
    - ``K_layer_*.bin``, ``V_layer_*.bin`` — per-layer float32 tensors
    - ``meta.json`` — metadata including vision token mask
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if not config.kv_dump_binary.exists():
        logger.error("llama-kv-dump binary not found: %s", config.kv_dump_binary)
        return False

    cmd = [
        str(config.kv_dump_binary),
        "-m", str(config.model_config.model_path),
        "--cache-type-k", config.runtime.cache_type_k,
        "--cache-type-v", config.runtime.cache_type_v,
        "-p", config.prompt,
        "-o", str(config.output_dir),
        "--ctx-size", str(config.ctx_size),
        "-ngl", str(config.n_gpu_layers),
        "-fa", "on",
    ]

    if config.model_config.mmproj_path is not None:
        cmd.extend(["--mmproj", str(config.model_config.mmproj_path)])

    if config.image_path is not None:
        cmd.extend(["--image", str(config.image_path)])

    env = os.environ.copy()
    if config.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    if config.disable_attn_rot:
        env["LLAMA_ATTN_ROT_DISABLE"] = "1"

    logger.info(
        "Running llama-kv-dump: runtime=%s, cache_k=%s, cache_v=%s, output=%s",
        config.runtime.id,
        config.runtime.cache_type_k,
        config.runtime.cache_type_v,
        config.output_dir,
    )
    logger.debug("Command: %s", " ".join(cmd))

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes should be plenty for a single forward pass
            env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            logger.error(
                "llama-kv-dump failed (exit=%d, %.1fs):\n%s",
                result.returncode,
                elapsed,
                result.stdout[-2000:] if result.stdout else "(no output)",
            )
            return False

        # Verify meta.json was produced
        meta_path = config.output_dir / "meta.json"
        if not meta_path.exists():
            logger.error("llama-kv-dump ran but meta.json not found in %s", config.output_dir)
            return False

        logger.info("llama-kv-dump completed in %.1fs → %s", elapsed, config.output_dir)
        return True

    except subprocess.TimeoutExpired:
        logger.error("llama-kv-dump timed out after 300s")
        return False
    except FileNotFoundError:
        logger.error("llama-kv-dump binary not executable: %s", config.kv_dump_binary)
        return False


# ---------------------------------------------------------------------------
# Full pipeline: dump + analyze
# ---------------------------------------------------------------------------


def run_kv_dump_and_analyze(
    *,
    kv_dump_binary: Path,
    model_config: ModelConfig,
    runtimes: list[RuntimeConfig],
    output_base_dir: Path,
    baseline_runtime_id: str = "baseline",
    prompt: str = _DEFAULT_VLM_PROMPT,
    image_path: Path | None = None,
    gpu_id: int | None = None,
    ctx_size: int = 4096,
    n_gpu_layers: int = 99,
    disable_attn_rot: bool = True,
) -> dict[str, Any]:
    """Run KV dump extraction for all runtimes, then analyze.

    1. For each runtime, run llama-kv-dump
    2. Run Python analysis: distribution, outliers, rotation (baseline only),
       quant_error + attention (baseline vs quantized pairs)
    3. Save results to ``output_base_dir/``

    Parameters
    ----------
    kv_dump_binary:
        Path to the llama-kv-dump binary.
    model_config:
        Model configuration (paths, mmproj, etc.).
    runtimes:
        List of runtime configs to dump. Must include baseline.
    output_base_dir:
        Root directory for all KV dump outputs.
    baseline_runtime_id:
        Which runtime ID to treat as the baseline for comparisons.

    Returns
    -------
    dict with keys:
        - "dumps": dict[str, Path] — runtime_id → dump directory
        - "analysis": Path | None — analysis report directory
        - "success": bool
    """
    output_base_dir.mkdir(parents=True, exist_ok=True)
    dumps_dir = output_base_dir / "dumps"
    analysis_dir = output_base_dir / "analysis"

    dump_paths: dict[str, Path] = {}
    success_count = 0

    # Step 1: Extract KV dumps for each runtime
    for rt in runtimes:
        rt_dump_dir = dumps_dir / rt.id
        config = KVDumpConfig(
            kv_dump_binary=kv_dump_binary,
            model_config=model_config,
            runtime=rt,
            output_dir=rt_dump_dir,
            prompt=prompt,
            image_path=image_path,
            gpu_id=gpu_id,
            ctx_size=ctx_size,
            n_gpu_layers=n_gpu_layers,
            disable_attn_rot=disable_attn_rot,
        )

        if run_kv_dump(config):
            dump_paths[rt.id] = rt_dump_dir
            success_count += 1
        else:
            logger.warning("KV dump failed for runtime %s, skipping", rt.id)

    if not dump_paths:
        logger.error("No KV dumps succeeded, skipping analysis")
        return {"dumps": dump_paths, "analysis": None, "success": False}

    # Step 2: Run Python analysis
    if baseline_runtime_id not in dump_paths:
        logger.warning(
            "Baseline runtime '%s' not in successful dumps (%s). "
            "Skipping comparative analysis.",
            baseline_runtime_id,
            list(dump_paths.keys()),
        )
        return {"dumps": dump_paths, "analysis": None, "success": True}

    try:
        from .kv_analysis.report import generate_full_report

        baseline_path = dump_paths[baseline_runtime_id]
        quant_dumps = {
            rt_id: path
            for rt_id, path in dump_paths.items()
            if rt_id != baseline_runtime_id
        }

        # Map runtime IDs to approximate bitwidths for theoretical comparison
        bits_by_run: dict[str, int] = {}
        for rt in runtimes:
            if rt.id in quant_dumps:
                try:
                    # bits field can be "4", "K4/V3", etc.
                    bits_str = rt.bits.split("/")[0] if "/" in str(rt.bits) else str(rt.bits)
                    bits_by_run[rt.id] = int(float(bits_str))
                except (ValueError, TypeError):
                    pass

        logger.info(
            "Running KV analysis: baseline=%s, quant=%s",
            baseline_runtime_id,
            list(quant_dumps.keys()),
        )

        result = generate_full_report(
            baseline_dump=baseline_path,
            quant_dumps=quant_dumps,
            output_dir=analysis_dir,
            bits_by_run=bits_by_run if bits_by_run else None,
        )

        logger.info(
            "KV analysis complete. Reports: %s",
            list(result.keys()),
        )
        return {"dumps": dump_paths, "analysis": analysis_dir, "success": True}

    except Exception:
        logger.exception("KV analysis failed (non-fatal)")
        return {"dumps": dump_paths, "analysis": None, "success": True}
