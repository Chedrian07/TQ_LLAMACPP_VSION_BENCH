#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tq_bench.colab import ensure_model_artifacts
from tq_bench.config import load_models


BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parent
MODELS_YAML = BENCH_DIR / "configs" / "models.yaml"


def main() -> None:
    models = load_models(MODELS_YAML)

    parser = argparse.ArgumentParser(
        description="Download and stage GGUF/mmproj artifacts described in models.yaml"
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        choices=sorted(models.keys()),
        help="Model id to download. Repeatable. Default: all.",
    )
    parser.add_argument(
        "--quant",
        action="append",
        dest="quants",
        choices=["bf16", "q4_k_m", "q8_0"],
        help="Quantization variant to stage. Repeatable. Default: bf16 + q4_k_m + q8_0.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root directory for persistent model cache (default: project root).",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Redownload remote artifacts even if the cache already contains them.",
    )
    args = parser.parse_args()

    selected_models = args.models or sorted(models.keys())
    selected_quants = args.quants or ["bf16", "q4_k_m", "q8_0"]

    for model_id in selected_models:
        for quant in selected_quants:
            copied = ensure_model_artifacts(
                MODELS_YAML,
                model_id=model_id,
                quant=quant,
                drive_root=args.cache_root,
                force_redownload=args.force_redownload,
            )
            print(
                f"{model_id} {quant}: model={copied[quant]} "
                f"mmproj={copied.get('mmproj')} cache={copied['cache_dir']}"
            )


if __name__ == "__main__":
    main()
