from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class RuntimeConfig:
    id: str
    method: str
    cache_type_k: str
    cache_type_v: str
    bits: str
    description: str = ""
    ctx_size: int = 32768


@dataclass(frozen=True)
class BenchmarkConfig:
    id: str
    task_type: str
    sample_count: int
    metric: str
    max_tokens: int = 256  # Per-benchmark generation budget
    # Parity-mode fields: when parity_mode is True, use parity_metric and
    # parity_sample_count instead of metric and sample_count.
    parity_mode: bool = False
    parity_metric: str | None = None
    parity_sample_count: int | None = None  # -1 = full split


@dataclass(frozen=True)
class ModelConfig:
    id: str
    family: str
    model_path: Path
    mmproj_path: Path | None = None
    description: str = ""
    # Model-specific max_tokens override (e.g. Thinking models need more room)
    max_tokens_override: int | None = None
    reasoning_mode: str = "off"  # "off" | "think"  — informational
    # Optional execution-lane hints for dual-GPU / mixed-model runs.
    gpu_id: int | None = None
    port: int | None = None
    parallel_requests: int | None = None
    # Optional model-specific sampling defaults. These are sent per request
    # to llama-server's OpenAI-compatible API.
    sampling_seed: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    quantized_model_paths: dict[str, Path] = field(default_factory=dict)
    model_quantization: str = "bf16"


@dataclass(frozen=True)
class ExperimentCell:
    runtime: RuntimeConfig
    benchmark: BenchmarkConfig
    model_id: str = ""


def _load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping at {resolved}, got {type(data).__name__}")
    return data


def load_runtimes(path: str | Path) -> list[RuntimeConfig]:
    data = _load_yaml(path)
    defaults = data.get("defaults", {})
    runtimes = data.get("runtimes", [])
    if not isinstance(defaults, dict):
        raise TypeError("runtimes.defaults must be a mapping")
    if not isinstance(runtimes, list):
        raise TypeError("runtimes must be a list")

    return [
        RuntimeConfig(
            id=item["id"],
            method=item["method"],
            cache_type_k=item["cache_type_k"],
            cache_type_v=item["cache_type_v"],
            bits=str(item["bits"]),
            description=item.get("description", ""),
            ctx_size=int(item.get("ctx_size", defaults.get("ctx_size", 32768))),
        )
        for item in runtimes
    ]


def load_benchmarks(path: str | Path) -> list[BenchmarkConfig]:
    data = _load_yaml(path)
    benchmarks = data.get("benchmarks", [])
    if not isinstance(benchmarks, list):
        raise TypeError("benchmarks must be a list")

    result = []
    for item in benchmarks:
        psc = item.get("parity_sample_count")
        result.append(
            BenchmarkConfig(
                id=item["id"],
                task_type=item["type"],
                sample_count=int(item["sample_count"]),
                metric=item["metric"],
                max_tokens=int(item.get("max_tokens", 256)),
                parity_mode=bool(item.get("parity_mode", False)),
                parity_metric=item.get("parity_metric"),
                parity_sample_count=int(psc) if psc is not None else None,
            )
        )
    return result


def load_models(path: str | Path) -> dict[str, ModelConfig]:
    resolved = Path(path)
    data = _load_yaml(resolved)
    models = data.get("models", {})
    if not isinstance(models, dict):
        raise TypeError("models must be a mapping")

    loaded: dict[str, ModelConfig] = {}
    for model_id, item in models.items():
        if not isinstance(item, dict):
            raise TypeError(f"Model config for {model_id} must be a mapping")

        model_path = (resolved.parent / item["model_path"]).resolve()
        mmproj_value = item.get("mmproj_path")
        mmproj_path = (resolved.parent / mmproj_value).resolve() if mmproj_value else None
        mto = item.get("max_tokens_override")
        quantized_model_paths_raw = item.get("quantized_model_paths", {})
        if not isinstance(quantized_model_paths_raw, dict):
            raise TypeError(
                f"Model config for {model_id}.quantized_model_paths must be a mapping"
            )
        quantized_model_paths = {
            str(quant_name).lower(): (resolved.parent / rel_path).resolve()
            for quant_name, rel_path in quantized_model_paths_raw.items()
        }
        loaded[model_id] = ModelConfig(
            id=model_id,
            family=item["family"],
            model_path=model_path,
            mmproj_path=mmproj_path,
            description=item.get("description", ""),
            max_tokens_override=int(mto) if mto is not None else None,
            reasoning_mode=str(item.get("reasoning_mode", "off")),
            gpu_id=int(item["gpu_id"]) if item.get("gpu_id") is not None else None,
            port=int(item["port"]) if item.get("port") is not None else None,
            parallel_requests=(
                int(item["parallel_requests"])
                if item.get("parallel_requests") is not None
                else None
            ),
            sampling_seed=(
                int(item["sampling_seed"])
                if item.get("sampling_seed") is not None
                else None
            ),
            temperature=(
                float(item["temperature"])
                if item.get("temperature") is not None
                else None
            ),
            top_k=int(item["top_k"]) if item.get("top_k") is not None else None,
            top_p=float(item["top_p"]) if item.get("top_p") is not None else None,
            min_p=float(item["min_p"]) if item.get("min_p") is not None else None,
            repeat_penalty=(
                float(item["repeat_penalty"])
                if item.get("repeat_penalty") is not None
                else None
            ),
            presence_penalty=(
                float(item["presence_penalty"])
                if item.get("presence_penalty") is not None
                else None
            ),
            frequency_penalty=(
                float(item["frequency_penalty"])
                if item.get("frequency_penalty") is not None
                else None
            ),
            quantized_model_paths=quantized_model_paths,
        )
    return loaded


def build_matrix(
    runtimes: Iterable[RuntimeConfig],
    benchmarks: Iterable[BenchmarkConfig],
) -> list[ExperimentCell]:
    return [
        ExperimentCell(runtime=runtime, benchmark=benchmark)
        for runtime in runtimes
        for benchmark in benchmarks
    ]
