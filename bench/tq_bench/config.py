from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ExperimentCell:
    runtime: RuntimeConfig
    benchmark: BenchmarkConfig


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

    return [
        BenchmarkConfig(
            id=item["id"],
            task_type=item["type"],
            sample_count=int(item["sample_count"]),
            metric=item["metric"],
            max_tokens=int(item.get("max_tokens", 256)),
        )
        for item in benchmarks
    ]


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
        loaded[model_id] = ModelConfig(
            id=model_id,
            family=item["family"],
            model_path=model_path,
            mmproj_path=mmproj_path,
            description=item.get("description", ""),
            max_tokens_override=int(mto) if mto is not None else None,
            reasoning_mode=str(item.get("reasoning_mode", "off")),
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

