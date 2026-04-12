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
class DownloadFileSpec:
    filename: str | None = None
    suffix: str | None = None


@dataclass(frozen=True)
class ModelDownloadConfig:
    repo_id: str
    files: dict[str, DownloadFileSpec] = field(default_factory=dict)


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
    download: ModelDownloadConfig | None = None


@dataclass(frozen=True)
class ExecutionProfile:
    id: str
    description: str = ""
    gpu_id: int | None = None
    port: int | None = None
    parallel_requests: int | None = None
    cache_ram: int | None = None
    slot_save_path: Path | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    n_gpu_layers: int | None = None
    no_warmup: bool | None = None
    no_mmap: bool | None = None
    results_dir: Path | None = None


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
        download_raw = item.get("download")
        download_config: ModelDownloadConfig | None = None
        if download_raw is not None:
            if not isinstance(download_raw, dict):
                raise TypeError(f"Model config for {model_id}.download must be a mapping")
            repo_id = str(download_raw.get("repo_id", "")).strip()
            if not repo_id:
                raise TypeError(f"Model config for {model_id}.download.repo_id is required")
            files_raw = download_raw.get("files", {})
            if not isinstance(files_raw, dict):
                raise TypeError(f"Model config for {model_id}.download.files must be a mapping")
            files: dict[str, DownloadFileSpec] = {}
            for artifact_id, spec_raw in files_raw.items():
                if isinstance(spec_raw, str):
                    files[str(artifact_id).lower()] = DownloadFileSpec(suffix=spec_raw)
                    continue
                if not isinstance(spec_raw, dict):
                    raise TypeError(
                        f"Model config for {model_id}.download.files.{artifact_id} "
                        "must be a mapping or string"
                    )
                filename = spec_raw.get("filename")
                suffix = spec_raw.get("suffix")
                if filename is None and suffix is None:
                    raise TypeError(
                        f"Model config for {model_id}.download.files.{artifact_id} "
                        "must define filename or suffix"
                    )
                files[str(artifact_id).lower()] = DownloadFileSpec(
                    filename=str(filename) if filename is not None else None,
                    suffix=str(suffix) if suffix is not None else None,
                )
            download_config = ModelDownloadConfig(repo_id=repo_id, files=files)
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
            download=download_config,
        )
    return loaded


def load_profiles(path: str | Path) -> dict[str, ExecutionProfile]:
    resolved = Path(path)
    data = _load_yaml(resolved)
    profiles_raw = data.get("profiles", {})
    if not isinstance(profiles_raw, dict):
        raise TypeError("profiles must be a mapping")

    profiles: dict[str, ExecutionProfile] = {}
    for profile_id, item in profiles_raw.items():
        if not isinstance(item, dict):
            raise TypeError(f"Profile config for {profile_id} must be a mapping")
        slot_save_value = item.get("slot_save_path")
        results_dir_value = item.get("results_dir")
        profiles[str(profile_id)] = ExecutionProfile(
            id=str(profile_id),
            description=str(item.get("description", "")),
            gpu_id=int(item["gpu_id"]) if item.get("gpu_id") is not None else None,
            port=int(item["port"]) if item.get("port") is not None else None,
            parallel_requests=(
                int(item["parallel_requests"])
                if item.get("parallel_requests") is not None
                else None
            ),
            cache_ram=int(item["cache_ram"]) if item.get("cache_ram") is not None else None,
            slot_save_path=(
                (resolved.parent / slot_save_value).resolve()
                if slot_save_value and not Path(slot_save_value).is_absolute()
                else Path(slot_save_value).resolve() if slot_save_value else None
            ),
            batch_size=int(item["batch_size"]) if item.get("batch_size") is not None else None,
            ubatch_size=(
                int(item["ubatch_size"])
                if item.get("ubatch_size") is not None
                else None
            ),
            n_gpu_layers=(
                int(item["n_gpu_layers"])
                if item.get("n_gpu_layers") is not None
                else None
            ),
            no_warmup=(
                bool(item["no_warmup"])
                if item.get("no_warmup") is not None
                else None
            ),
            no_mmap=(
                bool(item["no_mmap"])
                if item.get("no_mmap") is not None
                else None
            ),
            results_dir=(
                (resolved.parent / results_dir_value).resolve()
                if results_dir_value and not Path(results_dir_value).is_absolute()
                else Path(results_dir_value).resolve() if results_dir_value else None
            ),
        )
    return profiles


def build_matrix(
    runtimes: Iterable[RuntimeConfig],
    benchmarks: Iterable[BenchmarkConfig],
) -> list[ExperimentCell]:
    return [
        ExperimentCell(runtime=runtime, benchmark=benchmark)
        for runtime in runtimes
        for benchmark in benchmarks
    ]
