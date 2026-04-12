from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import HfApi, hf_hub_download

from .config import DownloadFileSpec, ModelConfig, load_models
from .env import prepend_ld_library_path


def _run_text(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo_name_from_url(repo_url: str) -> str:
    repo_name = Path(repo_url.rstrip("/")).name
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    return repo_name


def _repo_slug_from_url(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if not path:
        path = _repo_name_from_url(repo_url)
    slug = path.replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", slug)


def _git_remote_url(repo_root: Path) -> str:
    return _run_text(["git", "config", "--get", "remote.origin.url"], cwd=repo_root)


def ensure_repo_checkout(repo_url: str, branch: str, workspace_dir: str | Path) -> Path:
    workspace = Path(workspace_dir).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    repo_name = _repo_name_from_url(repo_url)
    repo_root = workspace / repo_name

    if not (repo_root / ".git").exists():
        subprocess.run(
            ["git", "clone", "--branch", branch, "--depth", "1", repo_url, str(repo_root)],
            check=True,
        )
        return repo_root

    subprocess.run(["git", "fetch", "--depth", "1", "origin", branch], cwd=str(repo_root), check=True)
    subprocess.run(["git", "checkout", "-B", branch, "FETCH_HEAD"], cwd=str(repo_root), check=True)
    subprocess.run(["git", "reset", "--hard", "FETCH_HEAD"], cwd=str(repo_root), check=True)
    return repo_root


@dataclass(frozen=True)
class GitCheckout:
    repo_url: str
    branch: str
    commit: str
    path: Path


def ensure_llama_cpp_checkout(
    repo_root: str | Path,
    *,
    llama_repo_url: str,
    branch: str,
    commit: str | None = None,
    checkout_dir_name: str = "llama.cpp",
) -> GitCheckout:
    root = Path(repo_root).resolve()
    llama_root = root / checkout_dir_name

    if not llama_root.exists():
        subprocess.run(
            ["git", "clone", "--branch", branch, "--depth", "1", llama_repo_url, str(llama_root)],
            check=True,
        )
    elif not (llama_root / ".git").exists():
        raise RuntimeError(
            f"{llama_root} exists but is not a git checkout; cannot stage custom llama.cpp"
        )
    else:
        subprocess.run(["git", "fetch", "--tags", "origin"], cwd=str(llama_root), check=True)
        subprocess.run(["git", "fetch", "origin", branch], cwd=str(llama_root), check=True)

    if commit:
        subprocess.run(["git", "fetch", "origin", commit], cwd=str(llama_root), check=True)
        subprocess.run(["git", "checkout", "--detach", commit], cwd=str(llama_root), check=True)
    else:
        subprocess.run(["git", "checkout", branch], cwd=str(llama_root), check=True)
        subprocess.run(["git", "reset", "--hard", f"origin/{branch}"], cwd=str(llama_root), check=True)

    resolved_commit = _run_text(["git", "rev-parse", "HEAD"], cwd=llama_root)
    return GitCheckout(
        repo_url=llama_repo_url,
        branch=branch,
        commit=resolved_commit,
        path=llama_root,
    )


def install_bench_editable(repo_root: str | Path) -> None:
    root = Path(repo_root).resolve()
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(root / "bench")],
        check=True,
    )


def configure_hf_cache(drive_root: str | Path) -> dict[str, Path]:
    root = Path(drive_root).expanduser().resolve()
    hf_root = root / "cache" / "huggingface"
    paths = {
        "hf_home": hf_root,
        "datasets": hf_root / "datasets",
        "hub": hf_root / "hub",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(paths["hf_home"])
    os.environ["HF_DATASETS_CACHE"] = str(paths["datasets"])
    os.environ["HF_HUB_CACHE"] = str(paths["hub"])
    return paths


def _detect_from_compute_cap() -> str | None:
    try:
        value = _run_text(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
        ).splitlines()[0].strip()
    except Exception:
        return None

    match = re.match(r"^\s*(\d+)\.(\d+)\s*$", value)
    if match is None:
        return None
    major, minor = match.groups()
    return f"{major}{minor}"


def detect_cuda_architecture() -> str:
    detected = _detect_from_compute_cap()
    if detected:
        return detected

    name = _run_text(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines()[0].strip()
    fallback_map = (
        ("RTX PRO 6000 Blackwell", "120"),
        ("Blackwell", "120"),
        ("RTX 5070 Ti", "120"),
        ("RTX 5070", "120"),
        ("RTX 4060 Ti", "89"),
        ("L4", "89"),
        ("A10G", "86"),
        ("A100", "80"),
        ("T4", "75"),
    )
    for needle, arch in fallback_map:
        if needle in name:
            return arch
    raise RuntimeError(f"Could not detect CUDA architecture for GPU '{name}'")


def _nvcc_version() -> str:
    return _run_text(["nvcc", "--version"])


def _llama_commit(llama_root: Path) -> str:
    return _run_text(["git", "rev-parse", "--short", "HEAD"], cwd=llama_root)


@dataclass(frozen=True)
class LlamaServerArtifact:
    repo_root: Path
    llama_root: Path
    llama_repo_url: str
    binary_path: Path
    lib_dir: Path
    cache_dir: Path
    cuda_arch: str
    llama_commit: str
    nvcc_version: str
    build_type: str = "Release"

    def to_manifest(self) -> dict[str, str]:
        data = asdict(self)
        return {
            "repo_root": str(data["repo_root"]),
            "llama_root": str(data["llama_root"]),
            "llama_repo_url": data["llama_repo_url"],
            "binary_path": str(data["binary_path"]),
            "lib_dir": str(data["lib_dir"]),
            "cache_dir": str(data["cache_dir"]),
            "cuda_arch": data["cuda_arch"],
            "llama_commit": data["llama_commit"],
            "nvcc_version": data["nvcc_version"],
            "build_type": data["build_type"],
        }


def ensure_llama_server(
    repo_root: str | Path,
    drive_root: str | Path,
    *,
    force_rebuild: bool = False,
    build_dir_name: str = "build-colab",
) -> LlamaServerArtifact:
    root = Path(repo_root).resolve()
    drive = Path(drive_root).expanduser().resolve()
    llama_root = root / "llama.cpp"
    if not (llama_root / ".git").exists():
        raise FileNotFoundError(
            f"Custom llama.cpp checkout not found at {llama_root}. "
            "Call ensure_llama_cpp_checkout() first."
        )
    llama_repo_url = _git_remote_url(llama_root)
    repo_slug = _repo_slug_from_url(llama_repo_url)
    cuda_arch = detect_cuda_architecture()
    llama_commit = _llama_commit(llama_root)
    nvcc_version = _nvcc_version()
    cache_dir = drive / "cache" / "llama_server" / repo_slug / llama_commit / f"sm{cuda_arch}"
    cached_lib_dir = cache_dir / "bin"
    cached_binary = cached_lib_dir / "llama-server"
    manifest_path = cache_dir / "manifest.json"

    if not force_rebuild and cached_binary.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("llama_repo_url") == llama_repo_url
            and manifest.get("cuda_arch") == cuda_arch
            and manifest.get("llama_commit") == llama_commit
            and manifest.get("nvcc_version") == nvcc_version
            and manifest.get("build_type") == "Release"
        ):
            prepend_ld_library_path(cached_lib_dir)
            return LlamaServerArtifact(
                repo_root=root,
                llama_root=llama_root,
                llama_repo_url=llama_repo_url,
                binary_path=cached_binary,
                lib_dir=cached_lib_dir,
                cache_dir=cache_dir,
                cuda_arch=cuda_arch,
                llama_commit=llama_commit,
                nvcc_version=nvcc_version,
            )

    build_dir = llama_root / build_dir_name
    subprocess.run(
        [
            "cmake",
            "-B",
            str(build_dir),
            "-G",
            "Ninja",
            "-DGGML_CUDA=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=str(llama_root),
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "-j", str(os.cpu_count() or 2), "--target", "llama-server"],
        cwd=str(llama_root),
        check=True,
    )

    local_lib_dir = build_dir / "bin"
    local_binary = local_lib_dir / "llama-server"
    if not local_binary.exists():
        raise FileNotFoundError(f"Expected llama-server at {local_binary}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    if cached_lib_dir.exists():
        shutil.rmtree(cached_lib_dir)
    shutil.copytree(local_lib_dir, cached_lib_dir)
    artifact = LlamaServerArtifact(
        repo_root=root,
        llama_root=llama_root,
        llama_repo_url=llama_repo_url,
        binary_path=local_binary,
        lib_dir=local_lib_dir,
        cache_dir=cache_dir,
        cuda_arch=cuda_arch,
        llama_commit=llama_commit,
        nvcc_version=nvcc_version,
    )
    manifest_path.write_text(
        json.dumps(artifact.to_manifest(), indent=2),
        encoding="utf-8",
    )
    prepend_ld_library_path(local_lib_dir)
    return artifact


def _resolve_download_filename(repo_id: str, spec: DownloadFileSpec) -> str:
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    if spec.filename and spec.filename in repo_files:
        return spec.filename
    if spec.suffix:
        matches = [name for name in repo_files if name.endswith(spec.suffix)]
        if matches:
            matches = sorted(matches, key=lambda item: (item.count("/"), len(item), item))
            return matches[0]
    raise FileNotFoundError(
        f"Could not resolve file in {repo_id} for spec "
        f"filename={spec.filename!r} suffix={spec.suffix!r}"
    )


def _select_model_target(model: ModelConfig, quant: str) -> Path:
    quant_key = quant.lower()
    if quant_key == "bf16":
        return model.model_path
    try:
        return model.quantized_model_paths[quant_key]
    except KeyError as exc:
        raise KeyError(f"Model '{model.id}' does not define quant '{quant}'") from exc


def ensure_model_artifacts(
    models_yaml: str | Path,
    *,
    model_id: str,
    quant: str,
    drive_root: str | Path,
    force_redownload: bool = False,
) -> dict[str, Path]:
    models = load_models(models_yaml)
    if model_id not in models:
        raise KeyError(f"Unknown model '{model_id}' in {models_yaml}")
    model = models[model_id]
    if model.download is None:
        raise ValueError(f"Model '{model_id}' does not define download metadata")

    repo_root = Path(models_yaml).resolve().parents[2]
    drive = Path(drive_root).expanduser().resolve()
    cache_dir = drive / "models" / model.model_path.parent.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    quant_key = quant.lower()
    required_artifacts = {
        quant_key: _select_model_target(model, quant_key),
    }
    if model.mmproj_path is not None:
        required_artifacts["mmproj"] = model.mmproj_path

    copied: dict[str, Path] = {}
    for artifact_id, target_path in required_artifacts.items():
        spec = model.download.files.get(artifact_id)
        if spec is None:
            raise KeyError(
                f"Model '{model.id}' download metadata does not define artifact '{artifact_id}'"
            )
        remote_filename = _resolve_download_filename(model.download.repo_id, spec)
        cached_path = cache_dir / Path(remote_filename).name
        if force_redownload and cached_path.exists():
            cached_path.unlink()
        if not cached_path.exists():
            downloaded = hf_hub_download(
                repo_id=model.download.repo_id,
                filename=remote_filename,
                repo_type="model",
                local_dir=str(cache_dir),
                local_dir_use_symlinks=False,
            )
            cached_path = Path(downloaded)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_path, target_path)
        copied[artifact_id] = target_path

    copied["cache_dir"] = cache_dir
    copied["repo_root"] = repo_root
    return copied


def find_latest_run_file(results_dir: str | Path, *, model_id: str | None = None) -> Path | None:
    root = Path(results_dir).expanduser().resolve()
    if not root.exists():
        return None
    candidates = sorted(root.glob("bench_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if model_id is None:
        return candidates[0] if candidates else None
    for candidate in candidates:
        if model_id in candidate.name:
            return candidate
    return None


def build_run_bench_command(
    repo_root: str | Path,
    *,
    num: int,
    model_id: str,
    model_quant: str,
    runtimes: list[str],
    benchmarks: list[str] | None = None,
    profile: str = "colab",
    gpu_id: int | None = None,
    parallel: int | None = None,
    port: int | None = None,
    results_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    slot_save_path: str | Path | None = None,
) -> list[str]:
    root = Path(repo_root).resolve()
    cmd = [
        sys.executable,
        str(root / "bench" / "run_bench.py"),
        "--num",
        str(num),
        "--profile",
        profile,
        "--model",
        model_id,
        "--model-quant",
        model_quant,
        "--runtimes",
        *runtimes,
    ]
    if benchmarks:
        cmd.extend(["--benchmarks", *benchmarks])
    if gpu_id is not None:
        cmd.extend(["--gpu", str(gpu_id)])
    if parallel is not None:
        cmd.extend(["--parallel", str(parallel)])
    if port is not None:
        cmd.extend(["--port", str(port)])
    if results_dir is not None:
        cmd.extend(["--results-dir", str(Path(results_dir).expanduser().resolve())])
    if output_path is not None:
        cmd.extend(["--output", str(Path(output_path).expanduser().resolve())])
    if resume_path is not None:
        cmd.extend(["--resume", str(Path(resume_path).expanduser().resolve())])
    if slot_save_path is not None:
        cmd.extend(["--slot-save-path", str(Path(slot_save_path).expanduser().resolve())])
    return cmd
