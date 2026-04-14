from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

from tq_bench.colab import (
    _detect_llama_binary,
    _materialize_local_binaries,
    _repo_slug_from_url,
    _nvcc_version,
    ensure_llama_server,
    build_run_bench_command,
    detect_cuda_architecture,
    find_latest_run_file,
    install_bench_editable,
    run_command_live,
)


def _write_minimal_bench_pyproject(bench_dir: Path) -> None:
    (bench_dir / "pyproject.toml").write_text(
        """
[project]
name = "demo-bench"
version = "0.1.0"
dependencies = [
    "pandas>=2.0.0",
]
""".strip(),
        encoding="utf-8",
    )


def test_build_run_bench_command_includes_colab_overrides(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "bench").mkdir(parents=True)
    build_colab_bin = repo_root / "llama.cpp" / "build-colab" / "bin"
    build_colab_bin.mkdir(parents=True)
    (build_colab_bin / "llama-server").write_text("", encoding="utf-8")
    (build_colab_bin / "llama-kv-dump").write_text("", encoding="utf-8")
    cmd = build_run_bench_command(
        repo_root,
        num=10,
        model_id="qwen3_vl_2b_instruct",
        model_quant="q4_k_m",
        runtimes=["baseline", "tq-4"],
        benchmarks=["ai2d"],
        profile="colab",
        results_dir=tmp_path / "results",
        output_path=tmp_path / "results" / "run.json",
        resume_path=tmp_path / "results" / "resume.json",
        slot_save_path=tmp_path / "kvcache",
    )

    joined = " ".join(cmd)
    assert str(repo_root / "bench" / "run_bench.py") in joined
    assert "--profile colab" in joined
    assert "--model qwen3_vl_2b_instruct" in joined
    assert "--model-quant q4_k_m" in joined
    assert "--output" in cmd
    assert "--resume" in cmd
    assert "--slot-save-path" in cmd
    assert "--server-binary" in cmd
    assert str(build_colab_bin / "llama-server") in joined
    assert "--kv-dump-binary" in cmd
    assert str(build_colab_bin / "llama-kv-dump") in joined


def test_build_run_bench_command_appends_extra_args(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "bench").mkdir(parents=True)
    cmd = build_run_bench_command(
        repo_root,
        num=10,
        model_id="qwen3_vl_2b_thinking",
        model_quant="bf16",
        runtimes=["core", "prod"],
        benchmarks=["ai2d"],
        profile="colab",
        extra_args=["--seed", "123", "--kv-dump-prompt", "hello"],
    )

    assert cmd[-4:] == ["--seed", "123", "--kv-dump-prompt", "hello"]


def test_build_run_bench_command_localizes_drive_binary_paths(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "bench").mkdir(parents=True)
    drive_server = Path("/content/drive/MyDrive/tq_vlm_bench/cache/llama_server/demo__llama.cpp/abc1234/sm75/bin/llama-server")
    drive_kv_dump = Path("/content/drive/MyDrive/tq_vlm_bench/cache/llama_server/demo__llama.cpp/abc1234/sm75/bin/llama-kv-dump")

    def fake_localize(root: Path, path: Path) -> Path:
        assert root == repo_root.resolve()
        if path == drive_server:
            return repo_root / ".tq_bench_runtime" / "llama-server"
        if path == drive_kv_dump:
            return repo_root / ".tq_bench_runtime" / "llama-kv-dump"
        raise AssertionError(path)

    monkeypatch.setattr("tq_bench.colab._materialize_executable_path", fake_localize)

    cmd = build_run_bench_command(
        repo_root,
        num=1,
        model_id="qwen3_vl_2b_instruct",
        model_quant="q4_k_m",
        runtimes=["baseline"],
        benchmarks=["ai2d"],
        profile="colab",
        server_binary_path=drive_server,
        kv_dump_binary_path=drive_kv_dump,
    )

    joined = " ".join(cmd)
    assert str(repo_root / ".tq_bench_runtime") in joined
    assert str(drive_server) not in joined


def test_detect_llama_binary_prefers_colab_build_for_colab_profile(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    build_bin = repo_root / "llama.cpp" / "build" / "bin"
    build_colab_bin = repo_root / "llama.cpp" / "build-colab" / "bin"
    build_bin.mkdir(parents=True)
    build_colab_bin.mkdir(parents=True)
    (build_bin / "llama-server").write_text("", encoding="utf-8")
    (build_colab_bin / "llama-server").write_text("", encoding="utf-8")

    assert _detect_llama_binary(repo_root, "llama-server", profile="colab") == (
        build_colab_bin / "llama-server"
    )
    assert _detect_llama_binary(repo_root, "llama-server", profile="local") == (
        build_bin / "llama-server"
    )


def test_detect_cuda_architecture_fallback_supports_colab_gpu_names(monkeypatch) -> None:
    cases = [
        ("Tesla T4", "75"),
        ("NVIDIA L40S", "89"),
        ("NVIDIA A100-SXM4-40GB", "80"),
        ("NVIDIA H100 80GB HBM3", "90"),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "120"),
    ]

    for name, expected_arch in cases:
        monkeypatch.setattr("tq_bench.colab._find_tool", lambda tool_name, extra_candidates=None: tool_name)
        monkeypatch.setattr("tq_bench.colab._detect_cuda_via_torch", lambda: (None, None))

        def fake_run_text(cmd, *, cwd=None, env=None, gpu_name=name):
            del cwd, env
            if "--query-gpu=compute_cap" in cmd[1]:
                raise RuntimeError("compute capability unsupported")
            if "--query-gpu=name" in cmd[1]:
                return gpu_name
            raise AssertionError(cmd)

        monkeypatch.setattr("tq_bench.colab._run_text", fake_run_text)
        assert detect_cuda_architecture() == expected_arch


def test_detect_cuda_architecture_uses_compute_cap_when_available(monkeypatch) -> None:
    def fake_run_text(cmd, *, cwd=None, env=None):
        del cwd, env
        if "--query-gpu=compute_cap" in cmd[1]:
            return "12.0"
        raise AssertionError(cmd)

    monkeypatch.setattr("tq_bench.colab._find_tool", lambda name, extra_candidates=None: name)
    monkeypatch.setattr("tq_bench.colab._run_text", fake_run_text)
    assert detect_cuda_architecture() == "120"


def test_detect_cuda_architecture_falls_back_to_torch(monkeypatch) -> None:
    monkeypatch.setattr("tq_bench.colab._find_tool", lambda name, extra_candidates=None: None)
    monkeypatch.setattr("tq_bench.colab._detect_cuda_via_torch", lambda: ("NVIDIA H100 80GB HBM3", "90"))
    assert detect_cuda_architecture() == "90"


def test_detect_cuda_architecture_raises_clear_error_without_gpu(monkeypatch) -> None:
    monkeypatch.setattr("tq_bench.colab._find_tool", lambda name, extra_candidates=None: None)
    monkeypatch.setattr("tq_bench.colab._detect_cuda_via_torch", lambda: (None, None))

    try:
        detect_cuda_architecture()
    except RuntimeError as exc:
        assert "nvidia-smi" in str(exc)
        assert "runtime to GPU" in str(exc)
    else:
        raise AssertionError("detect_cuda_architecture should have raised RuntimeError")


def test_nvcc_version_uses_common_cuda_path(monkeypatch) -> None:
    monkeypatch.setattr("tq_bench.colab._find_tool", lambda name, extra_candidates=None: "/usr/local/cuda/bin/nvcc")
    monkeypatch.setattr("tq_bench.colab._run_text", lambda cmd, *, cwd=None, env=None: "Cuda compilation tools, release 12.8")
    assert "12.8" in _nvcc_version()


def test_nvcc_version_raises_clear_error_when_missing(monkeypatch) -> None:
    monkeypatch.setattr("tq_bench.colab._find_tool", lambda name, extra_candidates=None: None)
    try:
        _nvcc_version()
    except RuntimeError as exc:
        assert "nvcc" in str(exc)
        assert "CUDA toolkit" in str(exc)
    else:
        raise AssertionError("_nvcc_version should have raised RuntimeError")


def test_install_bench_editable_retries_with_break_system_packages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    bench_dir = repo_root / "bench"
    bench_dir.mkdir(parents=True)
    _write_minimal_bench_pyproject(bench_dir)
    calls: list[list[str]] = []

    def fake_run(cmd, check, capture_output, text):
        del check, capture_output, text
        calls.append(cmd)
        if "--break-system-packages" in cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="error: externally-managed-environment",
        )

    monkeypatch.setattr("tq_bench.colab.subprocess.run", fake_run)
    install_bench_editable(repo_root)

    assert len(calls) == 2
    assert "--break-system-packages" not in calls[0]
    assert "--break-system-packages" in calls[1]


def test_install_bench_editable_raises_with_pip_output(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    bench_dir = repo_root / "bench"
    bench_dir.mkdir(parents=True)
    _write_minimal_bench_pyproject(bench_dir)

    def fake_run(cmd, check, capture_output, text):
        del cmd, check, capture_output, text
        return SimpleNamespace(returncode=1, stdout="pip stdout", stderr="pip stderr")

    monkeypatch.setattr("tq_bench.colab.subprocess.run", fake_run)

    try:
        install_bench_editable(repo_root)
    except RuntimeError as exc:
        msg = str(exc)
        assert "pip stdout" in msg
        assert "pip stderr" in msg
    else:
        raise AssertionError("install_bench_editable should have raised RuntimeError")


def test_install_bench_editable_falls_back_to_dependency_install(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    bench_dir = repo_root / "bench"
    bench_dir.mkdir(parents=True)
    _write_minimal_bench_pyproject(bench_dir)
    calls: list[list[str]] = []

    def fake_run(cmd, check, capture_output, text):
        del check, capture_output, text
        calls.append(cmd)
        if "-e" in cmd:
            return SimpleNamespace(returncode=1, stdout="", stderr="metadata-generation-failed")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("tq_bench.colab.subprocess.run", fake_run)

    install_bench_editable(repo_root)

    assert len(calls) == 2
    assert "-e" in calls[0]
    assert "pandas>=2.0.0" in calls[1]


def test_install_bench_editable_always_adds_bench_to_sys_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    bench_dir = repo_root / "bench"
    bench_dir.mkdir(parents=True)
    _write_minimal_bench_pyproject(bench_dir)

    def fake_run(cmd, check, capture_output, text):
        del cmd, check, capture_output, text
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("tq_bench.colab.subprocess.run", fake_run)
    before = list(__import__("sys").path)

    install_bench_editable(repo_root)

    assert str(bench_dir.resolve()) in __import__("sys").path
    __import__("sys").path[:] = before


def test_run_command_live_streams_output(capsys) -> None:
    run_command_live(
        ["python3", "-c", "print('hello from child')"],
        step="demo",
    )
    out = capsys.readouterr().out
    assert "hello from child" in out
    assert "[tq-bench] demo:" in out


def test_run_command_live_raises_on_failure() -> None:
    try:
        run_command_live(["python3", "-c", "import sys; sys.exit(3)"])
    except RuntimeError as exc:
        assert "exit code 3" in str(exc)
    else:
        raise AssertionError("run_command_live should have raised RuntimeError")


def test_materialize_local_binaries_copies_and_marks_executable(tmp_path: Path) -> None:
    source = tmp_path / "cache" / "bin"
    source.mkdir(parents=True)
    server = source / "llama-server"
    server.write_text("x", encoding="utf-8")
    server.chmod(0o644)

    out = _materialize_local_binaries(
        tmp_path / "repo",
        repo_slug="demo__llama.cpp",
        llama_commit="abc1234",
        cuda_arch="75",
        source_lib_dir=source,
    )

    copied = out / "llama-server"
    assert copied.exists()
    assert copied != server
    assert copied.stat().st_mode & 0o111


def test_ensure_llama_server_uses_local_exec_copy_for_cached_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    llama_root = repo_root / "llama.cpp"
    (llama_root / ".git").mkdir(parents=True)
    drive_root = tmp_path / "drive"
    cache_dir = drive_root / "cache" / "llama_server" / "demo__llama.cpp" / "abc1234" / "sm75"
    cached_lib_dir = cache_dir / "bin"
    cached_lib_dir.mkdir(parents=True)
    (cached_lib_dir / "llama-server").write_text("server", encoding="utf-8")
    (cached_lib_dir / "llama-kv-dump").write_text("dump", encoding="utf-8")
    (cache_dir / "manifest.json").write_text(
        """
{
  "llama_repo_url": "https://github.com/demo/llama.cpp.git",
  "cuda_arch": "75",
  "llama_commit": "abc1234",
  "nvcc_version": "Cuda compilation tools, release 12.8",
  "build_type": "Release"
}
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr("tq_bench.colab.detect_cuda_architecture", lambda: "75")
    monkeypatch.setattr("tq_bench.colab._llama_commit", lambda llama_root: "abc1234")
    monkeypatch.setattr(
        "tq_bench.colab._nvcc_version",
        lambda: "Cuda compilation tools, release 12.8",
    )
    monkeypatch.setattr(
        "tq_bench.colab._git_remote_url",
        lambda repo_root: "https://github.com/demo/llama.cpp.git",
    )

    artifact = ensure_llama_server(repo_root, drive_root)

    assert str(artifact.binary_path).startswith(str(repo_root))
    assert str(artifact.kv_dump_binary_path).startswith(str(repo_root))
    assert artifact.binary_path.exists()
    assert artifact.kv_dump_binary_path.exists()


def test_find_latest_run_file_prefers_latest_matching_model(tmp_path: Path) -> None:
    older = tmp_path / "bench_model_a_old.json"
    newer = tmp_path / "bench_model_a_new.json"
    other = tmp_path / "bench_model_b.json"
    for idx, path in enumerate((older, other, newer)):
        path.write_text("{}", encoding="utf-8")
        ts = time.time() + idx
        os.utime(path, (ts, ts))

    assert find_latest_run_file(tmp_path, model_id="model_a") == newer
    assert find_latest_run_file(tmp_path) == newer


def test_repo_slug_from_url_is_stable() -> None:
    assert _repo_slug_from_url("https://github.com/Chedrian07/llama.cpp.git") == "Chedrian07__llama.cpp"
