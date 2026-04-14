from __future__ import annotations

from pathlib import Path

from tq_bench.config import RuntimeConfig
from tq_bench.server import LlamaServer, ServerLaunchConfig


def _make_server() -> LlamaServer:
    cfg = ServerLaunchConfig(
        binary_path=Path("/tmp/llama-server"),
        model_path=Path("/tmp/model.gguf"),
        port=18080,
    )
    return LlamaServer(cfg)


def _runtime(runtime_id: str, cache_k: str, cache_v: str) -> RuntimeConfig:
    return RuntimeConfig(
        id=runtime_id,
        method="test",
        cache_type_k=cache_k,
        cache_type_v=cache_v,
        bits="x",
    )


def test_build_command_disables_flash_attn_for_turbo_on_sm75(monkeypatch) -> None:
    monkeypatch.setattr("tq_bench.server._query_gpu_compute_cap", lambda gpu_id: 75)
    server = _make_server()
    cmd = server.build_command(_runtime("tq-4", "turbo4", "turbo4"), gpu_id=0)
    idx = cmd.index("-fa")
    assert cmd[idx + 1] == "on"


def test_build_command_keeps_flash_attn_for_turbo_on_sm120(monkeypatch) -> None:
    monkeypatch.setattr("tq_bench.server._query_gpu_compute_cap", lambda gpu_id: 120)
    server = _make_server()
    cmd = server.build_command(_runtime("tq-4", "turbo4", "turbo4"), gpu_id=0)
    idx = cmd.index("-fa")
    assert cmd[idx + 1] == "on"


def test_build_command_keeps_flash_attn_for_non_turbo(monkeypatch) -> None:
    monkeypatch.setattr("tq_bench.server._query_gpu_compute_cap", lambda gpu_id: 75)
    server = _make_server()
    cmd = server.build_command(_runtime("baseline", "f16", "f16"), gpu_id=0)
    idx = cmd.index("-fa")
    assert cmd[idx + 1] == "on"
