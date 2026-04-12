"""Shared fixtures for the kv_analysis test-suite.

These helpers build synthetic KV dumps with known statistics so the
Python pipeline can be exercised before the C++ ``llama-kv-dump``
tool is available.  Anything that needs a dump on disk should use
:func:`build_synthetic_dump`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from tq_bench.kv_analysis.loader import KVDump, KVDumpWriter


@dataclass
class SyntheticDumpSpec:
    n_tokens: int = 16
    n_layers: int = 3
    n_kv_head: int = 2
    head_dim: int = 128
    vision_token_count: int = 8
    seed: int = 1234
    k_scale: float = 1.0
    v_scale: float = 1.0
    noise: float = 0.0  # added to quantized versions


def _make_tensor(
    rng: np.random.Generator,
    n_tokens: int,
    n_kv_head: int,
    head_dim: int,
    scale: float,
) -> np.ndarray:
    """Return a (T, H, D) float32 tensor sampled from ``N(0, scale^2)``."""
    return (rng.standard_normal(size=(n_tokens, n_kv_head, head_dim)) * scale).astype(
        np.float32
    )


def build_synthetic_dump(
    dump_dir: Path,
    spec: SyntheticDumpSpec | None = None,
    *,
    run_name: str = "synthetic",
) -> tuple[KVDump, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Create a synthetic dump directory and return the loaded KVDump.

    The returned dict objects are the exact numpy arrays written to
    disk so callers can compare against them.
    """
    spec = spec or SyntheticDumpSpec()
    rng = np.random.default_rng(spec.seed)

    K: dict[int, np.ndarray] = {}
    V: dict[int, np.ndarray] = {}
    for layer in range(spec.n_layers):
        K[layer] = _make_tensor(
            rng, spec.n_tokens, spec.n_kv_head, spec.head_dim, spec.k_scale
        )
        V[layer] = _make_tensor(
            rng, spec.n_tokens, spec.n_kv_head, spec.head_dim, spec.v_scale
        )

    mask = np.zeros(spec.n_tokens, dtype=bool)
    mask[: spec.vision_token_count] = True

    writer = KVDumpWriter(dump_dir)
    writer.write(
        K=K,
        V=V,
        vision_token_mask=mask,
        run_name=run_name,
        extra_meta={
            "model_path": "mock/Qwen3-VL-2B-Instruct",
            "cache_type_k": "f16",
            "cache_type_v": "f16",
        },
    )
    return KVDump(dump_dir), K, V


def build_pair_of_dumps(
    tmp_path: Path,
    spec: SyntheticDumpSpec,
    noise: float,
) -> tuple[KVDump, KVDump]:
    """Build a baseline dump and a noisy copy (simulating quantization)."""
    baseline_dump, K, V = build_synthetic_dump(tmp_path / "baseline", spec)

    rng = np.random.default_rng(spec.seed + 1)
    noisy_K = {layer: (arr + rng.standard_normal(arr.shape).astype(np.float32) * noise)
               for layer, arr in K.items()}
    noisy_V = {layer: (arr + rng.standard_normal(arr.shape).astype(np.float32) * noise)
               for layer, arr in V.items()}

    mask = np.zeros(spec.n_tokens, dtype=bool)
    mask[: spec.vision_token_count] = True
    writer = KVDumpWriter(tmp_path / "quant")
    writer.write(
        K=noisy_K,
        V=noisy_V,
        vision_token_mask=mask,
        run_name="quant",
    )
    quant_dump = KVDump(tmp_path / "quant")
    return baseline_dump, quant_dump


@pytest.fixture
def default_spec() -> SyntheticDumpSpec:
    return SyntheticDumpSpec()


@pytest.fixture
def synthetic_dump(tmp_path: Path, default_spec: SyntheticDumpSpec) -> KVDump:
    dump, _, _ = build_synthetic_dump(tmp_path / "dump", default_spec)
    return dump


@pytest.fixture
def pair_of_dumps(
    tmp_path: Path,
    default_spec: SyntheticDumpSpec,
) -> tuple[KVDump, KVDump]:
    return build_pair_of_dumps(tmp_path, default_spec, noise=0.01)
