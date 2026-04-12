"""Tests for :mod:`tq_bench.kv_analysis.distribution`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tq_bench.kv_analysis.distribution import (
    compute_norm_stats,
    compute_per_layer_stats,
    compute_value_stats,
)
from tq_bench.kv_analysis.tests.conftest import (
    SyntheticDumpSpec,
    build_synthetic_dump,
)


def test_value_stats_known_input() -> None:
    arr = np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
    stats = compute_value_stats(arr)
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["min"] == pytest.approx(-1.0)
    assert stats["max"] == pytest.approx(2.0)
    assert stats["abs_max"] == pytest.approx(2.0)
    # std (population) of [-1, 0, 1, 2] = sqrt(((-1.5)^2+(-0.5)^2+0.5^2+1.5^2)/4)
    expected_std = float(np.std(arr, ddof=0))
    assert stats["std"] == pytest.approx(expected_std)


def test_value_stats_empty() -> None:
    stats = compute_value_stats(np.zeros((0,)))
    for v in stats.values():
        assert np.isnan(v)


def test_norm_stats_unit_vectors() -> None:
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((100, 8))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    stats = compute_norm_stats(vecs)
    assert stats["norm_mean"] == pytest.approx(1.0, abs=1e-6)
    assert stats["norm_std"] == pytest.approx(0.0, abs=1e-6)


def test_norm_stats_3d_input_flattens() -> None:
    rng = np.random.default_rng(0)
    # (4 tokens, 2 heads, 8 dim), norm should be across H*D = 16.
    tensor = rng.standard_normal((4, 2, 8))
    stats = compute_norm_stats(tensor)
    # Manually compute expected:
    flat = tensor.reshape(4, 16)
    expected = np.linalg.norm(flat, axis=1).mean()
    assert stats["norm_mean"] == pytest.approx(float(expected))


def test_per_layer_stats_shape_and_columns(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=20, n_layers=3, n_kv_head=2, head_dim=16, vision_token_count=10
    )
    dump, _, _ = build_synthetic_dump(tmp_path / "per_layer", spec)
    df = compute_per_layer_stats(dump, separate_vision_text=True)
    assert isinstance(df, pd.DataFrame)
    assert set(df["token_type"].unique()) >= {"all", "vision", "text"}
    assert len(df) == spec.n_layers * 3  # all + vision + text per layer

    # Required columns
    needed = [
        "layer",
        "token_type",
        "k_mean",
        "k_std",
        "k_norm_mean",
        "v_mean",
        "v_norm_mean",
        "kv_norm_ratio",
    ]
    for col in needed:
        assert col in df.columns, f"missing column: {col}"


def test_per_layer_stats_no_vision_tokens(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=10, n_layers=2, n_kv_head=1, head_dim=8, vision_token_count=0
    )
    dump, _, _ = build_synthetic_dump(tmp_path / "no_vision", spec)
    df = compute_per_layer_stats(dump, separate_vision_text=True)
    assert "vision" not in df["token_type"].unique()
    assert "text" in df["token_type"].unique()
    assert "all" in df["token_type"].unique()
