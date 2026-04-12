"""Tests for :mod:`tq_bench.kv_analysis.outliers`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tq_bench.kv_analysis.outliers import (
    find_outlier_channels,
    outlier_ratio_per_layer,
    outlier_statistics,
)
from tq_bench.kv_analysis.tests.conftest import (
    SyntheticDumpSpec,
    build_synthetic_dump,
)


def test_find_outlier_channels_catches_large_column() -> None:
    rng = np.random.default_rng(42)
    tensor = rng.standard_normal((64, 4, 8)).astype(np.float32)
    # Boost one channel by a factor that beats the default threshold.
    tensor[:, 0, 3] *= 100.0
    flat_index = 0 * 8 + 3

    mask = find_outlier_channels(tensor, threshold=10.0)
    assert mask.shape == (4 * 8,)
    assert bool(mask[flat_index])
    # All other channels should be within normal range.
    assert int(mask.sum()) == 1


def test_outlier_statistics_fields() -> None:
    rng = np.random.default_rng(0)
    tensor = rng.standard_normal((32, 2, 4))
    stats = outlier_statistics(tensor)
    assert set(stats.keys()) == {
        "n_channels",
        "median_channel_norm",
        "max_channel_norm",
        "max_to_median_ratio",
        "n_outliers",
        "outlier_ratio",
    }
    assert stats["n_channels"] == 8
    assert stats["median_channel_norm"] > 0
    assert 0 <= stats["outlier_ratio"] <= 1


def test_outlier_ratio_per_layer_df(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=32, n_layers=3, n_kv_head=2, head_dim=16, vision_token_count=16
    )
    dump, _, _ = build_synthetic_dump(tmp_path / "outlier", spec)
    df = outlier_ratio_per_layer(dump)
    assert len(df) == spec.n_layers
    assert {"layer", "k_outlier_ratio", "v_outlier_ratio"}.issubset(df.columns)
    assert df["threshold"].iloc[0] == 10.0


def test_outlier_ratio_vision_subset(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=32, n_layers=2, n_kv_head=1, head_dim=8, vision_token_count=16
    )
    dump, _, _ = build_synthetic_dump(tmp_path / "outlier_vision", spec)
    df_all = outlier_ratio_per_layer(dump, token_type="all")
    df_vision = outlier_ratio_per_layer(dump, token_type="vision")
    assert len(df_all) == spec.n_layers
    assert len(df_vision) == spec.n_layers
    assert df_vision["token_type"].iloc[0] == "vision"


def test_outlier_ratio_returns_empty_when_mask_empty(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=8, n_layers=1, n_kv_head=1, head_dim=8, vision_token_count=0
    )
    dump, _, _ = build_synthetic_dump(tmp_path / "no_vision_out", spec)
    df = outlier_ratio_per_layer(dump, token_type="vision")
    assert df.empty
