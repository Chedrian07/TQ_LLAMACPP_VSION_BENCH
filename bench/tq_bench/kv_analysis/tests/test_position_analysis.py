"""Tests for :mod:`tq_bench.kv_analysis.position_analysis`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tq_bench.kv_analysis.position_analysis import (
    compute_per_token_norms,
    per_position_outlier_ratio,
    per_position_quant_error,
    plot_token_norm_vs_position,
    plot_position_outlier_heatmap,
    plot_position_quant_error,
    token_index_vs_norm_df,
)
from tq_bench.kv_analysis.tests.conftest import (
    SyntheticDumpSpec,
    build_pair_of_dumps,
    build_synthetic_dump,
)


# ---------------------------------------------------------------------------
# compute_per_token_norms
# ---------------------------------------------------------------------------


def test_compute_per_token_norms_shape(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(n_tokens=10, n_layers=2, n_kv_head=2, head_dim=16)
    dump, _, _ = build_synthetic_dump(tmp_path / "norms", spec)
    norms = compute_per_token_norms(dump, kind="K", layer=0)
    assert norms.shape == (spec.n_tokens,)
    assert norms.dtype == np.float64


def test_compute_per_token_norms_positive(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(n_tokens=8, n_layers=1, n_kv_head=1, head_dim=8)
    dump, _, _ = build_synthetic_dump(tmp_path / "norms_pos", spec)
    norms = compute_per_token_norms(dump, kind="V", layer=0)
    assert np.all(norms >= 0.0)


def test_compute_per_token_norms_invalid_kind(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(n_tokens=4, n_layers=1, n_kv_head=1, head_dim=4)
    dump, _, _ = build_synthetic_dump(tmp_path / "norms_bad", spec)
    with pytest.raises(ValueError, match="kind must be"):
        compute_per_token_norms(dump, kind="X", layer=0)


# ---------------------------------------------------------------------------
# token_index_vs_norm_df
# ---------------------------------------------------------------------------


def test_token_index_vs_norm_df_columns(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=6, n_layers=2, n_kv_head=1, head_dim=8, vision_token_count=3
    )
    dump, _, _ = build_synthetic_dump(tmp_path / "tivn", spec)
    df = token_index_vs_norm_df(dump, kind="K")
    assert set(df.columns) == {"layer", "token_idx", "norm", "is_vision"}
    assert len(df) == spec.n_layers * spec.n_tokens


def test_token_index_vs_norm_df_vision_text_separation(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=10, n_layers=1, n_kv_head=1, head_dim=4, vision_token_count=4
    )
    dump, _, _ = build_synthetic_dump(tmp_path / "tivn_sep", spec)
    df = token_index_vs_norm_df(dump, kind="K")
    vision_rows = df[df["is_vision"]]
    text_rows = df[~df["is_vision"]]
    assert len(vision_rows) == 4
    assert len(text_rows) == 6


def test_token_index_vs_norm_df_no_vision(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=4, n_layers=1, n_kv_head=1, head_dim=4, vision_token_count=0
    )
    dump, _, _ = build_synthetic_dump(tmp_path / "tivn_novis", spec)
    df = token_index_vs_norm_df(dump, kind="V")
    assert len(df) == 4
    assert not df["is_vision"].any()


# ---------------------------------------------------------------------------
# plot_token_norm_vs_position
# ---------------------------------------------------------------------------


def test_plot_token_norm_vs_position_creates_files(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=12, n_layers=2, n_kv_head=1, head_dim=8, vision_token_count=6
    )
    dump, _, _ = build_synthetic_dump(tmp_path / "plot_norm", spec)
    out_dir = tmp_path / "plots_norm"
    paths = plot_token_norm_vs_position(dump, kind="K", out_dir=out_dir)
    assert len(paths) == spec.n_layers
    for p in paths:
        assert p.exists()
        assert p.suffix == ".png"


def test_plot_token_norm_vs_position_subset_layers(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(n_tokens=8, n_layers=4, n_kv_head=1, head_dim=4)
    dump, _, _ = build_synthetic_dump(tmp_path / "plot_sub", spec)
    out_dir = tmp_path / "plots_sub"
    paths = plot_token_norm_vs_position(dump, kind="V", layers=[0, 2], out_dir=out_dir)
    assert len(paths) == 2


def test_plot_token_norm_vs_position_no_vision(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=6, n_layers=1, n_kv_head=1, head_dim=4, vision_token_count=0
    )
    dump, _, _ = build_synthetic_dump(tmp_path / "plot_novis", spec)
    out_dir = tmp_path / "plots_novis"
    paths = plot_token_norm_vs_position(dump, kind="K", out_dir=out_dir)
    assert len(paths) == 1
    assert paths[0].exists()


# ---------------------------------------------------------------------------
# per_position_outlier_ratio
# ---------------------------------------------------------------------------


def test_per_position_outlier_ratio_columns(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=10, n_layers=2, n_kv_head=2, head_dim=8, vision_token_count=5
    )
    dump, _, _ = build_synthetic_dump(tmp_path / "ppo", spec)
    df = per_position_outlier_ratio(dump, kind="K")
    assert set(df.columns) == {"layer", "token_idx", "is_vision", "outlier_ratio"}
    assert len(df) == spec.n_layers * spec.n_tokens


def test_per_position_outlier_ratio_range(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=8, n_layers=1, n_kv_head=1, head_dim=16, vision_token_count=4
    )
    dump, _, _ = build_synthetic_dump(tmp_path / "ppo_range", spec)
    df = per_position_outlier_ratio(dump, kind="V", threshold=10.0)
    assert df["outlier_ratio"].min() >= 0.0
    assert df["outlier_ratio"].max() <= 1.0


def test_per_position_outlier_ratio_detects_spike(tmp_path: Path) -> None:
    """Inject a token with extreme values and verify it gets a higher ratio."""
    spec = SyntheticDumpSpec(
        n_tokens=16, n_layers=1, n_kv_head=1, head_dim=32, vision_token_count=0
    )
    dump, K, _ = build_synthetic_dump(tmp_path / "ppo_spike", spec)
    # Overwrite token 0 in layer 0 with 100x magnitude.
    K[0][0, :, :] *= 100.0
    # Rewrite dump with the spiked data.
    from tq_bench.kv_analysis.loader import KVDumpWriter
    writer = KVDumpWriter(tmp_path / "ppo_spike_mod")
    writer.write(
        K=K,
        V={0: np.ones((16, 1, 32), dtype=np.float32)},
        vision_token_mask=[False] * 16,
        run_name="spiked",
    )
    from tq_bench.kv_analysis.loader import KVDump
    spiked_dump = KVDump(tmp_path / "ppo_spike_mod")
    df = per_position_outlier_ratio(spiked_dump, kind="K", threshold=5.0)
    # Token 0 should have the highest outlier ratio.
    token0_ratio = df[df["token_idx"] == 0]["outlier_ratio"].iloc[0]
    other_max = df[df["token_idx"] != 0]["outlier_ratio"].max()
    assert token0_ratio > other_max


# ---------------------------------------------------------------------------
# plot_position_outlier_heatmap
# ---------------------------------------------------------------------------


def test_plot_position_outlier_heatmap_creates_file(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=12, n_layers=3, n_kv_head=1, head_dim=8, vision_token_count=6
    )
    dump, _, _ = build_synthetic_dump(tmp_path / "heatmap", spec)
    out_path = tmp_path / "heatmap.png"
    result = plot_position_outlier_heatmap(dump, kind="K", out_path=out_path)
    assert result == out_path
    assert out_path.exists()


def test_plot_position_outlier_heatmap_default_path(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(n_tokens=6, n_layers=2, n_kv_head=1, head_dim=4)
    dump, _, _ = build_synthetic_dump(tmp_path / "heatmap_def", spec)
    result = plot_position_outlier_heatmap(dump, kind="V")
    assert result.exists()
    assert result.suffix == ".png"


# ---------------------------------------------------------------------------
# per_position_quant_error
# ---------------------------------------------------------------------------


def test_per_position_quant_error_columns(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=10, n_layers=2, n_kv_head=2, head_dim=8, vision_token_count=5
    )
    baseline, quant = build_pair_of_dumps(tmp_path, spec, noise=0.05)
    df = per_position_quant_error(baseline, quant, kind="K")
    assert set(df.columns) == {
        "layer", "token_idx", "is_vision", "per_coord_mse", "cosine_sim"
    }
    assert len(df) == spec.n_layers * spec.n_tokens


def test_per_position_quant_error_identity(tmp_path: Path) -> None:
    """When baseline == quantized, MSE should be zero and cosine 1."""
    spec = SyntheticDumpSpec(n_tokens=4, n_layers=1, n_kv_head=1, head_dim=8)
    dump, _, _ = build_synthetic_dump(tmp_path / "qe_id", spec)
    # Use the same dump as both baseline and quantized.
    df = per_position_quant_error(dump, dump, kind="K")
    assert df["per_coord_mse"].max() == pytest.approx(0.0, abs=1e-8)
    assert df["cosine_sim"].min() == pytest.approx(1.0, abs=1e-6)


def test_per_position_quant_error_vision_text(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=8, n_layers=1, n_kv_head=1, head_dim=16, vision_token_count=4
    )
    baseline, quant = build_pair_of_dumps(tmp_path, spec, noise=0.1)
    df = per_position_quant_error(baseline, quant, kind="V")
    vision_rows = df[df["is_vision"]]
    text_rows = df[~df["is_vision"]]
    assert len(vision_rows) == 4
    assert len(text_rows) == 4


def test_per_position_quant_error_shape_mismatch(tmp_path: Path) -> None:
    spec_a = SyntheticDumpSpec(n_tokens=8, n_layers=1, n_kv_head=1, head_dim=8)
    spec_b = SyntheticDumpSpec(n_tokens=10, n_layers=1, n_kv_head=1, head_dim=8, seed=999)
    dump_a, _, _ = build_synthetic_dump(tmp_path / "qe_a", spec_a)
    dump_b, _, _ = build_synthetic_dump(tmp_path / "qe_b", spec_b)
    with pytest.raises(ValueError, match="shape mismatch"):
        per_position_quant_error(dump_a, dump_b, kind="K")


# ---------------------------------------------------------------------------
# plot_position_quant_error
# ---------------------------------------------------------------------------


def test_plot_position_quant_error_creates_files(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=12, n_layers=2, n_kv_head=1, head_dim=8, vision_token_count=6
    )
    baseline, quant = build_pair_of_dumps(tmp_path, spec, noise=0.05)
    out_dir = tmp_path / "plots_qe"
    paths = plot_position_quant_error(baseline, quant, kind="K", out_dir=out_dir)
    assert len(paths) == spec.n_layers
    for p in paths:
        assert p.exists()
        assert p.suffix == ".png"


def test_plot_position_quant_error_subset_layers(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(n_tokens=8, n_layers=4, n_kv_head=1, head_dim=4)
    baseline, quant = build_pair_of_dumps(tmp_path, spec, noise=0.02)
    out_dir = tmp_path / "plots_qe_sub"
    paths = plot_position_quant_error(
        baseline, quant, kind="V", layers=[1, 3], out_dir=out_dir
    )
    assert len(paths) == 2


def test_plot_position_quant_error_no_vision(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=6, n_layers=1, n_kv_head=1, head_dim=4, vision_token_count=0
    )
    baseline, quant = build_pair_of_dumps(tmp_path, spec, noise=0.01)
    out_dir = tmp_path / "plots_qe_novis"
    paths = plot_position_quant_error(baseline, quant, kind="K", out_dir=out_dir)
    assert len(paths) == 1
    assert paths[0].exists()
