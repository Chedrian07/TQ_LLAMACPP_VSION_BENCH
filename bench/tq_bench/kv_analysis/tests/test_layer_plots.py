"""Tests for :mod:`tq_bench.kv_analysis.layer_plots`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tq_bench.kv_analysis.layer_plots import (
    plot_all_layer_curves,
    plot_kv_norm_ratio_curve,
    plot_layer_cosine_curves,
    plot_layer_distortion_curves,
    plot_layer_relative_error_curves,
)


# ---------------------------------------------------------------------------
# Fixtures: mock DataFrames matching the expected schemas
# ---------------------------------------------------------------------------

N_LAYERS = 28  # Qwen3-VL-2B


def _make_dist_df() -> pd.DataFrame:
    """Mock DataFrame matching compute_per_layer_stats output.

    Columns: run, layer, token_type, k_norm_mean, v_norm_mean, kv_norm_ratio.
    """
    rng = np.random.default_rng(42)
    rows = []
    for layer in range(N_LAYERS):
        for token_type in ("all", "vision", "text"):
            k_norm = 1.0 + 0.1 * rng.standard_normal()
            v_norm = 0.8 + 0.1 * rng.standard_normal()
            rows.append({
                "run": "baseline",
                "layer": layer,
                "token_type": token_type,
                "n_tokens": 16,
                "k_norm_mean": k_norm,
                "v_norm_mean": v_norm,
                "kv_norm_ratio": k_norm / v_norm if v_norm > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


def _make_quant_df(runtimes: list[str] | None = None) -> pd.DataFrame:
    """Mock DataFrame matching compare_dumps output.

    Columns: run, layer, token_type, kind, per_coord_mse, cosine_sim,
    relative_error.
    """
    if runtimes is None:
        runtimes = ["tq-3", "tq-4"]
    rng = np.random.default_rng(99)
    rows = []
    for runtime in runtimes:
        for layer in range(N_LAYERS):
            for token_type in ("all", "vision", "text"):
                for kind in ("K", "V"):
                    mse = 0.03 + 0.005 * rng.standard_normal()
                    rows.append({
                        "run": runtime,
                        "layer": layer,
                        "token_type": token_type,
                        "kind": kind,
                        "per_coord_mse": abs(mse),
                        "cosine_sim": 0.99 + 0.005 * rng.standard_normal(),
                        "relative_error": abs(0.05 + 0.01 * rng.standard_normal()),
                        "max_abs_error": abs(0.1 + 0.02 * rng.standard_normal()),
                        "inner_product_bias": 1.0 + 0.01 * rng.standard_normal(),
                        "mean_baseline_norm": abs(1.0 + 0.1 * rng.standard_normal()),
                    })
    return pd.DataFrame(rows)


@pytest.fixture
def dist_df() -> pd.DataFrame:
    return _make_dist_df()


@pytest.fixture
def quant_df() -> pd.DataFrame:
    return _make_quant_df()


# ---------------------------------------------------------------------------
# Tests: KV norm ratio curve
# ---------------------------------------------------------------------------


class TestPlotKvNormRatioCurve:
    def test_creates_png(self, dist_df: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "kv_norm_ratio.png"
        plot_kv_norm_ratio_curve(dist_df, out_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_returns_figure_without_saving(self, dist_df: pd.DataFrame) -> None:
        import matplotlib.pyplot as plt
        fig = plot_kv_norm_ratio_curve(dist_df, out_path=None)
        assert fig is not None
        plt.close(fig)

    def test_handles_empty_df(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.png"
        plot_kv_norm_ratio_curve(pd.DataFrame(), out_path=out)
        assert out.exists()

    def test_handles_no_run_column(self, tmp_path: Path) -> None:
        df = _make_dist_df().drop(columns=["run"])
        out = tmp_path / "no_run.png"
        plot_kv_norm_ratio_curve(df, out_path=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Tests: layer distortion curves
# ---------------------------------------------------------------------------


class TestPlotLayerDistortionCurves:
    def test_creates_png(self, quant_df: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "distortion.png"
        plot_layer_distortion_curves(quant_df, out_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_returns_figure_without_saving(self, quant_df: pd.DataFrame) -> None:
        import matplotlib.pyplot as plt
        fig = plot_layer_distortion_curves(quant_df, out_path=None)
        assert fig is not None
        plt.close(fig)

    def test_handles_empty_df(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.png"
        plot_layer_distortion_curves(pd.DataFrame(), out_path=out)
        assert out.exists()

    def test_single_runtime(self, tmp_path: Path) -> None:
        df = _make_quant_df(runtimes=["tq-3"])
        out = tmp_path / "single.png"
        plot_layer_distortion_curves(df, out_path=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Tests: layer cosine curves
# ---------------------------------------------------------------------------


class TestPlotLayerCosineCurves:
    def test_creates_png(self, quant_df: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "cosine.png"
        plot_layer_cosine_curves(quant_df, out_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_handles_empty_df(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.png"
        plot_layer_cosine_curves(pd.DataFrame(), out_path=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Tests: layer relative error curves
# ---------------------------------------------------------------------------


class TestPlotLayerRelativeErrorCurves:
    def test_creates_png(self, quant_df: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "relerr.png"
        plot_layer_relative_error_curves(quant_df, out_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_handles_empty_df(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.png"
        plot_layer_relative_error_curves(pd.DataFrame(), out_path=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Tests: convenience batch
# ---------------------------------------------------------------------------


class TestPlotAllLayerCurves:
    def test_creates_all_pngs(
        self,
        dist_df: pd.DataFrame,
        quant_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        written = plot_all_layer_curves(dist_df, quant_df, tmp_path)
        assert len(written) == 4
        for p in written:
            assert p.exists()
            assert p.suffix == ".png"
            assert p.stat().st_size > 0

    def test_creates_plots_subdir(
        self,
        dist_df: pd.DataFrame,
        quant_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        plot_all_layer_curves(dist_df, quant_df, tmp_path)
        assert (tmp_path / "plots").is_dir()

    def test_with_empty_dfs(self, tmp_path: Path) -> None:
        written = plot_all_layer_curves(pd.DataFrame(), pd.DataFrame(), tmp_path)
        assert len(written) == 4
        for p in written:
            assert p.exists()

    def test_no_run_column(self, tmp_path: Path) -> None:
        dist = _make_dist_df().drop(columns=["run"])
        quant = _make_quant_df().drop(columns=["run"])
        written = plot_all_layer_curves(dist, quant, tmp_path)
        assert len(written) == 4
        for p in written:
            assert p.exists()
