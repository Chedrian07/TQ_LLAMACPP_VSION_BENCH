"""Tests for :mod:`tq_bench.kv_analysis.visualization`."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
import numpy as np
import pytest

from tq_bench.kv_analysis.visualization import (
    generate_all_histograms,
    plot_prerot_vs_postrot,
    plot_value_histogram,
    plot_vision_vs_text_histogram,
)
from tq_bench.kv_analysis.tests.conftest import (
    SyntheticDumpSpec,
    build_synthetic_dump,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def spec_with_vision() -> SyntheticDumpSpec:
    """Spec with both vision and text tokens."""
    return SyntheticDumpSpec(
        n_tokens=16,
        n_layers=3,
        n_kv_head=2,
        head_dim=128,
        vision_token_count=8,
        seed=42,
    )


@pytest.fixture
def spec_no_vision() -> SyntheticDumpSpec:
    """Spec where all tokens are text (vision_token_count=0)."""
    return SyntheticDumpSpec(
        n_tokens=16,
        n_layers=3,
        n_kv_head=2,
        head_dim=128,
        vision_token_count=0,
        seed=42,
    )


@pytest.fixture
def dump_with_vision(tmp_path: Path, spec_with_vision: SyntheticDumpSpec):
    dump, _, _ = build_synthetic_dump(tmp_path / "vis", spec_with_vision)
    return dump


@pytest.fixture
def dump_no_vision(tmp_path: Path, spec_no_vision: SyntheticDumpSpec):
    dump, _, _ = build_synthetic_dump(tmp_path / "novis", spec_no_vision)
    return dump


# ---------------------------------------------------------------------------
# plot_value_histogram
# ---------------------------------------------------------------------------


class TestPlotValueHistogram:

    def test_returns_figure(self, dump_with_vision) -> None:
        fig = plot_value_histogram(dump_with_vision, layer=0, kind="K")
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_png(self, dump_with_vision, tmp_path: Path) -> None:
        out = tmp_path / "hist.png"
        plot_value_histogram(dump_with_vision, layer=0, kind="V", out_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_kind_v(self, dump_with_vision) -> None:
        fig = plot_value_histogram(dump_with_vision, layer=1, kind="V")
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_token_type_vision(self, dump_with_vision) -> None:
        fig = plot_value_histogram(
            dump_with_vision, layer=0, kind="K", token_type="vision",
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_token_type_text(self, dump_with_vision) -> None:
        fig = plot_value_histogram(
            dump_with_vision, layer=0, kind="K", token_type="text",
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_invalid_kind_raises(self, dump_with_vision) -> None:
        with pytest.raises(ValueError, match="kind"):
            plot_value_histogram(dump_with_vision, layer=0, kind="X")

    def test_empty_vision_mask(self, dump_no_vision, tmp_path: Path) -> None:
        """When there are no vision tokens the plot should not crash."""
        out = tmp_path / "empty_vis.png"
        fig = plot_value_histogram(
            dump_no_vision, layer=0, kind="K",
            token_type="vision", out_path=out,
        )
        # Figure saved without error.
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_prerot_vs_postrot
# ---------------------------------------------------------------------------


class TestPlotPrerotVsPostrot:

    def test_returns_figure(self, dump_with_vision) -> None:
        fig = plot_prerot_vs_postrot(dump_with_vision, layer=0, kind="K")
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_png(self, dump_with_vision, tmp_path: Path) -> None:
        out = tmp_path / "prerot.png"
        plot_prerot_vs_postrot(dump_with_vision, layer=0, kind="V", out_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_kind_v(self, dump_with_vision) -> None:
        fig = plot_prerot_vs_postrot(dump_with_vision, layer=2, kind="V")
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_invalid_kind_raises(self, dump_with_vision) -> None:
        with pytest.raises(ValueError, match="kind"):
            plot_prerot_vs_postrot(dump_with_vision, layer=0, kind="Z")


# ---------------------------------------------------------------------------
# plot_vision_vs_text_histogram
# ---------------------------------------------------------------------------


class TestPlotVisionVsTextHistogram:

    def test_returns_figure(self, dump_with_vision) -> None:
        fig = plot_vision_vs_text_histogram(
            dump_with_vision, layer=0, kind="K",
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_png(self, dump_with_vision, tmp_path: Path) -> None:
        out = tmp_path / "vt.png"
        plot_vision_vs_text_histogram(
            dump_with_vision, layer=0, kind="V", out_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_no_vision_tokens(self, dump_no_vision, tmp_path: Path) -> None:
        """Should not crash; logs a warning, skips vision distribution."""
        out = tmp_path / "novision.png"
        fig = plot_vision_vs_text_histogram(
            dump_no_vision, layer=0, kind="K", out_path=out,
        )
        assert out.exists()

    def test_kind_default_is_K(self, dump_with_vision) -> None:
        fig = plot_vision_vs_text_histogram(dump_with_vision, layer=1)
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# generate_all_histograms
# ---------------------------------------------------------------------------


class TestGenerateAllHistograms:

    def test_writes_pngs(self, dump_with_vision, tmp_path: Path) -> None:
        out_dir = tmp_path / "all"
        paths = generate_all_histograms(
            dump_with_vision, out_dir, layers=[0, 2], bins=50,
        )
        assert len(paths) > 0
        for p in paths:
            assert p.exists(), f"missing: {p}"
            assert p.suffix == ".png"

    def test_expected_count(self, dump_with_vision, tmp_path: Path) -> None:
        """2 layers x 2 kinds x 3 plot types = 12 PNGs."""
        out_dir = tmp_path / "cnt"
        paths = generate_all_histograms(
            dump_with_vision, out_dir, layers=[0, 1], bins=50,
        )
        assert len(paths) == 2 * 2 * 3  # 12

    def test_default_layers_clamped(self, dump_with_vision, tmp_path: Path) -> None:
        """Default layers are [0,7,14,21,27] but dump has only 3 layers."""
        out_dir = tmp_path / "def"
        paths = generate_all_histograms(
            dump_with_vision, out_dir, bins=50,
        )
        # Only layer 0 is valid from the default set (7, 14, 21, 27 > 3).
        assert len(paths) == 1 * 2 * 3  # 6

    def test_empty_layers_list(self, dump_with_vision, tmp_path: Path) -> None:
        paths = generate_all_histograms(
            dump_with_vision, tmp_path / "e", layers=[], bins=50,
        )
        assert paths == []

    def test_out_of_range_layers_skipped(
        self, dump_with_vision, tmp_path: Path,
    ) -> None:
        paths = generate_all_histograms(
            dump_with_vision, tmp_path / "oor", layers=[99], bins=50,
        )
        assert paths == []

    def test_histograms_subdir_created(
        self, dump_with_vision, tmp_path: Path,
    ) -> None:
        out_dir = tmp_path / "sub"
        generate_all_histograms(
            dump_with_vision, out_dir, layers=[0], bins=50,
        )
        assert (out_dir / "histograms").is_dir()
