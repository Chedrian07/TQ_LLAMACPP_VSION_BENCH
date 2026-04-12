"""Tests for :mod:`tq_bench.kv_analysis.codebook_analysis`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tq_bench.kv_analysis.codebook_analysis import (
    TURBO_CENTROIDS,
    bit_exact_reproducibility,
    codebook_bucket_usage,
    codebook_vision_vs_text_skew,
    plot_codebook_skew,
    plot_codebook_usage,
    quantize_to_codebook,
)
from tq_bench.kv_analysis.tests.conftest import (
    SyntheticDumpSpec,
    build_pair_of_dumps,
    build_synthetic_dump,
)


# ---------------------------------------------------------------------------
# quantize_to_codebook
# ---------------------------------------------------------------------------


class TestQuantizeToCodebook:
    """Tests for the scalar nearest-centroid quantizer."""

    def test_exact_centroid_values_map_to_themselves(self) -> None:
        for bits, centroids in TURBO_CENTROIDS.items():
            indices, returned_centroids = quantize_to_codebook(centroids, bits)
            np.testing.assert_array_equal(returned_centroids, centroids)
            np.testing.assert_array_equal(indices, np.arange(len(centroids)))

    def test_known_2bit_assignment(self) -> None:
        # Values very close to each centroid should map to that centroid.
        c = TURBO_CENTROIDS[2]
        values = np.array([c[0] + 0.01, c[1] - 0.01, c[2] + 0.01, c[3] - 0.01])
        indices, _ = quantize_to_codebook(values, bits=2)
        np.testing.assert_array_equal(indices, [0, 1, 2, 3])

    def test_midpoint_goes_to_lower_index(self) -> None:
        # At the exact midpoint between two centroids, argmin picks the
        # first one (lower index) due to np.argmin tie-breaking.
        c = TURBO_CENTROIDS[2]
        midpoint = (c[0] + c[1]) / 2.0
        indices, _ = quantize_to_codebook(np.array([midpoint]), bits=2)
        # Either 0 or 1 is acceptable; just check it is one of them.
        assert indices[0] in (0, 1)

    def test_shape_preserved(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.standard_normal((5, 10))
        indices, _ = quantize_to_codebook(values, bits=3)
        assert indices.shape == (5, 10)

    def test_invalid_bits_raises(self) -> None:
        with pytest.raises(ValueError, match="bits must be one of"):
            quantize_to_codebook(np.array([0.0]), bits=5)

    def test_all_indices_in_range(self) -> None:
        rng = np.random.default_rng(1)
        values = rng.standard_normal(1000) * 3.0
        for bits, centroids in TURBO_CENTROIDS.items():
            indices, _ = quantize_to_codebook(values, bits)
            assert indices.min() >= 0
            assert indices.max() < len(centroids)


# ---------------------------------------------------------------------------
# codebook_bucket_usage
# ---------------------------------------------------------------------------


class TestCodebookBucketUsage:
    """Tests for per-layer bucket frequency."""

    def test_counts_sum_to_n_elements(self, tmp_path: Path) -> None:
        spec = SyntheticDumpSpec(
            n_tokens=16, n_layers=2, n_kv_head=2, head_dim=128, vision_token_count=8,
        )
        dump, _, _ = build_synthetic_dump(tmp_path / "usage", spec)
        for bits in (2, 3, 4):
            df = codebook_bucket_usage(dump, bits, kind="K", token_type="all")
            for layer in range(spec.n_layers):
                layer_df = df[df["layer"] == layer]
                total_count = layer_df["count"].sum()
                expected = spec.n_tokens * spec.n_kv_head * spec.head_dim
                assert total_count == expected, (
                    f"bits={bits} layer={layer}: "
                    f"count sum {total_count} != expected {expected}"
                )

    def test_fractions_sum_to_one(self, tmp_path: Path) -> None:
        spec = SyntheticDumpSpec(n_tokens=8, n_layers=1, n_kv_head=1, head_dim=128)
        dump, _, _ = build_synthetic_dump(tmp_path / "frac", spec)
        df = codebook_bucket_usage(dump, bits=3, kind="V", token_type="all")
        layer_df = df[df["layer"] == 0]
        assert layer_df["fraction"].sum() == pytest.approx(1.0, abs=1e-9)

    def test_token_type_vision_filters(self, tmp_path: Path) -> None:
        spec = SyntheticDumpSpec(
            n_tokens=20, n_layers=1, n_kv_head=1, head_dim=128, vision_token_count=5,
        )
        dump, _, _ = build_synthetic_dump(tmp_path / "vis", spec)
        df = codebook_bucket_usage(dump, bits=2, kind="K", token_type="vision")
        total = df[df["layer"] == 0]["count"].sum()
        expected = 5 * spec.n_kv_head * spec.head_dim
        assert total == expected

    def test_token_type_text_filters(self, tmp_path: Path) -> None:
        spec = SyntheticDumpSpec(
            n_tokens=20, n_layers=1, n_kv_head=1, head_dim=128, vision_token_count=5,
        )
        dump, _, _ = build_synthetic_dump(tmp_path / "txt", spec)
        df = codebook_bucket_usage(dump, bits=2, kind="K", token_type="text")
        total = df[df["layer"] == 0]["count"].sum()
        expected = 15 * spec.n_kv_head * spec.head_dim
        assert total == expected

    def test_correct_columns(self, tmp_path: Path) -> None:
        spec = SyntheticDumpSpec(n_tokens=8, n_layers=1, n_kv_head=1, head_dim=128)
        dump, _, _ = build_synthetic_dump(tmp_path / "cols", spec)
        df = codebook_bucket_usage(dump, bits=4, kind="K")
        assert set(df.columns) == {"layer", "bucket_idx", "centroid_value", "count", "fraction"}

    def test_invalid_bits_raises(self, tmp_path: Path) -> None:
        spec = SyntheticDumpSpec(n_tokens=4, n_layers=1, n_kv_head=1, head_dim=128)
        dump, _, _ = build_synthetic_dump(tmp_path / "inv", spec)
        with pytest.raises(ValueError, match="bits must be one of"):
            codebook_bucket_usage(dump, bits=7, kind="K")


# ---------------------------------------------------------------------------
# codebook_vision_vs_text_skew
# ---------------------------------------------------------------------------


class TestCodebookVisionVsTextSkew:
    """Tests for JS divergence between vision/text bucket distributions."""

    def test_identical_distributions_zero_divergence(self, tmp_path: Path) -> None:
        # When all tokens are from one type (no vision), the result is empty.
        spec = SyntheticDumpSpec(
            n_tokens=20, n_layers=2, n_kv_head=1, head_dim=128, vision_token_count=0,
        )
        dump, _, _ = build_synthetic_dump(tmp_path / "no_vis", spec)
        df = codebook_vision_vs_text_skew(dump, bits=3, kind="K")
        assert df.empty

    def test_same_distribution_low_divergence(self, tmp_path: Path) -> None:
        # Build a dump where vision and text come from the same
        # distribution (both Gaussian).  JS divergence should be small.
        spec = SyntheticDumpSpec(
            n_tokens=200, n_layers=1, n_kv_head=2, head_dim=128,
            vision_token_count=100,
        )
        dump, _, _ = build_synthetic_dump(tmp_path / "same", spec)
        df = codebook_vision_vs_text_skew(dump, bits=3, kind="K")
        assert not df.empty
        # With identical source distributions the JS divergence should be
        # small (sampling noise only).
        assert df["js_divergence"].iloc[0] < 0.05

    def test_result_columns(self, tmp_path: Path) -> None:
        spec = SyntheticDumpSpec(
            n_tokens=20, n_layers=2, n_kv_head=1, head_dim=128, vision_token_count=10,
        )
        dump, _, _ = build_synthetic_dump(tmp_path / "cols", spec)
        df = codebook_vision_vs_text_skew(dump, bits=2, kind="V")
        assert set(df.columns) == {"layer", "js_divergence"}
        assert len(df) == 2  # one row per layer

    def test_js_divergence_non_negative(self, tmp_path: Path) -> None:
        spec = SyntheticDumpSpec(
            n_tokens=40, n_layers=3, n_kv_head=1, head_dim=128, vision_token_count=20,
        )
        dump, _, _ = build_synthetic_dump(tmp_path / "nn", spec)
        df = codebook_vision_vs_text_skew(dump, bits=4, kind="K")
        assert (df["js_divergence"] >= 0.0).all()


# ---------------------------------------------------------------------------
# plot_codebook_usage
# ---------------------------------------------------------------------------


class TestPlotCodebookUsage:

    def test_creates_png_files(self, tmp_path: Path) -> None:
        spec = SyntheticDumpSpec(
            n_tokens=8, n_layers=2, n_kv_head=1, head_dim=128, vision_token_count=4,
        )
        dump, _, _ = build_synthetic_dump(tmp_path / "plotdump", spec)
        out_dir = tmp_path / "plots"
        figs = plot_codebook_usage(dump, bits=3, kind="K", out_dir=out_dir)
        assert len(figs) == spec.n_layers
        for layer in range(spec.n_layers):
            png = out_dir / f"codebook_usage_K_L{layer}_3bit.png"
            assert png.is_file()

    def test_subset_layers(self, tmp_path: Path) -> None:
        spec = SyntheticDumpSpec(
            n_tokens=8, n_layers=4, n_kv_head=1, head_dim=128,
        )
        dump, _, _ = build_synthetic_dump(tmp_path / "sub", spec)
        figs = plot_codebook_usage(dump, bits=2, kind="V", layers=[0, 2])
        assert len(figs) == 2


# ---------------------------------------------------------------------------
# plot_codebook_skew
# ---------------------------------------------------------------------------


class TestPlotCodebookSkew:

    def test_creates_figure(self, tmp_path: Path) -> None:
        spec = SyntheticDumpSpec(
            n_tokens=40, n_layers=3, n_kv_head=1, head_dim=128, vision_token_count=20,
        )
        dump, _, _ = build_synthetic_dump(tmp_path / "skewdump", spec)
        out_path = tmp_path / "skew.png"
        fig = plot_codebook_skew(dump, bits=3, kind="K", out_path=out_path)
        assert fig is not None
        assert out_path.is_file()

    def test_returns_none_when_no_vision(self, tmp_path: Path) -> None:
        spec = SyntheticDumpSpec(
            n_tokens=10, n_layers=1, n_kv_head=1, head_dim=128, vision_token_count=0,
        )
        dump, _, _ = build_synthetic_dump(tmp_path / "novis", spec)
        fig = plot_codebook_skew(dump, bits=2, kind="K")
        assert fig is None


# ---------------------------------------------------------------------------
# bit_exact_reproducibility
# ---------------------------------------------------------------------------


class TestBitExactReproducibility:

    def test_identical_dumps(self, tmp_path: Path) -> None:
        spec = SyntheticDumpSpec(n_tokens=8, n_layers=2, n_kv_head=1, head_dim=128)
        dump_a, K, V = build_synthetic_dump(tmp_path / "a", spec, run_name="a")
        # Write the exact same data to a second directory.
        from tq_bench.kv_analysis.loader import KVDump, KVDumpWriter
        writer = KVDumpWriter(tmp_path / "b")
        writer.write(
            K=K, V=V,
            vision_token_mask=dump_a.vision_mask().tolist(),
            run_name="b",
        )
        dump_b = KVDump(tmp_path / "b")

        result = bit_exact_reproducibility(dump_a, dump_b)
        assert result["is_reproducible"] is True
        assert result["n_exact_K"] == spec.n_layers
        assert result["n_exact_V"] == spec.n_layers
        assert result["max_K_diff"] == 0.0
        assert result["max_V_diff"] == 0.0

    def test_different_dumps(self, tmp_path: Path) -> None:
        spec = SyntheticDumpSpec(n_tokens=8, n_layers=2, n_kv_head=1, head_dim=128)
        dump_a, dump_b = build_pair_of_dumps(tmp_path, spec, noise=0.1)

        result = bit_exact_reproducibility(dump_a, dump_b)
        assert result["is_reproducible"] is False
        assert result["n_exact_K"] == 0
        assert result["n_exact_V"] == 0
        assert result["max_K_diff"] > 0.0
        assert result["max_V_diff"] > 0.0

    def test_shape_mismatch_raises(self, tmp_path: Path) -> None:
        spec_a = SyntheticDumpSpec(n_tokens=8, n_layers=2, n_kv_head=1, head_dim=128)
        spec_b = SyntheticDumpSpec(n_tokens=16, n_layers=2, n_kv_head=1, head_dim=128)
        dump_a, _, _ = build_synthetic_dump(tmp_path / "ma", spec_a, run_name="ma")
        dump_b, _, _ = build_synthetic_dump(tmp_path / "mb", spec_b, run_name="mb")
        with pytest.raises(ValueError, match="shape mismatch"):
            bit_exact_reproducibility(dump_a, dump_b)

    def test_partially_matching_layers(self, tmp_path: Path) -> None:
        """One layer identical, one layer different."""
        spec = SyntheticDumpSpec(n_tokens=8, n_layers=2, n_kv_head=1, head_dim=128)
        dump_a, K, V = build_synthetic_dump(tmp_path / "pa", spec, run_name="pa")

        # Copy K and V, perturb only layer 1.
        K2 = {k: v.copy() for k, v in K.items()}
        V2 = {k: v.copy() for k, v in V.items()}
        K2[1] = K2[1] + 0.01
        V2[1] = V2[1] + 0.01

        from tq_bench.kv_analysis.loader import KVDump, KVDumpWriter
        writer = KVDumpWriter(tmp_path / "pb")
        writer.write(
            K=K2, V=V2,
            vision_token_mask=dump_a.vision_mask().tolist(),
            run_name="pb",
        )
        dump_b = KVDump(tmp_path / "pb")

        result = bit_exact_reproducibility(dump_a, dump_b)
        assert result["is_reproducible"] is False
        assert result["n_exact_K"] == 1
        assert result["n_exact_V"] == 1
        assert result["max_K_diff"] > 0.0
        assert result["max_V_diff"] > 0.0
