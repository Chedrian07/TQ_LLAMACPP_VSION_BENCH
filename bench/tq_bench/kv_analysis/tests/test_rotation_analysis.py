"""Tests for :mod:`tq_bench.kv_analysis.rotation_analysis`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tq_bench.kv_analysis.rotation_analysis import (
    TURBO_DIM,
    TURBO_SEED,
    analyze_rotation_per_layer,
    apply_fwht,
    beta_distribution_fit_test,
    coordinate_independence_test,
    fwht_round_trip,
    sign_flip_mask,
    vision_vs_text_rotation_analysis,
)
from tq_bench.kv_analysis.tests.conftest import (
    SyntheticDumpSpec,
    build_synthetic_dump,
)


# ---------------------------------------------------------------------------
# Sign flip reference
# ---------------------------------------------------------------------------


def _python_sign_flip(n: int, seed: int = TURBO_SEED) -> np.ndarray:
    """Mirror of the C++ reference implementation (pure Python)."""
    out = np.empty(n, dtype=np.int8)
    for i in range(n):
        h = ((seed * 2654435761) + (i * 2246822519)) & 0xFFFFFFFF
        flip = 1 if (h >> 31) else 0
        out[i] = -1 if flip else 1
    return out


def test_sign_flip_matches_reference() -> None:
    np_mask = sign_flip_mask(128, seed=TURBO_SEED)
    ref = _python_sign_flip(128, seed=TURBO_SEED)
    np.testing.assert_array_equal(np_mask, ref)


def test_sign_flip_mask_nonconstant() -> None:
    mask = sign_flip_mask(128)
    assert set(np.unique(mask).tolist()).issubset({-1, 1})
    # Mask should have both +1 and -1 entries.
    assert (mask == 1).any()
    assert (mask == -1).any()


# ---------------------------------------------------------------------------
# FWHT correctness
# ---------------------------------------------------------------------------


def test_fwht_round_trip_single_vector() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(128)
    # fwht_round_trip does forward then inverse; should recover the
    # original with tight tolerance.
    back = fwht_round_trip(x)
    np.testing.assert_allclose(back, x, atol=1e-10, rtol=1e-10)


def test_fwht_round_trip_batched() -> None:
    rng = np.random.default_rng(1)
    batch = rng.standard_normal((10, 128))
    back = fwht_round_trip(batch)
    np.testing.assert_allclose(back, batch, atol=1e-10, rtol=1e-10)


def test_fwht_requires_power_of_two() -> None:
    with pytest.raises(ValueError, match="power of two"):
        apply_fwht(np.zeros(7), normalize=False)


def test_fwht_output_shape_preserved() -> None:
    out_1d = apply_fwht(np.ones(8), normalize=False)
    assert out_1d.shape == (8,)
    out_2d = apply_fwht(np.ones((3, 8)), normalize=False)
    assert out_2d.shape == (3, 8)


def test_fwht_normalizes_input() -> None:
    # An all-ones vector is normalized to uniform, then sign-flipped,
    # then FWHT'd.  After the FWHT the energy should be conserved.
    v = np.ones(128)
    out = apply_fwht(v, normalize=True)
    # After a unit-normalise + FWHT, ||out||^2 = n (energy conservation
    # for the unscaled FWHT).
    assert np.linalg.norm(out) ** 2 == pytest.approx(128.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Beta fit test
# ---------------------------------------------------------------------------


def test_beta_fit_on_rotated_gaussian_passes() -> None:
    # Rotating Gaussian samples projects them onto the sphere, which
    # is exactly the TurboQuant assumption.  The resulting marginal
    # should look Beta((d-1)/2, (d-1)/2).
    rng = np.random.default_rng(3)
    d = 128
    samples = rng.standard_normal((2000, d))
    rotated = apply_fwht(samples, normalize=True)
    res = beta_distribution_fit_test(rotated.ravel(), n_dim=d)
    assert res["fit_quality"] in {"excellent", "good"}
    assert res["ks_statistic"] < 0.10


def test_beta_fit_on_uniform_fails() -> None:
    # Uniform [-1, 1] samples are NOT Beta((d-1)/2, (d-1)/2); the KS
    # test should return a fail/poor label.
    rng = np.random.default_rng(2)
    samples = rng.uniform(-1.0, 1.0, size=4000)
    res = beta_distribution_fit_test(samples, n_dim=128, scale=1.0)
    assert res["fit_quality"] in {"fail", "poor"}
    assert res["ks_statistic"] > 0.05


def test_beta_fit_insufficient_samples() -> None:
    res = beta_distribution_fit_test(np.array([0.1, 0.2]), n_dim=128)
    assert res["fit_quality"] == "insufficient"
    assert np.isnan(res["ks_statistic"])


# ---------------------------------------------------------------------------
# Coordinate independence
# ---------------------------------------------------------------------------


def test_coordinate_independence_iid_gaussian() -> None:
    # iid Gaussian columns have approximately zero pairwise correlation.
    rng = np.random.default_rng(4)
    samples = rng.standard_normal((5000, 64))
    res = coordinate_independence_test(samples)
    assert res["mean_abs_correlation"] < 0.05
    assert res["max_correlation"] < 0.25


def test_coordinate_independence_perfectly_correlated() -> None:
    rng = np.random.default_rng(5)
    base = rng.standard_normal(500)
    stacked = np.stack([base, base, base], axis=1)
    res = coordinate_independence_test(stacked)
    assert res["mean_abs_correlation"] == pytest.approx(1.0, rel=1e-6)
    assert res["max_correlation"] == pytest.approx(1.0, rel=1e-6)


def test_coordinate_independence_rejects_1d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        coordinate_independence_test(np.zeros(10))


# ---------------------------------------------------------------------------
# Per-layer analysis
# ---------------------------------------------------------------------------


def test_analyze_rotation_per_layer_shape(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=40,
        n_layers=2,
        n_kv_head=2,
        head_dim=128,
        vision_token_count=20,
    )
    dump, _, _ = build_synthetic_dump(tmp_path / "rot", spec)
    df = analyze_rotation_per_layer(dump, token_type="all", kind="both")
    assert len(df) == spec.n_layers * 2  # both K and V
    assert set(df["kind"].unique()) == {"K", "V"}
    assert set(df.columns) >= {
        "layer",
        "token_type",
        "kind",
        "ks_statistic",
        "mean_abs_correlation",
    }


def test_vision_vs_text_rotation_analysis(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=32,
        n_layers=2,
        n_kv_head=2,
        head_dim=128,
        vision_token_count=16,
    )
    dump, _, _ = build_synthetic_dump(tmp_path / "vvr", spec)
    df = vision_vs_text_rotation_analysis(dump, kind="K")
    assert set(df["token_type"].unique()) >= {"all", "vision", "text"}
    # Should be (n_layers * 3 token_types * 1 kind) rows.
    assert len(df) == spec.n_layers * 3
