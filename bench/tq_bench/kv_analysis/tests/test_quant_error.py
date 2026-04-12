"""Tests for :mod:`tq_bench.kv_analysis.quant_error`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tq_bench.kv_analysis.quant_error import (
    THEORETICAL_MSE_PER_COORD,
    compare_dumps,
    compare_tensors,
    compare_with_theoretical,
    summarize_against_theoretical,
)
from tq_bench.kv_analysis.tests.conftest import (
    SyntheticDumpSpec,
    build_pair_of_dumps,
)


def test_compare_tensors_identity() -> None:
    rng = np.random.default_rng(0)
    tensor = rng.standard_normal((8, 2, 16)).astype(np.float32)
    metrics = compare_tensors(tensor, tensor)
    assert metrics["per_coord_mse"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["max_abs_error"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["cosine_sim"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["inner_product_bias"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["relative_error"] == pytest.approx(0.0, abs=1e-8)


def test_compare_tensors_known_noise() -> None:
    rng = np.random.default_rng(1)
    tensor = rng.standard_normal((200, 1, 32)).astype(np.float32)
    noise_level = 0.1
    noisy = tensor + rng.standard_normal(tensor.shape).astype(np.float32) * noise_level
    metrics = compare_tensors(tensor, noisy)
    # For N(0, noise^2) noise the per-coord MSE should be ~noise^2.
    assert metrics["per_coord_mse"] == pytest.approx(noise_level ** 2, rel=0.2)
    assert metrics["cosine_sim"] > 0.9
    # Relative error is also ~ noise.
    assert metrics["relative_error"] == pytest.approx(noise_level, rel=0.2)


def test_compare_tensors_shape_mismatch() -> None:
    a = np.zeros((2, 1, 4))
    b = np.zeros((2, 1, 8))
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_tensors(a, b)


def test_compare_with_theoretical_known_bits() -> None:
    res = compare_with_theoretical(0.034, bits=3)
    assert res["bits"] == 3
    assert res["theoretical_mse_per_coord"] == pytest.approx(
        THEORETICAL_MSE_PER_COORD[3]
    )
    assert res["ratio"] == pytest.approx(1.0, rel=0.05)
    assert res["within_tolerance"]


def test_compare_with_theoretical_unknown_bits() -> None:
    with pytest.raises(ValueError, match="no theoretical MSE"):
        compare_with_theoretical(0.1, bits=7)


def test_compare_dumps_produces_rows(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(n_tokens=12, n_layers=2, n_kv_head=2, head_dim=16)
    baseline, quant = build_pair_of_dumps(tmp_path, spec, noise=0.05)
    df = compare_dumps(baseline, quant)
    assert not df.empty
    # 2 layers * ( 3 token_types ) * 2 kinds = 12, assuming vision+text present.
    expected_n_rows = spec.n_layers * 3 * 2
    assert len(df) == expected_n_rows
    # Basic metric sanity.
    assert df["per_coord_mse"].min() >= 0.0
    assert df["cosine_sim"].max() <= 1.0 + 1e-9


def test_summarize_against_theoretical_single_bits(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(n_tokens=10, n_layers=1, n_kv_head=1, head_dim=16)
    baseline, quant = build_pair_of_dumps(tmp_path, spec, noise=0.02)
    df = compare_dumps(baseline, quant)
    summary = summarize_against_theoretical(df, bits_by_kind=3)
    assert not summary.empty
    assert set(summary["kind"].unique()) == {"K", "V"}
    assert "ratio" in summary.columns
