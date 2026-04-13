"""Tests for :mod:`tq_bench.kv_analysis.attention_analysis`."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tq_bench.kv_analysis.attention_analysis import (
    attention_entropy,
    compare_attention_from_dumps,
    compute_attention_weights_from_kv,
    js_divergence_attention,
    kl_divergence_attention,
    top1_position_match_rate,
    topk_overlap_rate,
)
from tq_bench.kv_analysis.loader import KVDumpWriter, load_dump


def test_compute_attention_weights_shape_and_simplex() -> None:
    rng = np.random.default_rng(0)
    K = rng.standard_normal((8, 2, 16)).astype(np.float32)
    Q = rng.standard_normal((1, 2, 16)).astype(np.float32)
    attn = compute_attention_weights_from_kv(K, Q)
    assert attn.shape == (2, 1, 8)  # (heads, queries, keys)
    # Each row should sum to 1 and be non-negative.
    sums = attn.sum(axis=-1)
    np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-10)
    assert (attn >= 0).all()


def test_kl_divergence_identity_is_zero() -> None:
    rng = np.random.default_rng(1)
    attn = rng.dirichlet(np.ones(10), size=(3, 1))
    assert kl_divergence_attention(attn, attn) == pytest.approx(0.0, abs=1e-12)


def test_kl_divergence_asymmetry() -> None:
    p = np.array([[0.9, 0.05, 0.05]])
    q = np.array([[0.1, 0.45, 0.45]])
    kl_pq = kl_divergence_attention(p, q)
    kl_qp = kl_divergence_attention(q, p)
    assert kl_pq > 0
    assert kl_qp > 0
    assert kl_pq != kl_qp  # asymmetric


def test_js_divergence_symmetric() -> None:
    p = np.array([[0.9, 0.05, 0.05]])
    q = np.array([[0.1, 0.45, 0.45]])
    js_pq = js_divergence_attention(p, q)
    js_qp = js_divergence_attention(q, p)
    assert js_pq == pytest.approx(js_qp, rel=1e-6)


def test_top1_position_match_rate_perfect() -> None:
    p = np.eye(5)
    q = np.eye(5)
    assert top1_position_match_rate(p, q) == pytest.approx(1.0)


def test_top1_position_match_rate_half() -> None:
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    q = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert top1_position_match_rate(p, q) == pytest.approx(0.5)


def test_topk_overlap_rate_full_match() -> None:
    rng = np.random.default_rng(2)
    attn = rng.dirichlet(np.ones(12), size=(4,))
    assert topk_overlap_rate(attn, attn, k=3) == pytest.approx(1.0)


def test_attention_entropy_bounds() -> None:
    # Uniform distribution has maximum entropy ln(n).
    n = 8
    uniform = np.full((1, 1, n), 1.0 / n)
    e = attention_entropy(uniform)
    assert e == pytest.approx(np.log(n), rel=1e-6)

    delta = np.zeros((1, 1, n))
    delta[0, 0, 0] = 1.0
    assert attention_entropy(delta) == pytest.approx(0.0, abs=1e-9)


def test_compute_attention_invalid_shape() -> None:
    K = np.zeros((8, 2, 16))
    Q = np.zeros((1, 2, 8))  # mismatched head_dim
    with pytest.raises(ValueError, match="head/dim mismatch"):
        compute_attention_weights_from_kv(K, Q)


# ---------------------------------------------------------------------------
# compare_attention_from_dumps tests
# ---------------------------------------------------------------------------


def _make_dump_pair(tmp_path: Path, n_tokens=12, n_layers=2, n_heads=2, head_dim=16, noise=0.1, seed=42):
    """Create a baseline + quantized dump pair with controlled noise."""
    rng = np.random.default_rng(seed)
    K_base = {i: rng.standard_normal((n_tokens, n_heads, head_dim)).astype(np.float32) for i in range(n_layers)}
    V_base = {i: rng.standard_normal((n_tokens, n_heads, head_dim)).astype(np.float32) for i in range(n_layers)}

    K_quant = {i: K_base[i] + rng.normal(0, noise, K_base[i].shape).astype(np.float32) for i in range(n_layers)}
    V_quant = {i: V_base[i] + rng.normal(0, noise, V_base[i].shape).astype(np.float32) for i in range(n_layers)}

    vision_mask = [True] * 4 + [False] * (n_tokens - 4)

    base_dir = tmp_path / "baseline"
    quant_dir = tmp_path / "quantized"

    KVDumpWriter(base_dir).write(K=K_base, V=V_base, vision_token_mask=vision_mask)
    KVDumpWriter(quant_dir).write(K=K_quant, V=V_quant, vision_token_mask=vision_mask)

    return load_dump(base_dir), load_dump(quant_dir)


def test_compare_attention_from_dumps_basic(tmp_path: Path) -> None:
    baseline, quantized = _make_dump_pair(tmp_path)
    df = compare_attention_from_dumps(baseline, quantized)

    assert not df.empty
    # 2 layers × 3 token types = 6 rows
    assert len(df) == 6
    assert set(df["token_type"].unique()) == {"all", "vision", "text"}
    assert set(df["layer"].unique()) == {0, 1}

    # KL should be > 0 (noise was added)
    assert (df["kl_divergence"] >= 0).all()
    assert (df["js_divergence"] >= 0).all()
    assert (df["top1_match_rate"] >= 0).all()
    assert (df["top1_match_rate"] <= 1).all()


def test_compare_attention_identical_dumps(tmp_path: Path) -> None:
    baseline, _ = _make_dump_pair(tmp_path, noise=0.0)
    df = compare_attention_from_dumps(baseline, baseline)

    # Identical K → zero divergence, perfect match
    for _, row in df.iterrows():
        assert row["kl_divergence"] == pytest.approx(0.0, abs=1e-10)
        assert row["js_divergence"] == pytest.approx(0.0, abs=1e-10)
        assert row["top1_match_rate"] == pytest.approx(1.0)
        assert row["topk_overlap_k5"] == pytest.approx(1.0)
        assert row["entropy_delta"] == pytest.approx(0.0, abs=1e-10)


def test_compare_attention_high_noise(tmp_path: Path) -> None:
    baseline, quantized = _make_dump_pair(tmp_path, noise=5.0)
    df = compare_attention_from_dumps(baseline, quantized)

    all_rows = df[df["token_type"] == "all"]
    # With high noise, top-1 match should drop significantly
    mean_top1 = all_rows["top1_match_rate"].mean()
    assert mean_top1 < 0.9  # unlikely to be perfect with noise=5.0

    # KL should be substantial
    assert all_rows["kl_divergence"].mean() > 0.01


def test_compare_attention_token_count_mismatch(tmp_path: Path) -> None:
    rng = np.random.default_rng(99)
    K1 = {0: rng.standard_normal((10, 2, 8)).astype(np.float32)}
    V1 = {0: rng.standard_normal((10, 2, 8)).astype(np.float32)}
    K2 = {0: rng.standard_normal((12, 2, 8)).astype(np.float32)}
    V2 = {0: rng.standard_normal((12, 2, 8)).astype(np.float32)}

    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    KVDumpWriter(d1).write(K=K1, V=V1)
    KVDumpWriter(d2).write(K=K2, V=V2)

    with pytest.raises(ValueError, match="Token count mismatch"):
        compare_attention_from_dumps(load_dump(d1), load_dump(d2))


def test_compare_attention_max_query_tokens(tmp_path: Path) -> None:
    baseline, quantized = _make_dump_pair(tmp_path, n_tokens=100)
    df = compare_attention_from_dumps(baseline, quantized, max_query_tokens=8)

    # Should cap query tokens at 8
    assert (df["n_query_tokens"] <= 8).all()
