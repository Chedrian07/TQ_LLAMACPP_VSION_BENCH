"""Tests for :mod:`tq_bench.kv_analysis.attention_analysis`."""

from __future__ import annotations

import numpy as np
import pytest

from tq_bench.kv_analysis.attention_analysis import (
    attention_entropy,
    compute_attention_weights_from_kv,
    js_divergence_attention,
    kl_divergence_attention,
    top1_position_match_rate,
    topk_overlap_rate,
)


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
