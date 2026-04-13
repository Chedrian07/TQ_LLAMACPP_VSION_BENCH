"""Attention weight analysis helpers.

This module provides two levels of functionality:

1. **Primitives** — ``compute_attention_weights_from_kv``,
   ``kl_divergence_attention``, etc.  These operate on raw numpy
   arrays and are the building blocks.

2. **Dump-level comparison** — ``compare_attention_from_dumps`` takes
   a baseline and a quantized :class:`KVDump` and produces a
   per-layer DataFrame with KL divergence, top-1 match rate,
   attention entropy delta, etc., optionally split by vision/text
   token type.

**Q-probe strategy**: The C++ kv-dump tool extracts K and V but not Q.
During inference with quantized KV cache, Q is computed fresh (not
quantized) and is therefore **identical** between the baseline and the
quantized run for the same input.  This means any fixed Q probe yields
a valid comparison of attention distributions.  We use each row of the
**baseline K** as a Q probe (simulating the scenario where each token
position attends to all other positions).  The resulting attention
distribution differences are entirely due to K/V quantization.

Shapes
------

* ``K``: ``(n_tokens, n_kv_head, head_dim)``
* ``Q``: ``(n_q_tokens, n_kv_head, head_dim)`` — typically
  ``n_q_tokens == 1`` for single-query decoding.

Output of :func:`compute_attention_weights_from_kv` is
``(n_kv_head, n_q_tokens, n_tokens)`` — one softmax distribution per
head per query position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .loader import KVDump


# ---------------------------------------------------------------------------
# Softmax attention
# ---------------------------------------------------------------------------


def _as_3d(tensor: np.ndarray, name: str) -> np.ndarray:
    if tensor.ndim == 2:
        n_tokens, f = tensor.shape
        # Best-effort: assume a single head.
        return tensor.reshape(n_tokens, 1, f)
    if tensor.ndim == 3:
        return tensor
    raise ValueError(f"{name} must be 2-D or 3-D, got shape {tensor.shape}")


def compute_attention_weights_from_kv(
    K: np.ndarray,
    Q: np.ndarray,
    *,
    scale: float | None = None,
) -> np.ndarray:
    """Return attention weights ``softmax(Q K^T / sqrt(d))``.

    Parameters
    ----------
    K:
        Key tensor ``(n_tokens, n_kv_head, head_dim)``.
    Q:
        Query tensor ``(n_q_tokens, n_kv_head, head_dim)``.
    scale:
        Optional scale override.  Defaults to ``1 / sqrt(head_dim)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_kv_head, n_q_tokens, n_tokens)`` of float64
        probabilities along the last axis.
    """
    K3 = _as_3d(K, "K")
    Q3 = _as_3d(Q, "Q")
    if K3.shape[1] != Q3.shape[1] or K3.shape[2] != Q3.shape[2]:
        raise ValueError(
            f"head/dim mismatch between K {K3.shape} and Q {Q3.shape}"
        )

    n_tokens, n_heads, head_dim = K3.shape
    n_q_tokens = Q3.shape[0]
    scale_factor = scale if scale is not None else 1.0 / float(np.sqrt(head_dim))

    # Use float64 for numerical stability.
    K64 = K3.astype(np.float64, copy=False)
    Q64 = Q3.astype(np.float64, copy=False)

    # (h, q, d) x (h, d, t) -> (h, q, t)
    K_perm = np.transpose(K64, (1, 2, 0))  # (h, d, t)
    Q_perm = np.transpose(Q64, (1, 0, 2))  # (h, q, d)
    logits = np.matmul(Q_perm, K_perm) * scale_factor  # (h, q, t)

    # Stable softmax along the token axis.
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Divergences
# ---------------------------------------------------------------------------


def kl_divergence_attention(
    baseline_attn: np.ndarray,
    quant_attn: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Mean KL(baseline || quant) over the leading axes.

    The last axis is assumed to be the probability simplex.  Arrays
    must share shape.  A small ``eps`` is added to ``quant_attn`` to
    avoid ``log(0)``.
    """
    if baseline_attn.shape != quant_attn.shape:
        raise ValueError(
            f"shape mismatch: {baseline_attn.shape} vs {quant_attn.shape}"
        )
    p = np.asarray(baseline_attn, dtype=np.float64)
    q = np.asarray(quant_attn, dtype=np.float64)
    q = np.clip(q, eps, 1.0)
    mask = p > 0
    log_ratio = np.zeros_like(p)
    log_ratio[mask] = np.log(p[mask] / q[mask])
    kl = (p * log_ratio).sum(axis=-1)
    return float(kl.mean())


def js_divergence_attention(
    baseline_attn: np.ndarray,
    quant_attn: np.ndarray,
) -> float:
    """Mean Jensen-Shannon divergence across the leading axes."""
    if baseline_attn.shape != quant_attn.shape:
        raise ValueError(
            f"shape mismatch: {baseline_attn.shape} vs {quant_attn.shape}"
        )
    p = np.asarray(baseline_attn, dtype=np.float64)
    q = np.asarray(quant_attn, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence_attention(p, m) + kl_divergence_attention(q, m))


def top1_position_match_rate(
    baseline_attn: np.ndarray,
    quant_attn: np.ndarray,
) -> float:
    """Fraction of (head, query) pairs whose arg-max token matches."""
    if baseline_attn.shape != quant_attn.shape:
        raise ValueError(
            f"shape mismatch: {baseline_attn.shape} vs {quant_attn.shape}"
        )
    top1_base = np.argmax(baseline_attn, axis=-1)
    top1_quant = np.argmax(quant_attn, axis=-1)
    return float(np.mean(top1_base == top1_quant))


def topk_overlap_rate(
    baseline_attn: np.ndarray,
    quant_attn: np.ndarray,
    *,
    k: int = 5,
) -> float:
    """Average Jaccard overlap of top-k tokens across heads/queries."""
    if baseline_attn.shape != quant_attn.shape:
        raise ValueError(
            f"shape mismatch: {baseline_attn.shape} vs {quant_attn.shape}"
        )
    if k <= 0:
        raise ValueError("k must be positive")
    n_last = baseline_attn.shape[-1]
    k_eff = min(k, n_last)

    base_flat = baseline_attn.reshape(-1, n_last)
    quant_flat = quant_attn.reshape(-1, n_last)

    overlaps = []
    for i in range(base_flat.shape[0]):
        b_top = set(np.argpartition(base_flat[i], -k_eff)[-k_eff:])
        q_top = set(np.argpartition(quant_flat[i], -k_eff)[-k_eff:])
        union = b_top | q_top
        overlaps.append(len(b_top & q_top) / len(union))
    return float(np.mean(overlaps))


def attention_entropy(attn: np.ndarray) -> float:
    """Mean Shannon entropy (nats) of the attention distribution.

    Uses the natural logarithm.  Shape ``(..., n_tokens)``.
    """
    a = np.asarray(attn, dtype=np.float64)
    safe = np.where(a > 0, a, 1.0)
    ent = -(a * np.log(safe)).sum(axis=-1)
    return float(ent.mean())


# ---------------------------------------------------------------------------
# Per-layer attention comparison (single layer, single token type)
# ---------------------------------------------------------------------------


@dataclass
class LayerAttentionMetrics:
    """Attention comparison metrics for one layer."""

    layer: int = 0
    token_type: str = "all"  # "all" | "vision" | "text"
    n_query_tokens: int = 0

    kl_divergence: float = 0.0
    js_divergence: float = 0.0
    top1_match_rate: float = 0.0
    topk_overlap_k5: float = 0.0
    entropy_baseline: float = 0.0
    entropy_quantized: float = 0.0
    entropy_delta: float = 0.0  # quantized - baseline (positive = more uniform)


def _compare_attention_layer(
    K_baseline: np.ndarray,
    K_quantized: np.ndarray,
    Q_probes: np.ndarray,
    *,
    layer: int,
    token_type: str,
) -> LayerAttentionMetrics:
    """Compare attention distributions for a single layer.

    Parameters
    ----------
    K_baseline, K_quantized:
        Key tensors ``(n_tokens, n_kv_head, head_dim)``.
    Q_probes:
        Query probes ``(n_q, n_kv_head, head_dim)`` — a subset of
        baseline K rows filtered to the requested token type.
    """
    if Q_probes.shape[0] == 0:
        return LayerAttentionMetrics(layer=layer, token_type=token_type)

    attn_base = compute_attention_weights_from_kv(K_baseline, Q_probes)
    attn_quant = compute_attention_weights_from_kv(K_quantized, Q_probes)

    return LayerAttentionMetrics(
        layer=layer,
        token_type=token_type,
        n_query_tokens=Q_probes.shape[0],
        kl_divergence=kl_divergence_attention(attn_base, attn_quant),
        js_divergence=js_divergence_attention(attn_base, attn_quant),
        top1_match_rate=top1_position_match_rate(attn_base, attn_quant),
        topk_overlap_k5=topk_overlap_rate(attn_base, attn_quant, k=5),
        entropy_baseline=attention_entropy(attn_base),
        entropy_quantized=attention_entropy(attn_quant),
        entropy_delta=attention_entropy(attn_quant) - attention_entropy(attn_base),
    )


# ---------------------------------------------------------------------------
# Dump-level comparison
# ---------------------------------------------------------------------------


def compare_attention_from_dumps(
    baseline: "KVDump",
    quantized: "KVDump",
    *,
    max_query_tokens: int = 64,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare attention distributions between baseline and quantized dumps.

    Uses baseline K rows as Q probes (see module docstring for
    rationale).  To keep memory and compute bounded, at most
    ``max_query_tokens`` probes are sampled per token type.

    Parameters
    ----------
    baseline, quantized:
        KVDump instances from the same prompt (same n_tokens).
    max_query_tokens:
        Cap on Q probes per token type per layer.
    seed:
        RNG seed for probe sampling.

    Returns
    -------
    pd.DataFrame
        One row per (layer, token_type) with columns:
        layer, token_type, n_query_tokens, kl_divergence,
        js_divergence, top1_match_rate, topk_overlap_k5,
        entropy_baseline, entropy_quantized, entropy_delta.
    """
    if baseline.n_tokens != quantized.n_tokens:
        raise ValueError(
            f"Token count mismatch: baseline={baseline.n_tokens} "
            f"vs quantized={quantized.n_tokens}"
        )

    rng = np.random.default_rng(seed)

    vision_idx = baseline.token_indices("vision")
    text_idx = baseline.token_indices("text")
    all_idx = baseline.token_indices("all")

    def _sample_indices(indices: np.ndarray) -> np.ndarray:
        if len(indices) <= max_query_tokens:
            return indices
        return rng.choice(indices, size=max_query_tokens, replace=False)

    rows: list[dict] = []

    for layer in range(baseline.n_layers):
        K_base = baseline.get_K(layer)
        K_quant = quantized.get_K(layer)

        for token_type, idx in [("all", all_idx), ("vision", vision_idx), ("text", text_idx)]:
            if len(idx) == 0:
                continue
            sampled = _sample_indices(idx)
            Q_probes = K_base[sampled]  # (n_q, n_kv_head, head_dim)

            metrics = _compare_attention_layer(
                K_base, K_quant, Q_probes,
                layer=layer, token_type=token_type,
            )
            rows.append({
                "layer": metrics.layer,
                "token_type": metrics.token_type,
                "n_query_tokens": metrics.n_query_tokens,
                "kl_divergence": metrics.kl_divergence,
                "js_divergence": metrics.js_divergence,
                "top1_match_rate": metrics.top1_match_rate,
                "topk_overlap_k5": metrics.topk_overlap_k5,
                "entropy_baseline": metrics.entropy_baseline,
                "entropy_quantized": metrics.entropy_quantized,
                "entropy_delta": metrics.entropy_delta,
            })

    return pd.DataFrame(rows)
