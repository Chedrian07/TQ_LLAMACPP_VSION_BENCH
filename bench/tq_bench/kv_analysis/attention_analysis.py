"""Attention weight analysis helpers.

The full attention-level analysis is predicated on having matching Q
tensors alongside the dumped K tensors, which the C++ tool does not
yet emit.  This module provides the primitives so that the notebook
can already wire up the comparisons using synthetic Q probes, and so
that any future dump containing Q can just call them directly.

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

import numpy as np


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
