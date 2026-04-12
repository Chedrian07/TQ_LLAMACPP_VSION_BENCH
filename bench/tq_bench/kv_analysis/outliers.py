"""Outlier channel detection.

Community reports on TurboQuant call out that a small fraction (5-20%)
of K channels carry 10-100x the magnitude of the median channel, and
that this is what drives quantization failure at low bitwidths.  This
module quantifies that directly from KV dumps.

A "channel" here is a single ``(head_index, dim_index)`` column across
all tokens — i.e. the norm of the length-``n_tokens`` vector obtained
by fixing ``head_index`` and ``dim_index`` and letting the token axis
vary.  A channel is flagged as an *outlier* when its norm exceeds
``threshold * median(all channel norms)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import KVDump


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------


def _channel_norms(tensor: np.ndarray) -> np.ndarray:
    """Return the length-``H*D`` L2 norms of the channel columns."""
    if tensor.size == 0:
        return np.zeros((0,), dtype=np.float64)
    flat = tensor.reshape(tensor.shape[0], -1).astype(np.float64, copy=False)
    return np.linalg.norm(flat, axis=0)


def find_outlier_channels(
    K_or_V: np.ndarray,
    threshold: float = 10.0,
) -> np.ndarray:
    """Return a boolean mask of outlier channels.

    Parameters
    ----------
    K_or_V:
        Tensor with shape ``(n_tokens, n_kv_head, head_dim)`` or
        ``(n_tokens, n_features)``.
    threshold:
        Channels whose norm exceeds ``threshold * median(norms)`` are
        flagged.  A ``threshold`` of 10 matches the community reports.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(n_features,)``.  All ``False`` if the
        tensor is empty.
    """
    norms = _channel_norms(K_or_V)
    if norms.size == 0:
        return np.zeros((0,), dtype=bool)
    med = float(np.median(norms))
    if med <= 0.0:
        return np.zeros_like(norms, dtype=bool)
    return norms > threshold * med


def outlier_statistics(
    tensor: np.ndarray,
    *,
    threshold: float = 10.0,
) -> dict[str, float]:
    """Return per-channel norm statistics + outlier counts."""
    norms = _channel_norms(tensor)
    if norms.size == 0:
        return {
            "n_channels": 0,
            "median_channel_norm": float("nan"),
            "max_channel_norm": float("nan"),
            "max_to_median_ratio": float("nan"),
            "n_outliers": 0,
            "outlier_ratio": 0.0,
        }
    med = float(np.median(norms))
    mx = float(norms.max())
    mask = norms > threshold * med if med > 0 else np.zeros_like(norms, dtype=bool)
    return {
        "n_channels": int(norms.size),
        "median_channel_norm": med,
        "max_channel_norm": mx,
        "max_to_median_ratio": (mx / med) if med > 0 else float("inf"),
        "n_outliers": int(mask.sum()),
        "outlier_ratio": float(mask.mean()),
    }


# ---------------------------------------------------------------------------
# Per-layer DataFrame
# ---------------------------------------------------------------------------


def outlier_ratio_per_layer(
    dump: KVDump,
    *,
    threshold: float = 10.0,
    token_type: str = "all",
) -> pd.DataFrame:
    """Per-layer outlier summary for ``dump``.

    Parameters
    ----------
    dump:
        Loaded dump.
    threshold:
        Channel outlier threshold (default 10x median).
    token_type:
        ``'all'`` (default), ``'vision'`` or ``'text'``.
    """
    mask: np.ndarray | None
    if token_type == "all":
        mask = None
    elif token_type == "vision":
        mask = dump.vision_mask()
    elif token_type == "text":
        mask = dump.text_mask()
    else:
        raise ValueError(f"token_type must be all|vision|text, got {token_type!r}")

    if mask is not None and not mask.any():
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for layer in range(dump.n_layers):
        K = dump.get_K(layer)
        V = dump.get_V(layer)
        if mask is not None:
            K = K[mask]
            V = V[mask]
        k_stats = outlier_statistics(K, threshold=threshold)
        v_stats = outlier_statistics(V, threshold=threshold)

        row: dict[str, float | int | str] = {
            "layer": layer,
            "token_type": token_type,
            "threshold": float(threshold),
        }
        for name, value in k_stats.items():
            row[f"k_{name}"] = value
        for name, value in v_stats.items():
            row[f"v_{name}"] = value
        rows.append(row)

    return pd.DataFrame(rows)


def outlier_ratio_vision_vs_text(
    dump: KVDump,
    *,
    threshold: float = 10.0,
) -> pd.DataFrame:
    """Stack ``all`` / ``vision`` / ``text`` outlier tables."""
    frames: list[pd.DataFrame] = []
    for tt in ("all", "vision", "text"):
        if tt == "vision" and not dump.has_vision():
            continue
        if tt == "text" and not dump.has_text():
            continue
        frames.append(outlier_ratio_per_layer(dump, threshold=threshold, token_type=tt))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
