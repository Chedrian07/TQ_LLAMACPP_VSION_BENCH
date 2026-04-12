"""Per-layer K/V distribution statistics.

Three kinds of statistics are exported:

* :func:`compute_value_stats` — elementwise statistics
  (mean / std / min / max / quantiles / abs_max) over a tensor.
* :func:`compute_norm_stats` — per-token L2 norm statistics, where
  a "token" means a row of the reshaped ``(T, H*D)`` tensor.
* :func:`compute_per_layer_stats` — run both of the above across
  all layers and return a tidy :class:`pandas.DataFrame`.

All functions accept 3-D arrays of shape ``(n_tokens, n_kv_head,
head_dim)`` or 2-D arrays of shape ``(n_tokens, n_features)``; the 3-D
case is internally flattened to 2-D.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .loader import KVDump


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------


_QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.5, 0.95, 0.99)
_QLABELS: dict[float, str] = {
    0.01: "q01",
    0.05: "q05",
    0.5: "q50",
    0.95: "q95",
    0.99: "q99",
}


def _flatten_tokens(tensor: np.ndarray) -> np.ndarray:
    """Return a ``(n_tokens, n_features)`` view of ``tensor``.

    Accepts either ``(T, H, D)`` or ``(T, F)``.
    """
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3:
        T, H, D = tensor.shape
        return tensor.reshape(T, H * D)
    raise ValueError(f"expected 2-D or 3-D tensor, got shape {tensor.shape}")


def _empty_value_stats() -> dict[str, float]:
    keys = ["mean", "std", "min", "max", "abs_max"]
    keys += [_QLABELS[q] for q in _QUANTILES]
    return {k: float("nan") for k in keys}


def _empty_norm_stats() -> dict[str, float]:
    keys = ["norm_mean", "norm_std", "norm_min", "norm_max", "norm_median"]
    keys += [f"norm_{_QLABELS[q]}" for q in _QUANTILES]
    return {k: float("nan") for k in keys}


# ---------------------------------------------------------------------------
# Value statistics
# ---------------------------------------------------------------------------


def compute_value_stats(tensor: np.ndarray) -> dict[str, float]:
    """Elementwise statistics of ``tensor`` as a plain dict.

    Returns ``NaN`` for all fields when the tensor has zero elements.
    """
    if tensor.size == 0:
        return _empty_value_stats()

    flat = tensor.astype(np.float64, copy=False).ravel()
    q_values = np.quantile(flat, _QUANTILES)

    stats: dict[str, float] = {
        "mean": float(flat.mean()),
        "std": float(flat.std(ddof=0)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "abs_max": float(np.abs(flat).max()),
    }
    for q, v in zip(_QUANTILES, q_values):
        stats[_QLABELS[q]] = float(v)
    return stats


# ---------------------------------------------------------------------------
# Norm statistics
# ---------------------------------------------------------------------------


def compute_norm_stats(tensor: np.ndarray) -> dict[str, float]:
    """Per-token L2 norm statistics.

    Each token is a row of the reshaped ``(T, F)`` view; the norm is
    the sum of squares across all head/head_dim coordinates.  When the
    input is ``(T, H, D)`` the norms are taken across ``H*D``.
    """
    if tensor.size == 0:
        return _empty_norm_stats()

    flat2d = _flatten_tokens(tensor).astype(np.float64, copy=False)
    norms = np.linalg.norm(flat2d, axis=1)

    q_values = np.quantile(norms, _QUANTILES)
    stats: dict[str, float] = {
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std(ddof=0)),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_median": float(np.median(norms)),
    }
    for q, v in zip(_QUANTILES, q_values):
        stats[f"norm_{_QLABELS[q]}"] = float(v)
    return stats


# ---------------------------------------------------------------------------
# Per-layer aggregation
# ---------------------------------------------------------------------------


def _token_subset(
    tensor: np.ndarray, mask: np.ndarray | None
) -> np.ndarray:
    if mask is None:
        return tensor
    if mask.dtype == bool:
        return tensor[mask]
    return tensor[np.asarray(mask, dtype=np.intp)]


def _layer_row(
    layer: int,
    token_type: str,
    K: np.ndarray,
    V: np.ndarray,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "layer": layer,
        "token_type": token_type,
        "n_tokens": int(K.shape[0]),
    }

    k_stats = compute_value_stats(K)
    v_stats = compute_value_stats(V)
    k_norms = compute_norm_stats(K)
    v_norms = compute_norm_stats(V)

    for name, value in k_stats.items():
        row[f"k_{name}"] = value
    for name, value in v_stats.items():
        row[f"v_{name}"] = value
    for name, value in k_norms.items():
        row[f"k_{name}"] = value
    for name, value in v_norms.items():
        row[f"v_{name}"] = value

    row["kv_norm_ratio"] = (
        float(k_norms["norm_mean"] / v_norms["norm_mean"])
        if v_norms["norm_mean"]
        else float("nan")
    )
    return row


def compute_per_layer_stats(
    dump: KVDump,
    *,
    separate_vision_text: bool = True,
    token_types: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Per-layer K/V statistics for ``dump``.

    Parameters
    ----------
    dump:
        The loaded :class:`KVDump` instance.
    separate_vision_text:
        When true, emits rows for ``all`` plus ``vision`` and ``text``
        subsets (whichever are present).  When false, emits only ``all``.
    token_types:
        Optional explicit override, e.g. ``["vision"]``.
    """
    if token_types is None:
        if separate_vision_text:
            types: list[str] = ["all"]
            if dump.has_vision():
                types.append("vision")
            if dump.has_text():
                types.append("text")
        else:
            types = ["all"]
    else:
        types = list(token_types)

    rows: list[dict[str, float | int | str]] = []
    for layer in range(dump.n_layers):
        K = dump.get_K(layer)
        V = dump.get_V(layer)
        for token_type in types:
            if token_type == "all":
                K_sub = K
                V_sub = V
            elif token_type == "vision":
                mask = dump.vision_mask()
                if not mask.any():
                    continue
                K_sub = K[mask]
                V_sub = V[mask]
            elif token_type == "text":
                mask = dump.text_mask()
                if not mask.any():
                    continue
                K_sub = K[mask]
                V_sub = V[mask]
            else:
                raise ValueError(f"unknown token_type: {token_type}")
            rows.append(_layer_row(layer, token_type, K_sub, V_sub))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    sort_cols = [c for c in ("layer", "token_type") if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)
