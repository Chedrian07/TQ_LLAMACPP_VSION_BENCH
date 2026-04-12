"""TurboQuant codebook analysis and reproducibility checks.

This module provides tools for analysing how the TurboQuant Lloyd-Max
codebook interacts with real KV cache data:

1. :func:`quantize_to_codebook` -- assign rotated values to nearest
   centroid.
2. :func:`codebook_bucket_usage` -- per-layer bucket frequency after
   rotation + quantization.
3. :func:`plot_codebook_usage` -- bar chart visualising bucket occupancy
   versus the theoretical uniform baseline.
4. :func:`codebook_vision_vs_text_skew` -- Jensen-Shannon divergence
   between vision and text token bucket distributions.
5. :func:`plot_codebook_skew` -- line chart of JS divergence per layer.
6. :func:`bit_exact_reproducibility` -- verify two KV dumps are
   bitwise identical.

Codebook constants are copied from the C++ reference
(``ggml-common-turbo.h``) and must be kept in sync.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .loader import KVDump
from .rotation_analysis import TURBO_SEED, apply_fwht

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lloyd-Max centroids (Beta-optimal, d=128)
# ---------------------------------------------------------------------------

TURBO_CENTROIDS: dict[int, np.ndarray] = {
    2: np.array(
        [-1.5104176085, -0.4527800346, 0.4527800346, 1.5104176085],
        dtype=np.float64,
    ),
    3: np.array(
        [
            -2.1519457045,
            -1.3439092785,
            -0.7560052812,
            -0.2450941789,
            0.2450941789,
            0.7560052812,
            1.3439092785,
            2.1519457045,
        ],
        dtype=np.float64,
    ),
    4: np.array(
        [
            -2.7325895588,
            -2.0690172128,
            -1.6180463720,
            -1.2562311842,
            -0.9423404451,
            -0.6567591097,
            -0.3880482939,
            -0.1283950280,
            0.1283950280,
            0.3880482939,
            0.6567591097,
            0.9423404451,
            1.2562311842,
            1.6180463720,
            2.0690172128,
            2.7325895588,
        ],
        dtype=np.float64,
    ),
}


# ---------------------------------------------------------------------------
# Core quantization
# ---------------------------------------------------------------------------


def quantize_to_codebook(
    values: np.ndarray,
    bits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each value to the nearest centroid.

    Parameters
    ----------
    values:
        Arbitrary-shape array of float values to quantize.
    bits:
        Codebook bitwidth (2, 3, or 4).

    Returns
    -------
    indices:
        Integer array (same shape as *values*) of centroid indices.
    centroids:
        The 1-D centroid array used for the quantization.

    Raises
    ------
    ValueError
        If *bits* is not in :data:`TURBO_CENTROIDS`.
    """
    if bits not in TURBO_CENTROIDS:
        raise ValueError(
            f"bits must be one of {sorted(TURBO_CENTROIDS.keys())}, got {bits}"
        )
    centroids = TURBO_CENTROIDS[bits]
    flat = np.asarray(values, dtype=np.float64).ravel()
    # distances: (n_values, n_centroids)
    diffs = np.abs(flat[:, np.newaxis] - centroids[np.newaxis, :])
    indices_flat = np.argmin(diffs, axis=1)
    return indices_flat.reshape(values.shape), centroids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_tensor(dump: KVDump, layer: int, kind: str) -> np.ndarray:
    """Return the K or V tensor for a given layer."""
    if kind == "K":
        return dump.get_K(layer)
    if kind == "V":
        return dump.get_V(layer)
    raise ValueError(f"kind must be 'K' or 'V', got {kind!r}")


def _rotate_and_quantize(
    tensor: np.ndarray,
    bits: int,
    seed: int = TURBO_SEED,
) -> np.ndarray:
    """Rotate all head vectors in *tensor* (T, H, D) and quantize.

    Returns a 1-D index array of length ``T * H * D``.
    """
    T, H, D = tensor.shape
    flat = tensor.reshape(T * H, D)
    rotated = apply_fwht(flat, normalize=True, seed=seed)
    indices, _ = quantize_to_codebook(rotated, bits)
    return indices.ravel()


def _bucket_counts(indices: np.ndarray, n_buckets: int) -> np.ndarray:
    """Count occurrences of each bucket index (0..n_buckets-1)."""
    return np.bincount(indices.ravel().astype(np.intp), minlength=n_buckets)


# ---------------------------------------------------------------------------
# Bucket usage analysis
# ---------------------------------------------------------------------------


def codebook_bucket_usage(
    dump: KVDump,
    bits: int,
    kind: str = "K",
    token_type: str = "all",
    *,
    seed: int = TURBO_SEED,
) -> pd.DataFrame:
    """Per-layer bucket frequency after rotation + quantization.

    Parameters
    ----------
    dump:
        Loaded :class:`KVDump`.
    bits:
        Codebook bitwidth (2, 3, or 4).
    kind:
        ``'K'`` or ``'V'``.
    token_type:
        ``'all'``, ``'vision'``, or ``'text'``.
    seed:
        FWHT sign-flip seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``layer``, ``bucket_idx``, ``centroid_value``,
        ``count``, ``fraction``.
    """
    if bits not in TURBO_CENTROIDS:
        raise ValueError(
            f"bits must be one of {sorted(TURBO_CENTROIDS.keys())}, got {bits}"
        )
    centroids = TURBO_CENTROIDS[bits]
    n_buckets = len(centroids)
    tok_indices = dump.token_indices(token_type)

    rows: list[dict] = []
    for layer in range(dump.n_layers):
        tensor = _get_tensor(dump, layer, kind)
        sub = tensor[tok_indices] if token_type != "all" else tensor
        if sub.size == 0:
            continue
        indices = _rotate_and_quantize(sub, bits, seed=seed)
        counts = _bucket_counts(indices, n_buckets)
        total = int(counts.sum())
        for bucket_idx in range(n_buckets):
            rows.append({
                "layer": int(layer),
                "bucket_idx": int(bucket_idx),
                "centroid_value": float(centroids[bucket_idx]),
                "count": int(counts[bucket_idx]),
                "fraction": float(counts[bucket_idx]) / total if total > 0 else 0.0,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting: bucket usage
# ---------------------------------------------------------------------------


def plot_codebook_usage(
    dump: KVDump,
    bits: int,
    kind: str = "K",
    layers: Sequence[int] | None = None,
    out_dir: str | Path | None = None,
    *,
    seed: int = TURBO_SEED,
) -> list[plt.Figure]:
    """Bar chart per layer showing bucket usage.

    An overlay line marks the theoretical uniform distribution
    (``1 / n_buckets``).

    Parameters
    ----------
    dump:
        Loaded :class:`KVDump`.
    bits:
        Codebook bitwidth (2, 3, or 4).
    kind:
        ``'K'`` or ``'V'``.
    layers:
        Subset of layers to plot.  Defaults to all layers.
    out_dir:
        When set, PNG files are saved to this directory.
    seed:
        FWHT sign-flip seed.

    Returns
    -------
    list[plt.Figure]
        One figure per layer (closed after saving when *out_dir* is set).
    """
    df = codebook_bucket_usage(dump, bits, kind=kind, token_type="all", seed=seed)
    if df.empty:
        return []

    if layers is None:
        layers = sorted(df["layer"].unique())

    centroids = TURBO_CENTROIDS[bits]
    n_buckets = len(centroids)
    uniform_line = 1.0 / n_buckets

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    figs: list[plt.Figure] = []
    for layer in layers:
        layer_df = df[df["layer"] == layer].sort_values("bucket_idx")
        if layer_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        x_labels = [f"{c:.2f}" for c in layer_df["centroid_value"]]
        ax.bar(range(n_buckets), layer_df["fraction"].values, color="steelblue", alpha=0.8)
        ax.axhline(uniform_line, color="red", linestyle="--", linewidth=1.2, label="uniform")
        ax.set_xticks(range(n_buckets))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("centroid value")
        ax.set_ylabel("fraction")
        ax.set_title(f"Codebook bucket usage  |  {kind} layer {layer}  |  {bits}-bit")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        if out_dir is not None:
            fig.savefig(out_dir / f"codebook_usage_{kind}_L{layer}_{bits}bit.png", dpi=150)
            plt.close(fig)
        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# Vision vs text skew (Jensen-Shannon divergence)
# ---------------------------------------------------------------------------


def codebook_vision_vs_text_skew(
    dump: KVDump,
    bits: int,
    kind: str = "K",
    *,
    seed: int = TURBO_SEED,
) -> pd.DataFrame:
    """Jensen-Shannon divergence between vision and text bucket distributions.

    Parameters
    ----------
    dump:
        Loaded :class:`KVDump`.
    bits:
        Codebook bitwidth (2, 3, or 4).
    kind:
        ``'K'`` or ``'V'``.
    seed:
        FWHT sign-flip seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``layer``, ``js_divergence``.
        Empty if no vision tokens are present.
    """
    if not dump.has_vision():
        return pd.DataFrame(columns=["layer", "js_divergence"])

    centroids = TURBO_CENTROIDS[bits]
    n_buckets = len(centroids)
    vis_idx = dump.token_indices("vision")
    txt_idx = dump.token_indices("text")

    # Guard: if either token set is empty, divergence is undefined.
    if len(vis_idx) == 0 or len(txt_idx) == 0:
        return pd.DataFrame(columns=["layer", "js_divergence"])

    rows: list[dict[str, float | int]] = []
    for layer in range(dump.n_layers):
        tensor = _get_tensor(dump, layer, kind)

        vis_tensor = tensor[vis_idx]
        txt_tensor = tensor[txt_idx]

        vis_indices = _rotate_and_quantize(vis_tensor, bits, seed=seed)
        txt_indices = _rotate_and_quantize(txt_tensor, bits, seed=seed)

        vis_counts = _bucket_counts(vis_indices, n_buckets).astype(np.float64)
        txt_counts = _bucket_counts(txt_indices, n_buckets).astype(np.float64)

        # Normalise to probability distributions.
        vis_total = vis_counts.sum()
        txt_total = txt_counts.sum()
        if vis_total == 0 or txt_total == 0:
            rows.append({"layer": int(layer), "js_divergence": float("nan")})
            continue

        vis_prob = vis_counts / vis_total
        txt_prob = txt_counts / txt_total

        js_div = float(jensenshannon(vis_prob, txt_prob) ** 2)
        rows.append({"layer": int(layer), "js_divergence": js_div})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting: codebook skew
# ---------------------------------------------------------------------------


def plot_codebook_skew(
    dump: KVDump,
    bits: int,
    kind: str = "K",
    out_path: str | Path | None = None,
    *,
    seed: int = TURBO_SEED,
) -> plt.Figure | None:
    """Line plot: x = layer, y = JS divergence (vision vs text).

    Parameters
    ----------
    dump:
        Loaded :class:`KVDump`.
    bits:
        Codebook bitwidth (2, 3, or 4).
    kind:
        ``'K'`` or ``'V'``.
    out_path:
        When set, the figure is saved to this path.
    seed:
        FWHT sign-flip seed.

    Returns
    -------
    plt.Figure or None
        The figure, or ``None`` if no data was available.
    """
    df = codebook_vision_vs_text_skew(dump, bits, kind=kind, seed=seed)
    if df.empty:
        return None

    df = df.sort_values("layer")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["layer"], df["js_divergence"], marker="o", color="darkorange")
    ax.set_xlabel("layer")
    ax.set_ylabel("JS divergence (vision vs text)")
    ax.set_title(f"Codebook bucket skew  |  {kind}  |  {bits}-bit")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Bit-exact reproducibility
# ---------------------------------------------------------------------------


def bit_exact_reproducibility(
    dump_a: KVDump,
    dump_b: KVDump,
) -> dict[str, int | float | bool]:
    """Compare two KV dumps for bitwise equality.

    This verifies that two independent runs with the same model and
    input produce identical KV caches (important for benchmarking
    reproducibility).

    Parameters
    ----------
    dump_a, dump_b:
        Two loaded :class:`KVDump` instances.

    Returns
    -------
    dict
        ``n_layers``, ``n_exact_K``, ``n_exact_V``,
        ``max_K_diff``, ``max_V_diff``, ``is_reproducible``.

    Raises
    ------
    ValueError
        If the two dumps have different shapes.
    """
    if dump_a.shape != dump_b.shape:
        raise ValueError(
            f"shape mismatch: dump_a={dump_a.shape} dump_b={dump_b.shape}"
        )

    n_layers = dump_a.n_layers
    n_exact_K = 0
    n_exact_V = 0
    max_K_diff: float = 0.0
    max_V_diff: float = 0.0

    for layer in range(n_layers):
        k_a = dump_a.get_K(layer)
        k_b = dump_b.get_K(layer)
        v_a = dump_a.get_V(layer)
        v_b = dump_b.get_V(layer)

        k_equal = np.array_equal(k_a, k_b)
        v_equal = np.array_equal(v_a, v_b)

        if k_equal:
            n_exact_K += 1
        else:
            diff = float(np.max(np.abs(k_a.astype(np.float64) - k_b.astype(np.float64))))
            max_K_diff = max(max_K_diff, diff)

        if v_equal:
            n_exact_V += 1
        else:
            diff = float(np.max(np.abs(v_a.astype(np.float64) - v_b.astype(np.float64))))
            max_V_diff = max(max_V_diff, diff)

    is_reproducible = (n_exact_K == n_layers) and (n_exact_V == n_layers)

    return {
        "n_layers": n_layers,
        "n_exact_K": n_exact_K,
        "n_exact_V": n_exact_V,
        "max_K_diff": max_K_diff,
        "max_V_diff": max_V_diff,
        "is_reproducible": is_reproducible,
    }
