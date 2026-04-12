"""Histogram visualizations for KV cache analysis.

This module provides functions to plot coordinate-value distributions
of K/V cache tensors.  The three main plot types are:

1. **Value histogram** -- flat distribution of all (head, dim) values
   for a single layer, optionally filtered by token type.
2. **Pre-rotation vs post-rotation overlay** -- shows how the FWHT
   rotation reshapes the coordinate distribution toward the Beta
   density that TurboQuant's Lloyd-Max codebook assumes.
3. **Vision vs text overlay** -- compares the coordinate distribution
   of vision tokens against text tokens within the same layer.

All functions return a :class:`matplotlib.figure.Figure` and
optionally save to disk when ``out_path`` is provided.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .loader import KVDump
from .rotation_analysis import _rotate_all_head_vectors

logger = logging.getLogger(__name__)

#: Default layers to analyse when the caller does not specify.
#: Evenly spaced across a 28-layer model (Qwen3-VL-2B).
DEFAULT_LAYERS: list[int] = [0, 7, 14, 21, 27]

#: PNG save DPI (matches report.py convention).
_DPI: int = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_values(
    dump: KVDump,
    layer: int,
    kind: str,
    token_type: str = "all",
) -> np.ndarray:
    """Flatten KV tensor values for ``(layer, kind)`` filtered by token type.

    Returns a 1-D float array.  May be empty if the token mask selects
    no tokens.
    """
    if kind not in ("K", "V"):
        raise ValueError(f"kind must be 'K' or 'V', got {kind!r}")
    tensor = dump.get_K(layer) if kind == "K" else dump.get_V(layer)
    indices = dump.token_indices(token_type)
    if indices.size == 0:
        return np.array([], dtype=np.float32)
    return tensor[indices].ravel()


def _finish_figure(
    fig: plt.Figure,
    out_path: Path | None,
) -> plt.Figure:
    """Tight-layout, save if requested, close after save."""
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=_DPI)
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 1. Per-layer value histogram
# ---------------------------------------------------------------------------


def plot_value_histogram(
    dump: KVDump,
    layer: int,
    kind: str,
    token_type: str = "all",
    bins: int = 200,
    out_path: Path | None = None,
) -> plt.Figure:
    """Plot a histogram of flattened coordinate values for one layer.

    Parameters
    ----------
    dump:
        Loaded :class:`KVDump`.
    layer:
        Layer index (0-based).
    kind:
        ``'K'`` or ``'V'``.
    token_type:
        ``'all'`` | ``'vision'`` | ``'text'``.
    bins:
        Number of histogram bins.
    out_path:
        If provided the figure is saved as a PNG and closed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    values = _extract_values(dump, layer, kind, token_type)

    fig, ax = plt.subplots(figsize=(8, 4))
    if values.size == 0:
        ax.text(
            0.5, 0.5, f"No {token_type} tokens",
            ha="center", va="center", transform=ax.transAxes,
        )
    else:
        ax.hist(values, bins=bins, density=True, alpha=0.7, color="steelblue",
                edgecolor="none")

    ax.set_title(f"{kind} value distribution  (layer {layer}, {token_type})")
    ax.set_xlabel("coordinate value")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.3)

    return _finish_figure(fig, out_path)


# ---------------------------------------------------------------------------
# 2. Pre-rotation vs post-rotation overlay
# ---------------------------------------------------------------------------


def plot_prerot_vs_postrot(
    dump: KVDump,
    layer: int,
    kind: str = "K",
    bins: int = 200,
    out_path: Path | None = None,
) -> plt.Figure:
    """Overlay raw and FWHT-rotated coordinate distributions.

    The pre-rotation histogram shows the raw KV coordinate values
    flattened across all heads.  The post-rotation histogram applies
    :func:`_rotate_all_head_vectors` (unit-normalise, sign-flip, FWHT)
    before flattening.

    Parameters
    ----------
    dump:
        Loaded :class:`KVDump`.
    layer:
        Layer index.
    kind:
        ``'K'`` or ``'V'``.
    bins:
        Number of histogram bins.
    out_path:
        Optional PNG save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if kind not in ("K", "V"):
        raise ValueError(f"kind must be 'K' or 'V', got {kind!r}")

    tensor = dump.get_K(layer) if kind == "K" else dump.get_V(layer)
    raw_values = tensor.ravel()
    rotated = _rotate_all_head_vectors(tensor)
    rot_values = rotated.ravel()

    fig, ax = plt.subplots(figsize=(8, 4))

    # Compute a shared bin range that covers both distributions.
    lo = min(raw_values.min(), rot_values.min())
    hi = max(raw_values.max(), rot_values.max())
    bin_edges = np.linspace(lo, hi, bins + 1)

    ax.hist(raw_values, bins=bin_edges, density=True, alpha=0.5,
            color="coral", edgecolor="none", label="pre-rotation (raw)")
    ax.hist(rot_values, bins=bin_edges, density=True, alpha=0.5,
            color="royalblue", edgecolor="none", label="post-rotation (FWHT)")

    ax.set_title(f"{kind} pre- vs post-rotation  (layer {layer})")
    ax.set_xlabel("coordinate value")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    return _finish_figure(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Vision vs text histogram
# ---------------------------------------------------------------------------


def plot_vision_vs_text_histogram(
    dump: KVDump,
    layer: int,
    kind: str = "K",
    bins: int = 200,
    out_path: Path | None = None,
) -> plt.Figure:
    """Overlay vision-token and text-token coordinate distributions.

    When the dump has no vision tokens or no text tokens the missing
    distribution is skipped and a warning is logged.

    Parameters
    ----------
    dump:
        Loaded :class:`KVDump`.
    layer:
        Layer index.
    kind:
        ``'K'`` or ``'V'``.
    bins:
        Number of histogram bins.
    out_path:
        Optional PNG save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    vision_vals = _extract_values(dump, layer, kind, token_type="vision")
    text_vals = _extract_values(dump, layer, kind, token_type="text")

    has_vision = vision_vals.size > 0
    has_text = text_vals.size > 0

    if not has_vision:
        logger.warning(
            "No vision tokens for layer %d %s; skipping vision distribution.",
            layer, kind,
        )
    if not has_text:
        logger.warning(
            "No text tokens for layer %d %s; skipping text distribution.",
            layer, kind,
        )

    fig, ax = plt.subplots(figsize=(8, 4))

    if not has_vision and not has_text:
        ax.text(
            0.5, 0.5, "No vision or text tokens",
            ha="center", va="center", transform=ax.transAxes,
        )
    else:
        # Build shared bin edges from whichever values exist.
        all_vals = np.concatenate(
            [v for v in (vision_vals, text_vals) if v.size > 0]
        )
        bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)

        if has_vision:
            ax.hist(vision_vals, bins=bin_edges, density=True, alpha=0.5,
                    color="darkorange", edgecolor="none", label="vision")
        if has_text:
            ax.hist(text_vals, bins=bin_edges, density=True, alpha=0.5,
                    color="teal", edgecolor="none", label="text")

        ax.legend(fontsize=9)

    ax.set_title(f"{kind} vision vs text  (layer {layer})")
    ax.set_xlabel("coordinate value")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.3)

    return _finish_figure(fig, out_path)


# ---------------------------------------------------------------------------
# 4. Batch generation
# ---------------------------------------------------------------------------


def generate_all_histograms(
    dump: KVDump,
    out_dir: str | Path,
    layers: list[int] | None = None,
    bins: int = 200,
) -> list[Path]:
    """Generate all histogram types for selected layers.

    Output layout::

        out_dir/histograms/
            value_K_layer{L}_all.png
            value_V_layer{L}_all.png
            prerot_vs_postrot_K_layer{L}.png
            prerot_vs_postrot_V_layer{L}.png
            vision_vs_text_K_layer{L}.png
            vision_vs_text_V_layer{L}.png

    Parameters
    ----------
    dump:
        Loaded :class:`KVDump`.
    out_dir:
        Root output directory.  A ``histograms/`` subdirectory is
        created beneath it.
    layers:
        Layer indices to process.  Defaults to
        :data:`DEFAULT_LAYERS`, clamped to ``[0, dump.n_layers)``.
    bins:
        Number of histogram bins forwarded to every plot function.

    Returns
    -------
    list[Path]
        Absolute paths of all PNG files written.
    """
    out_dir = Path(out_dir)
    hist_dir = out_dir / "histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)

    if layers is None:
        layers = [l for l in DEFAULT_LAYERS if l < dump.n_layers]
    else:
        layers = [l for l in layers if 0 <= l < dump.n_layers]

    if not layers:
        logger.warning("No valid layers to plot (n_layers=%d).", dump.n_layers)
        return []

    written: list[Path] = []

    for layer in layers:
        for kind in ("K", "V"):
            # 1. Value histogram (all tokens)
            p = hist_dir / f"value_{kind}_layer{layer}_all.png"
            plot_value_histogram(dump, layer, kind, token_type="all",
                                bins=bins, out_path=p)
            written.append(p)

            # 2. Pre-rot vs post-rot
            p = hist_dir / f"prerot_vs_postrot_{kind}_layer{layer}.png"
            plot_prerot_vs_postrot(dump, layer, kind=kind,
                                  bins=bins, out_path=p)
            written.append(p)

            # 3. Vision vs text (only if both types present)
            p = hist_dir / f"vision_vs_text_{kind}_layer{layer}.png"
            plot_vision_vs_text_histogram(dump, layer, kind=kind,
                                         bins=bins, out_path=p)
            written.append(p)

    logger.info(
        "Wrote %d histogram PNGs to %s", len(written), hist_dir,
    )
    return written
