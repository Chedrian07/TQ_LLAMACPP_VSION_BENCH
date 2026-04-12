"""Layer-wise line plots for KV cache analysis.

This module produces per-layer curves that visualize how KV cache
statistics and quantization errors vary across transformer layers.
The main plot types are:

1. **KV norm ratio curve** -- per-layer K/V norm ratio split by
   token type (all / vision / text).
2. **Layer distortion curves** -- per-layer per-coord MSE (log scale)
   for each runtime, K and V in separate subplots.
3. **Layer cosine similarity curves** -- same layout, y = cosine_sim.
4. **Layer relative error curves** -- same layout, y = relative_error.
5. **Convenience batch** -- :func:`plot_all_layer_curves` calls all
   of the above and writes PNGs to ``out_dir/plots/``.

All functions accept :class:`pandas.DataFrame` inputs that match the
schemas produced by :func:`~tq_bench.kv_analysis.distribution.compute_per_layer_stats`
and :func:`~tq_bench.kv_analysis.quant_error.compare_dumps`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

#: PNG save DPI (matches report.py / visualization.py convention).
_DPI: int = 150

#: Default color cycle for runtimes (tab10 palette subset).
_RUNTIME_COLORS: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]

#: Markers for token types in the norm ratio plot.
_TOKEN_TYPE_STYLES: dict[str, dict] = {
    "all":    {"marker": "o", "linestyle": "-",  "color": "#1f77b4"},
    "vision": {"marker": "s", "linestyle": "--", "color": "#ff7f0e"},
    "text":   {"marker": "^", "linestyle": "-.", "color": "#2ca02c"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finish_figure(
    fig: plt.Figure,
    out_path: Optional[Path],
) -> plt.Figure:
    """Tight-layout, save if requested, close after save."""
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=_DPI)
        plt.close(fig)
    return fig


def _runtime_color(idx: int) -> str:
    return _RUNTIME_COLORS[idx % len(_RUNTIME_COLORS)]


# ---------------------------------------------------------------------------
# 1. KV norm ratio curve
# ---------------------------------------------------------------------------


def plot_kv_norm_ratio_curve(
    dist_df: pd.DataFrame,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Line plot of per-layer K/V norm ratio, split by token type.

    Parameters
    ----------
    dist_df:
        DataFrame from :func:`compute_per_layer_stats` (optionally
        tagged with a ``run`` column).  Must contain ``layer``,
        ``token_type``, and ``kv_norm_ratio`` columns.
    out_path:
        If provided the figure is saved as PNG and closed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    if dist_df.empty or "kv_norm_ratio" not in dist_df.columns:
        ax.text(
            0.5, 0.5, "No distribution data",
            ha="center", va="center", transform=ax.transAxes,
        )
        return _finish_figure(fig, out_path)

    # If a 'run' column exists, use only the first run (baseline).
    df = dist_df.copy()
    if "run" in df.columns:
        first_run = df["run"].iloc[0]
        df = df[df["run"] == first_run]

    for token_type, sub in df.groupby("token_type"):
        sub_sorted = sub.sort_values("layer")
        style = _TOKEN_TYPE_STYLES.get(
            str(token_type),
            {"marker": "D", "linestyle": ":", "color": "#7f7f7f"},
        )
        ax.plot(
            sub_sorted["layer"],
            sub_sorted["kv_norm_ratio"],
            label=str(token_type),
            **style,
        )

    ax.set_title("K/V norm ratio per layer")
    ax.set_xlabel("layer")
    ax.set_ylabel("||K||_2 / ||V||_2  (mean per token)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    return _finish_figure(fig, out_path)


# ---------------------------------------------------------------------------
# 2. Layer distortion curves (per-coord MSE, log scale)
# ---------------------------------------------------------------------------


def plot_layer_distortion_curves(
    quant_df: pd.DataFrame,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Line plot of per-layer per-coord MSE (log scale), K/V subplots.

    Parameters
    ----------
    quant_df:
        DataFrame from :func:`compare_dumps`, with columns ``layer``,
        ``kind`` (K/V), ``token_type``, ``per_coord_mse``, and
        optionally ``run`` to distinguish runtimes.
    out_path:
        If provided the figure is saved as PNG and closed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    if quant_df.empty or "per_coord_mse" not in quant_df.columns:
        for ax in axes:
            ax.text(
                0.5, 0.5, "No quant error data",
                ha="center", va="center", transform=ax.transAxes,
            )
        return _finish_figure(fig, out_path)

    # Filter to token_type == 'all' for clarity.
    df = quant_df.copy()
    if "token_type" in df.columns:
        df_all = df[df["token_type"] == "all"]
        if not df_all.empty:
            df = df_all

    # Group key: use 'run' if present, otherwise a single group.
    has_run = "run" in df.columns
    group_col = "run" if has_run else None

    for kind_idx, kind in enumerate(("K", "V")):
        ax = axes[kind_idx]
        kind_df = df[df["kind"] == kind] if "kind" in df.columns else df
        if kind_df.empty:
            ax.set_title(f"{kind} -- no data")
            continue

        if group_col:
            for idx, (run_name, sub) in enumerate(kind_df.groupby(group_col)):
                sub_sorted = sub.sort_values("layer")
                ax.plot(
                    sub_sorted["layer"],
                    sub_sorted["per_coord_mse"],
                    marker="o",
                    markersize=4,
                    label=str(run_name),
                    color=_runtime_color(idx),
                )
        else:
            sorted_df = kind_df.sort_values("layer")
            ax.plot(
                sorted_df["layer"],
                sorted_df["per_coord_mse"],
                marker="o",
                markersize=4,
                color=_RUNTIME_COLORS[0],
            )

        ax.set_yscale("log")
        ax.set_title(f"{kind} per-coord MSE per layer")
        ax.set_xlabel("layer")
        ax.set_ylabel("MSE / coord (log)")
        ax.grid(True, alpha=0.3)
        if has_run:
            ax.legend(fontsize=7)

    return _finish_figure(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Layer cosine similarity curves
# ---------------------------------------------------------------------------


def plot_layer_cosine_curves(
    quant_df: pd.DataFrame,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Line plot of per-layer cosine similarity, K/V subplots.

    Parameters
    ----------
    quant_df:
        DataFrame from :func:`compare_dumps`, with columns ``layer``,
        ``kind`` (K/V), ``token_type``, ``cosine_sim``, and optionally
        ``run``.
    out_path:
        If provided the figure is saved as PNG and closed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    if quant_df.empty or "cosine_sim" not in quant_df.columns:
        for ax in axes:
            ax.text(
                0.5, 0.5, "No quant error data",
                ha="center", va="center", transform=ax.transAxes,
            )
        return _finish_figure(fig, out_path)

    df = quant_df.copy()
    if "token_type" in df.columns:
        df_all = df[df["token_type"] == "all"]
        if not df_all.empty:
            df = df_all

    has_run = "run" in df.columns
    group_col = "run" if has_run else None

    for kind_idx, kind in enumerate(("K", "V")):
        ax = axes[kind_idx]
        kind_df = df[df["kind"] == kind] if "kind" in df.columns else df
        if kind_df.empty:
            ax.set_title(f"{kind} -- no data")
            continue

        if group_col:
            for idx, (run_name, sub) in enumerate(kind_df.groupby(group_col)):
                sub_sorted = sub.sort_values("layer")
                ax.plot(
                    sub_sorted["layer"],
                    sub_sorted["cosine_sim"],
                    marker="o",
                    markersize=4,
                    label=str(run_name),
                    color=_runtime_color(idx),
                )
        else:
            sorted_df = kind_df.sort_values("layer")
            ax.plot(
                sorted_df["layer"],
                sorted_df["cosine_sim"],
                marker="o",
                markersize=4,
                color=_RUNTIME_COLORS[0],
            )

        ax.set_title(f"{kind} cosine similarity per layer")
        ax.set_xlabel("layer")
        ax.set_ylabel("cosine similarity")
        ax.grid(True, alpha=0.3)
        if has_run:
            ax.legend(fontsize=7)

    return _finish_figure(fig, out_path)


# ---------------------------------------------------------------------------
# 4. Layer relative error curves
# ---------------------------------------------------------------------------


def plot_layer_relative_error_curves(
    quant_df: pd.DataFrame,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Line plot of per-layer relative error, K/V subplots.

    Parameters
    ----------
    quant_df:
        DataFrame from :func:`compare_dumps`, with columns ``layer``,
        ``kind`` (K/V), ``token_type``, ``relative_error``, and
        optionally ``run``.
    out_path:
        If provided the figure is saved as PNG and closed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    if quant_df.empty or "relative_error" not in quant_df.columns:
        for ax in axes:
            ax.text(
                0.5, 0.5, "No quant error data",
                ha="center", va="center", transform=ax.transAxes,
            )
        return _finish_figure(fig, out_path)

    df = quant_df.copy()
    if "token_type" in df.columns:
        df_all = df[df["token_type"] == "all"]
        if not df_all.empty:
            df = df_all

    has_run = "run" in df.columns
    group_col = "run" if has_run else None

    for kind_idx, kind in enumerate(("K", "V")):
        ax = axes[kind_idx]
        kind_df = df[df["kind"] == kind] if "kind" in df.columns else df
        if kind_df.empty:
            ax.set_title(f"{kind} -- no data")
            continue

        if group_col:
            for idx, (run_name, sub) in enumerate(kind_df.groupby(group_col)):
                sub_sorted = sub.sort_values("layer")
                ax.plot(
                    sub_sorted["layer"],
                    sub_sorted["relative_error"],
                    marker="o",
                    markersize=4,
                    label=str(run_name),
                    color=_runtime_color(idx),
                )
        else:
            sorted_df = kind_df.sort_values("layer")
            ax.plot(
                sorted_df["layer"],
                sorted_df["relative_error"],
                marker="o",
                markersize=4,
                color=_RUNTIME_COLORS[0],
            )

        ax.set_title(f"{kind} relative error per layer")
        ax.set_xlabel("layer")
        ax.set_ylabel("||baseline - quant||_F / ||baseline||_F")
        ax.grid(True, alpha=0.3)
        if has_run:
            ax.legend(fontsize=7)

    return _finish_figure(fig, out_path)


# ---------------------------------------------------------------------------
# 5. Convenience batch
# ---------------------------------------------------------------------------


def plot_all_layer_curves(
    dist_df: pd.DataFrame,
    quant_df: pd.DataFrame,
    out_dir: str | Path,
) -> list[Path]:
    """Generate all layer-wise curve plots and save to ``out_dir/plots/``.

    Parameters
    ----------
    dist_df:
        Distribution DataFrame (from :func:`compute_per_layer_stats`).
    quant_df:
        Quantization error DataFrame (from :func:`compare_dumps`).
    out_dir:
        Root output directory.  A ``plots/`` subdirectory is created.

    Returns
    -------
    list[Path]
        Absolute paths of all PNG files written.
    """
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    p = plots_dir / "kv_norm_ratio_curve.png"
    plot_kv_norm_ratio_curve(dist_df, out_path=p)
    written.append(p)

    p = plots_dir / "layer_distortion_curves.png"
    plot_layer_distortion_curves(quant_df, out_path=p)
    written.append(p)

    p = plots_dir / "layer_cosine_curves.png"
    plot_layer_cosine_curves(quant_df, out_path=p)
    written.append(p)

    p = plots_dir / "layer_relative_error_curves.png"
    plot_layer_relative_error_curves(quant_df, out_path=p)
    written.append(p)

    logger.info("Wrote %d layer-curve PNGs to %s", len(written), plots_dir)
    return written
