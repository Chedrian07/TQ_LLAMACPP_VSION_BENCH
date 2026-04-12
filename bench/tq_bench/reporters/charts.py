"""Visualization functions for benchmark results.

All chart functions accept a pandas DataFrame with at least:
  - ``runtime_id``: str
  - ``benchmark_id``: str
  - ``score``: float | None
  - ``status``: str

Charts are saved to ``results/reports/`` as 300-DPI PNGs.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; must be set before pyplot import

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Runtime ordering from low to high bits, grouped by method
_RUNTIME_ORDER = [
    "baseline",
    "lcpp-kv-8",
    "lcpp-kv-4",
    "lcpp-kv-2",
    "tq-4",
    "tq-3h",
    "tq-3",
    "tq-2h",
    "tq-2",
    "tq-K4V3",
    "tq-K4V2",
    "tq-K3V2",
    "tqp-5",
    "tqp-4",
    "tqp-3",
]

# Benchmark ordering: VLM first, then text
_BENCHMARK_ORDER = [
    "ai2d",
    "chartqa",
    "chartqapro",
    "docvqa",
    "mathvista",
    "mmmu",
    "ocrbench_v2",
    "textvqa",
    "mmlu",
    "commonsenseqa",
    "hellaswag",
]

# Color palette by method
_METHOD_COLORS: dict[str, str] = {
    "none": "#4a4a4a",           # dark gray for baseline
    "llama.cpp-native": "#3274a1",  # blue
    "turboquant-mse": "#e1812c",    # orange
    "turboquant-prod": "#d62728",   # red
}

# Map runtime_id to method for coloring
_RUNTIME_METHOD: dict[str, str] = {
    "baseline": "none",
    "lcpp-kv-8": "llama.cpp-native",
    "lcpp-kv-4": "llama.cpp-native",
    "lcpp-kv-2": "llama.cpp-native",
    "tq-2": "turboquant-mse",
    "tq-2h": "turboquant-mse",
    "tq-3": "turboquant-mse",
    "tq-3h": "turboquant-mse",
    "tq-4": "turboquant-mse",
    "tq-K4V2": "turboquant-mse",
    "tq-K4V3": "turboquant-mse",
    "tq-K3V2": "turboquant-mse",
    "tqp-3": "turboquant-prod",
    "tqp-4": "turboquant-prod",
    "tqp-5": "turboquant-prod",
}

# Bits mapping for degradation curves
_RUNTIME_BITS: dict[str, float] = {
    "baseline": 16.0,
    "lcpp-kv-8": 8.0,
    "lcpp-kv-4": 4.0,
    "lcpp-kv-2": 2.0,
    "tq-2": 2.0,
    "tq-2h": 2.5,
    "tq-3": 3.0,
    "tq-3h": 3.5,
    "tq-4": 4.0,
    "tqp-3": 3.0,
    "tqp-4": 4.0,
    "tqp-5": 5.0,
}

# VLM and text benchmark sets
_VLM_BENCHMARKS = {"ai2d", "chartqa", "chartqapro", "docvqa", "mathvista", "mmmu", "ocrbench_v2", "textvqa"}
_TEXT_BENCHMARKS = {"mmlu", "commonsenseqa", "hellaswag"}

_DPI = 300


# ---------------------------------------------------------------------------
# Helper: prepare dataframe
# ---------------------------------------------------------------------------

def _prepare_df(results_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has the expected columns and fill missing scores."""
    df = results_df.copy()
    if "score" not in df.columns:
        raise ValueError("DataFrame must contain a 'score' column")
    # Replace None scores with 0 for visualization
    df["score"] = df["score"].fillna(0.0)
    return df


def _get_runtime_color(runtime_id: str) -> str:
    """Return the color for a runtime based on its method category."""
    method = _RUNTIME_METHOD.get(runtime_id, "none")
    return _METHOD_COLORS.get(method, "#999999")


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Save figure to disk at 300 DPI and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Chart saved: %s", path)


# ---------------------------------------------------------------------------
# 1. Heatmap: Runtime x Benchmark
# ---------------------------------------------------------------------------

def heatmap(
    results_df: pd.DataFrame,
    output_path: Path | None = None,
    *,
    title: str = "KV Cache Quantization Benchmark Results",
    figsize: tuple[float, float] = (16, 10),
) -> plt.Figure:
    """Generate a heatmap of scores: rows = runtimes, columns = benchmarks.

    Runtimes are grouped by method (baseline, llama.cpp-native,
    turboquant-mse, turboquant-prod). Cells with server_crash/fail
    status are annotated differently.

    Args:
        results_df: DataFrame with runtime_id, benchmark_id, score, status.
        output_path: If provided, saves the figure to this path.
        title: Chart title.
        figsize: Figure size (width, height).

    Returns:
        The matplotlib Figure.
    """
    df = _prepare_df(results_df)

    # Pivot to matrix form
    pivot = df.pivot_table(
        index="runtime_id",
        columns="benchmark_id",
        values="score",
        aggfunc="first",
    )

    # Reorder rows and columns
    row_order = [r for r in _RUNTIME_ORDER if r in pivot.index]
    col_order = [c for c in _BENCHMARK_ORDER if c in pivot.columns]
    pivot = pivot.reindex(index=row_order, columns=col_order)

    # Build status matrix for annotations
    status_pivot = df.pivot_table(
        index="runtime_id",
        columns="benchmark_id",
        values="status",
        aggfunc="first",
    ).reindex(index=row_order, columns=col_order)

    # Create annotation matrix
    annot = pivot.copy().astype(str)
    for r in row_order:
        for c in col_order:
            score_val = pivot.loc[r, c] if r in pivot.index and c in pivot.columns else np.nan
            status_val = (
                status_pivot.loc[r, c]
                if r in status_pivot.index and c in status_pivot.columns
                else None
            )
            if pd.isna(score_val):
                annot.loc[r, c] = "-"
            elif status_val in ("server_crash", "fail"):
                annot.loc[r, c] = f"FAIL\n{score_val:.3f}"
            else:
                annot.loc[r, c] = f"{score_val:.3f}"

    fig, ax = plt.subplots(figsize=figsize)

    # Draw heatmap
    sns.heatmap(
        pivot.astype(float),
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Score", "shrink": 0.8},
        annot_kws={"fontsize": 7},
    )

    # Add method group separators
    group_boundaries = []
    current_method = None
    for i, rid in enumerate(row_order):
        method = _RUNTIME_METHOD.get(rid, "none")
        if method != current_method and current_method is not None:
            group_boundaries.append(i)
        current_method = method

    for boundary in group_boundaries:
        ax.axhline(y=boundary, color="black", linewidth=2)

    # VLM/Text separator
    vlm_count = sum(1 for c in col_order if c in _VLM_BENCHMARKS)
    if 0 < vlm_count < len(col_order):
        ax.axvline(x=vlm_count, color="black", linewidth=2)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Benchmark", fontsize=11)
    ax.set_ylabel("Runtime", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()

    if output_path is not None:
        _save_figure(fig, output_path)

    return fig


# ---------------------------------------------------------------------------
# 2. Bar chart: Compare runtimes for one benchmark
# ---------------------------------------------------------------------------

def bar_chart(
    results_df: pd.DataFrame,
    benchmark_id: str,
    output_path: Path | None = None,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (14, 6),
) -> plt.Figure:
    """Compare all runtimes for a single benchmark as a grouped bar chart.

    Args:
        results_df: DataFrame with runtime_id, benchmark_id, score, status.
        benchmark_id: Which benchmark to plot.
        output_path: If provided, saves the figure to this path.
        title: Optional title override.
        figsize: Figure size (width, height).

    Returns:
        The matplotlib Figure.
    """
    df = _prepare_df(results_df)
    bench_df = df[df["benchmark_id"] == benchmark_id].copy()

    if bench_df.empty:
        raise ValueError(f"No results found for benchmark '{benchmark_id}'")

    # Order runtimes
    ordered_runtimes = [r for r in _RUNTIME_ORDER if r in bench_df["runtime_id"].values]
    bench_df = bench_df.set_index("runtime_id").reindex(ordered_runtimes).reset_index()

    # Colors by method
    colors = [_get_runtime_color(rid) for rid in bench_df["runtime_id"]]

    # Identify failures
    is_fail = bench_df.get("status", pd.Series(dtype=str)).isin(
        ["server_crash", "fail"]
    )

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        range(len(bench_df)),
        bench_df["score"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    # Mark failed bars with hatching
    for i, (bar, fail) in enumerate(zip(bars, is_fail)):
        if fail:
            bar.set_hatch("///")
            bar.set_edgecolor("black")

    # Annotate bars with scores
    for i, (bar, score) in enumerate(zip(bars, bench_df["score"])):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=45,
        )

    # Baseline reference line
    baseline_row = bench_df[bench_df["runtime_id"] == "baseline"]
    if not baseline_row.empty:
        baseline_score = baseline_row.iloc[0]["score"]
        ax.axhline(
            y=baseline_score,
            color=_METHOD_COLORS["none"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Baseline ({baseline_score:.3f})",
        )

    ax.set_xticks(range(len(bench_df)))
    ax.set_xticklabels(bench_df["runtime_id"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, min(1.05, bench_df["score"].max() * 1.15) if bench_df["score"].max() > 0 else 1.0)

    chart_title = title or f"Runtime Comparison: {benchmark_id}"
    ax.set_title(chart_title, fontsize=13, fontweight="bold")

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=m.replace("-", " ").title())
        for m, c in _METHOD_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    fig.tight_layout()

    if output_path is not None:
        _save_figure(fig, output_path)

    return fig


# ---------------------------------------------------------------------------
# 3. Degradation curve: Score vs bitwidth
# ---------------------------------------------------------------------------

def degradation_curve(
    results_df: pd.DataFrame,
    benchmark_id: str,
    output_path: Path | None = None,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Plot score vs bitwidth for a single benchmark.

    Draws separate lines for llama.cpp-native, turboquant-mse, and
    turboquant-prod methods. The baseline (16-bit) is shown as a
    horizontal reference. Prod failures are highlighted with markers.

    Args:
        results_df: DataFrame with runtime_id, benchmark_id, score, status.
        benchmark_id: Which benchmark to plot.
        output_path: If provided, saves the figure to this path.
        title: Optional title override.
        figsize: Figure size (width, height).

    Returns:
        The matplotlib Figure.
    """
    df = _prepare_df(results_df)
    bench_df = df[df["benchmark_id"] == benchmark_id].copy()

    if bench_df.empty:
        raise ValueError(f"No results found for benchmark '{benchmark_id}'")

    # Add bits column
    bench_df["bits"] = bench_df["runtime_id"].map(_RUNTIME_BITS)
    bench_df["method"] = bench_df["runtime_id"].map(_RUNTIME_METHOD)

    # Drop runtimes that do not have a numeric bits mapping (asymmetric K/V)
    bench_df = bench_df.dropna(subset=["bits"])

    fig, ax = plt.subplots(figsize=figsize)

    # Baseline horizontal line
    baseline_row = bench_df[bench_df["runtime_id"] == "baseline"]
    if not baseline_row.empty:
        baseline_score = baseline_row.iloc[0]["score"]
        ax.axhline(
            y=baseline_score,
            color=_METHOD_COLORS["none"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Baseline FP16 ({baseline_score:.3f})",
        )

    # Plot each method as a separate line
    method_labels = {
        "llama.cpp-native": "llama.cpp native",
        "turboquant-mse": "TurboQuant MSE (Alg.1)",
        "turboquant-prod": "TurboQuant prod (Alg.2)",
    }

    for method, label in method_labels.items():
        method_df = bench_df[bench_df["method"] == method].sort_values("bits")
        if method_df.empty:
            continue

        color = _METHOD_COLORS[method]

        # Identify failures
        is_fail = method_df.get("status", pd.Series(dtype=str)).isin(
            ["server_crash", "fail"]
        )

        # Plot the line
        ax.plot(
            method_df["bits"],
            method_df["score"],
            color=color,
            marker="o",
            markersize=8,
            linewidth=2,
            label=label,
            zorder=3,
        )

        # Highlight failures with red X markers
        fail_df = method_df[is_fail]
        if not fail_df.empty:
            ax.scatter(
                fail_df["bits"],
                fail_df["score"],
                color="red",
                marker="X",
                s=150,
                zorder=4,
                label=f"{label} (FAIL)" if not fail_df.empty else None,
            )

        # Annotate points
        for _, row in method_df.iterrows():
            ax.annotate(
                f"{row['score']:.3f}",
                (row["bits"], row["score"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=7,
                color=color,
            )

    ax.set_xlabel("Bits per KV element", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlim(1.5, 17)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 2.5, 3, 3.5, 4, 5, 8, 16])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="x", rotation=0)

    chart_title = title or f"Score vs Bitwidth: {benchmark_id}"
    ax.set_title(chart_title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()

    if output_path is not None:
        _save_figure(fig, output_path)

    return fig


# ---------------------------------------------------------------------------
# 4. Scatter: VLM avg score vs Text avg score per runtime
# ---------------------------------------------------------------------------

def scatter_vlm_vs_text(
    results_df: pd.DataFrame,
    output_path: Path | None = None,
    *,
    title: str = "VLM vs Text Performance by Runtime",
    figsize: tuple[float, float] = (9, 9),
) -> plt.Figure:
    """Scatter plot of VLM average score vs Text average score per runtime.

    Each point represents one runtime. Points are colored by method and
    labeled with the runtime ID.

    Args:
        results_df: DataFrame with runtime_id, benchmark_id, score, status.
        output_path: If provided, saves the figure to this path.
        title: Chart title.
        figsize: Figure size (width, height).

    Returns:
        The matplotlib Figure.
    """
    df = _prepare_df(results_df)

    # Split into VLM and text
    vlm_df = df[df["benchmark_id"].isin(_VLM_BENCHMARKS)]
    text_df = df[df["benchmark_id"].isin(_TEXT_BENCHMARKS)]

    # Compute per-runtime averages
    vlm_avg = vlm_df.groupby("runtime_id")["score"].mean().rename("vlm_avg")
    text_avg = text_df.groupby("runtime_id")["score"].mean().rename("text_avg")

    combined = pd.concat([vlm_avg, text_avg], axis=1).dropna()
    if combined.empty:
        raise ValueError("No runtimes have both VLM and text results")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each runtime
    for runtime_id, row in combined.iterrows():
        color = _get_runtime_color(str(runtime_id))
        method = _RUNTIME_METHOD.get(str(runtime_id), "none")

        ax.scatter(
            row["text_avg"],
            row["vlm_avg"],
            c=color,
            s=120,
            edgecolors="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            str(runtime_id),
            (row["text_avg"], row["vlm_avg"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
            color=color,
            fontweight="bold",
        )

    # Reference line y=x
    all_scores = list(combined["vlm_avg"]) + list(combined["text_avg"])
    lo = min(all_scores) * 0.95 if all_scores else 0
    hi = max(all_scores) * 1.05 if all_scores else 1
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle=":", alpha=0.5, label="y = x")

    ax.set_xlabel("Text Benchmark Average Score", fontsize=11)
    ax.set_ylabel("VLM Benchmark Average Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=m.replace("-", " ").title())
        for m, c in _METHOD_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Equal aspect for better visual comparison
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    if output_path is not None:
        _save_figure(fig, output_path)

    return fig


# ---------------------------------------------------------------------------
# Convenience: generate all charts from a results directory
# ---------------------------------------------------------------------------

def generate_all_charts(
    results_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate all standard charts and save them to *output_dir*.

    Returns a list of file paths for the generated chart PNGs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # 1. Heatmap
    try:
        heatmap_path = output_dir / "heatmap.png"
        heatmap(results_df, heatmap_path)
        generated.append(heatmap_path)
    except Exception as exc:
        logger.warning("Failed to generate heatmap: %s", exc)

    # 2. Bar charts per benchmark
    benchmarks = results_df["benchmark_id"].unique()
    for bench_id in benchmarks:
        try:
            bar_path = output_dir / f"bar_{bench_id}.png"
            bar_chart(results_df, bench_id, bar_path)
            generated.append(bar_path)
        except Exception as exc:
            logger.warning("Failed to generate bar chart for %s: %s", bench_id, exc)

    # 3. Degradation curves per benchmark
    for bench_id in benchmarks:
        try:
            curve_path = output_dir / f"degradation_{bench_id}.png"
            degradation_curve(results_df, bench_id, curve_path)
            generated.append(curve_path)
        except Exception as exc:
            logger.warning(
                "Failed to generate degradation curve for %s: %s", bench_id, exc
            )

    # 4. VLM vs Text scatter
    try:
        scatter_path = output_dir / "scatter_vlm_vs_text.png"
        scatter_vlm_vs_text(results_df, scatter_path)
        generated.append(scatter_path)
    except Exception as exc:
        logger.warning("Failed to generate VLM vs Text scatter: %s", exc)

    logger.info("Generated %d charts in %s", len(generated), output_dir)
    return generated
