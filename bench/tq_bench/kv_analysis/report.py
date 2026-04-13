"""End-to-end KV dump report generation.

Given a baseline dump plus a set of quantized dumps, this module
produces:

* ``distribution_stats.csv`` — per-layer K/V value / norm stats, one
  row per (run, layer, token_type).
* ``outliers_per_layer.csv`` — per-layer outlier ratios.
* ``quant_errors.csv`` — per-layer diff metrics (compare_dumps output)
  for every non-baseline run.
* ``rotation_analysis.csv`` — per-layer Beta fit + independence for
  every run.
* ``report.md`` — human-readable markdown summary.
* ``plots/*.png`` — matplotlib charts (optional, controlled by
  ``make_plots``).

This module is intentionally self-contained and imports nothing from
the rest of ``tq_bench`` so it can be called straight from a Jupyter
notebook or a shell script.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .attention_analysis import compare_attention_from_dumps
from .distribution import compute_per_layer_stats
from .layer_plots import plot_kv_norm_ratio_curve, plot_layer_distortion_curves
from .loader import KVDump, load_dump
from .outliers import outlier_ratio_vision_vs_text
from .quant_error import compare_dumps, summarize_against_theoretical
from .rotation_analysis import vision_vs_text_rotation_analysis


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tag_run(df: pd.DataFrame, run_name: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.insert(0, "run", run_name)
    return out


def _load_dumps(
    baseline_dump: KVDump | str | Path,
    quant_dumps: Mapping[str, KVDump | str | Path],
) -> tuple[KVDump, dict[str, KVDump]]:
    if not isinstance(baseline_dump, KVDump):
        baseline_dump = load_dump(baseline_dump)
    resolved: dict[str, KVDump] = {}
    for name, d in quant_dumps.items():
        resolved[name] = d if isinstance(d, KVDump) else load_dump(d)
    return baseline_dump, resolved


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_kv_norm_per_layer(
    dist_df: pd.DataFrame,
    out_path: Path,
) -> None:
    if dist_df.empty or "run" not in dist_df.columns:
        return
    baseline_only = dist_df[dist_df["run"] == dist_df["run"].iloc[0]]
    pivot = baseline_only.pivot_table(
        index="layer",
        columns="token_type",
        values="k_norm_mean",
        aggfunc="mean",
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(ax=ax, marker="o")
    ax.set_title(f"K norm mean per layer ({baseline_only['run'].iloc[0]})")
    ax.set_xlabel("layer")
    ax.set_ylabel("E[||K_token||_2]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_outlier_ratio(
    outlier_df: pd.DataFrame,
    out_path: Path,
) -> None:
    if outlier_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for tt, sub in outlier_df.groupby("token_type"):
        sub_k = sub.sort_values("layer")
        ax.plot(sub_k["layer"], sub_k["k_outlier_ratio"], marker="o", label=f"K ({tt})")
        ax.plot(sub_k["layer"], sub_k["v_outlier_ratio"], marker="x", label=f"V ({tt})")
    ax.set_title("Outlier channel ratio per layer")
    ax.set_xlabel("layer")
    ax.set_ylabel("outlier ratio (> 10x median)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_rotation_quality(
    rotation_df: pd.DataFrame,
    out_path: Path,
) -> None:
    if rotation_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for tt, sub in rotation_df.groupby("token_type"):
        for knd, sub2 in sub.groupby("kind"):
            sub2 = sub2.sort_values("layer")
            axes[0].plot(
                sub2["layer"],
                sub2["ks_statistic"],
                marker="o",
                label=f"{knd} ({tt})",
            )
            axes[1].plot(
                sub2["layer"],
                sub2["mean_abs_correlation"],
                marker="o",
                label=f"{knd} ({tt})",
            )
    axes[0].set_title("Beta fit KS statistic per layer (lower=better)")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("KS statistic")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].set_title("Rotated coord mean |correlation| per layer")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("mean |corr|")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_attention_comparison(
    attn_df: pd.DataFrame,
    out_path: Path,
) -> None:
    if attn_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for (run, tt), sub in attn_df.groupby(["run", "token_type"]):
        sub = sub.sort_values("layer")
        label = f"{run} ({tt})"
        axes[0].plot(sub["layer"], sub["kl_divergence"], marker="o", label=label, markersize=3)
        axes[1].plot(sub["layer"], sub["top1_match_rate"], marker="o", label=label, markersize=3)
        axes[2].plot(sub["layer"], sub["entropy_delta"], marker="o", label=label, markersize=3)
    axes[0].set_title("KL divergence per layer")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("KL(baseline || quant)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=6)
    axes[1].set_title("Top-1 attention match rate per layer")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("match rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=6)
    axes[2].set_title("Attention entropy delta per layer")
    axes[2].set_xlabel("layer")
    axes[2].set_ylabel("entropy(quant) - entropy(base)")
    axes[2].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_quant_error_per_layer(
    quant_df: pd.DataFrame,
    out_path: Path,
) -> None:
    if quant_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for (run, kind), sub in quant_df.groupby(["run", "kind"]):
        sub_all = sub[sub["token_type"] == "all"].sort_values("layer")
        if sub_all.empty:
            continue
        ax.plot(
            sub_all["layer"],
            sub_all["per_coord_mse"],
            marker="o",
            label=f"{run} / {kind}",
        )
    ax.set_yscale("log")
    ax.set_title("Per-coord MSE vs baseline per layer")
    ax.set_xlabel("layer")
    ax.set_ylabel("MSE / coord (log)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


def _fmt_num(x: float | int | None, width: int = 8, prec: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):>{width}d}"
    return f"{float(x):>{width}.{prec}f}"


def _markdown_section_distribution(dist_df: pd.DataFrame) -> str:
    if dist_df.empty:
        return "## Distribution\n\n_no data_\n"
    rows = [
        "## Distribution (mean per run)",
        "",
        "| run | token_type | K mean | K std | V mean | V std | K/V norm ratio |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    agg = dist_df.groupby(["run", "token_type"]).agg(
        k_mean=("k_mean", "mean"),
        k_std=("k_std", "mean"),
        v_mean=("v_mean", "mean"),
        v_std=("v_std", "mean"),
        kv_norm_ratio=("kv_norm_ratio", "mean"),
    )
    for (run, tt), row in agg.iterrows():
        rows.append(
            f"| {run} | {tt} | {_fmt_num(row['k_mean'])} | {_fmt_num(row['k_std'])} | "
            f"{_fmt_num(row['v_mean'])} | {_fmt_num(row['v_std'])} | "
            f"{_fmt_num(row['kv_norm_ratio'])} |"
        )
    return "\n".join(rows) + "\n"


def _markdown_section_outliers(outlier_df: pd.DataFrame) -> str:
    if outlier_df.empty:
        return "## Outlier channels\n\n_no data_\n"
    rows = [
        "## Outlier channels (mean per run)",
        "",
        "| run | token_type | K outlier ratio | V outlier ratio | K max/median | V max/median |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    agg = outlier_df.groupby(["run", "token_type"]).agg(
        k_out=("k_outlier_ratio", "mean"),
        v_out=("v_outlier_ratio", "mean"),
        k_mxmed=("k_max_to_median_ratio", "mean"),
        v_mxmed=("v_max_to_median_ratio", "mean"),
    )
    for (run, tt), row in agg.iterrows():
        rows.append(
            f"| {run} | {tt} | {_fmt_num(row['k_out'])} | {_fmt_num(row['v_out'])} | "
            f"{_fmt_num(row['k_mxmed'])} | {_fmt_num(row['v_mxmed'])} |"
        )
    return "\n".join(rows) + "\n"


def _markdown_section_rotation(rotation_df: pd.DataFrame) -> str:
    if rotation_df.empty:
        return "## Rotation (Beta fit + independence)\n\n_no data_\n"
    rows = [
        "## Rotation diagnostics",
        "",
        "| run | token_type | kind | mean KS | mean p | mean |corr| |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    agg = rotation_df.groupby(["run", "token_type", "kind"]).agg(
        mean_ks=("ks_statistic", "mean"),
        mean_p=("p_value", "mean"),
        mean_corr=("mean_abs_correlation", "mean"),
    )
    for (run, tt, knd), row in agg.iterrows():
        rows.append(
            f"| {run} | {tt} | {knd} | {_fmt_num(row['mean_ks'])} | "
            f"{_fmt_num(row['mean_p'])} | {_fmt_num(row['mean_corr'])} |"
        )
    return "\n".join(rows) + "\n"


def _markdown_section_attention(attn_df: pd.DataFrame) -> str:
    if attn_df.empty:
        return "## Attention analysis\n\n_no data_\n"
    rows = [
        "## Attention analysis (baseline K as Q probe)",
        "",
        "| run | token_type | mean KL | mean JS | top-1 match | top-5 Jaccard | entropy delta |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    agg = attn_df.groupby(["run", "token_type"]).agg(
        kl=("kl_divergence", "mean"),
        js=("js_divergence", "mean"),
        top1=("top1_match_rate", "mean"),
        topk=("topk_overlap_k5", "mean"),
        edelta=("entropy_delta", "mean"),
    )
    for (run, tt), row in agg.iterrows():
        rows.append(
            f"| {run} | {tt} | {_fmt_num(row['kl'])} | "
            f"{_fmt_num(row['js'])} | {_fmt_num(row['top1'])} | "
            f"{_fmt_num(row['topk'])} | {_fmt_num(row['edelta'])} |"
        )
    return "\n".join(rows) + "\n"


def _markdown_section_quant(quant_df: pd.DataFrame) -> str:
    if quant_df.empty:
        return "## Quantization errors\n\n_no data_\n"
    rows = [
        "## Quantization errors (vs baseline)",
        "",
        "| run | kind | token_type | mean MSE/coord | mean cos sim | mean rel err | mean IP bias |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    agg = quant_df.groupby(["run", "kind", "token_type"]).agg(
        mse=("per_coord_mse", "mean"),
        cos=("cosine_sim", "mean"),
        rel=("relative_error", "mean"),
        ip=("inner_product_bias", "mean"),
    )
    for (run, knd, tt), row in agg.iterrows():
        rows.append(
            f"| {run} | {knd} | {tt} | {_fmt_num(row['mse'])} | "
            f"{_fmt_num(row['cos'])} | {_fmt_num(row['rel'])} | "
            f"{_fmt_num(row['ip'])} |"
        )
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def generate_full_report(
    baseline_dump: KVDump | str | Path,
    quant_dumps: Mapping[str, KVDump | str | Path],
    output_dir: str | Path,
    *,
    outlier_threshold: float = 10.0,
    make_plots: bool = True,
    bits_by_run: Mapping[str, int | Mapping[str, int]] | None = None,
) -> dict[str, Path]:
    """Run all analyses and dump the results to ``output_dir``.

    Parameters
    ----------
    baseline_dump:
        Baseline KVDump (FP16) used as the ground truth for the diff.
    quant_dumps:
        Mapping ``run_name -> KVDump``.  Each run is compared against
        ``baseline_dump``.
    output_dir:
        Destination directory.  Created if missing.
    outlier_threshold:
        Forwarded to :func:`outlier_ratio_vision_vs_text`.
    make_plots:
        When ``True`` (default) renders PNG charts into
        ``output_dir/plots``.
    bits_by_run:
        Optional mapping of run name to bitwidth(s), used to add a
        ``theoretical_ratio`` column.  A single int applies to both K
        and V.

    Returns
    -------
    dict[str, Path]
        Map of artifact name -> absolute path.
    """
    output_dir = _ensure_dir(Path(output_dir))
    plots_dir = _ensure_dir(output_dir / "plots") if make_plots else None

    baseline, quant_map = _load_dumps(baseline_dump, quant_dumps)
    all_runs: dict[str, KVDump] = {"baseline": baseline, **quant_map}

    # -----------------------------------------------------------------
    # Distribution
    # -----------------------------------------------------------------
    dist_frames: list[pd.DataFrame] = []
    for name, dump in all_runs.items():
        df = compute_per_layer_stats(dump, separate_vision_text=True)
        dist_frames.append(_tag_run(df, name))
    dist_df = (
        pd.concat(dist_frames, ignore_index=True)
        if dist_frames and any(not f.empty for f in dist_frames)
        else pd.DataFrame()
    )
    dist_path = output_dir / "distribution_stats.csv"
    dist_df.to_csv(dist_path, index=False)

    # -----------------------------------------------------------------
    # Outliers
    # -----------------------------------------------------------------
    outlier_frames: list[pd.DataFrame] = []
    for name, dump in all_runs.items():
        df = outlier_ratio_vision_vs_text(dump, threshold=outlier_threshold)
        outlier_frames.append(_tag_run(df, name))
    outlier_df = (
        pd.concat(outlier_frames, ignore_index=True)
        if outlier_frames and any(not f.empty for f in outlier_frames)
        else pd.DataFrame()
    )
    outlier_path = output_dir / "outliers_per_layer.csv"
    outlier_df.to_csv(outlier_path, index=False)

    # -----------------------------------------------------------------
    # Quantization error
    # -----------------------------------------------------------------
    quant_frames: list[pd.DataFrame] = []
    theoretical_frames: list[pd.DataFrame] = []
    for name, dump in quant_map.items():
        df = compare_dumps(baseline, dump)
        quant_frames.append(_tag_run(df, name))
        if bits_by_run and name in bits_by_run:
            theo = summarize_against_theoretical(df, bits_by_kind=bits_by_run[name])
            theoretical_frames.append(_tag_run(theo, name))
    quant_df = (
        pd.concat(quant_frames, ignore_index=True)
        if quant_frames and any(not f.empty for f in quant_frames)
        else pd.DataFrame()
    )
    quant_path = output_dir / "quant_errors.csv"
    quant_df.to_csv(quant_path, index=False)

    theo_path = output_dir / "quant_theoretical_comparison.csv"
    theo_df = (
        pd.concat(theoretical_frames, ignore_index=True)
        if theoretical_frames and any(not f.empty for f in theoretical_frames)
        else pd.DataFrame()
    )
    theo_df.to_csv(theo_path, index=False)

    # -----------------------------------------------------------------
    # Attention analysis (K-as-Q-probe)
    # -----------------------------------------------------------------
    attn_frames: list[pd.DataFrame] = []
    for name, dump in quant_map.items():
        try:
            df = compare_attention_from_dumps(baseline, dump)
            attn_frames.append(_tag_run(df, name))
        except Exception:
            logger.exception("attention analysis failed for %s (non-fatal)", name)
    attn_df = (
        pd.concat(attn_frames, ignore_index=True)
        if attn_frames and any(not f.empty for f in attn_frames)
        else pd.DataFrame()
    )
    attn_path = output_dir / "attention_analysis.csv"
    attn_df.to_csv(attn_path, index=False)

    # -----------------------------------------------------------------
    # Rotation analysis
    # -----------------------------------------------------------------
    rotation_frames: list[pd.DataFrame] = []
    for name, dump in all_runs.items():
        df = vision_vs_text_rotation_analysis(dump)
        rotation_frames.append(_tag_run(df, name))
    rotation_df = (
        pd.concat(rotation_frames, ignore_index=True)
        if rotation_frames and any(not f.empty for f in rotation_frames)
        else pd.DataFrame()
    )
    rotation_path = output_dir / "rotation_analysis.csv"
    rotation_df.to_csv(rotation_path, index=False)

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------
    if plots_dir is not None:
        try:
            _plot_kv_norm_per_layer(dist_df, plots_dir / "kv_norm_per_layer.png")
            _plot_outlier_ratio(
                outlier_df[outlier_df["run"] == "baseline"]
                if not outlier_df.empty
                else outlier_df,
                plots_dir / "outlier_ratio_baseline.png",
            )
            _plot_rotation_quality(
                rotation_df[rotation_df["run"] == "baseline"]
                if not rotation_df.empty
                else rotation_df,
                plots_dir / "rotation_quality_baseline.png",
            )
            _plot_quant_error_per_layer(quant_df, plots_dir / "quant_error_per_layer.png")
            _plot_attention_comparison(attn_df, plots_dir / "attention_comparison.png")

            # Layer-wise curve plots (norm ratio + distortion)
            plot_kv_norm_ratio_curve(dist_df, out_path=plots_dir / "kv_norm_ratio_curve.png")
            plot_layer_distortion_curves(quant_df, out_path=plots_dir / "layer_distortion_curves.png")
        except Exception:  # pragma: no cover - plot fallback
            logger.exception("plot generation failed (non-fatal)")

    # -----------------------------------------------------------------
    # Markdown summary
    # -----------------------------------------------------------------
    md_parts: list[str] = []
    md_parts.append(f"# KV cache analysis report\n")
    md_parts.append(f"Baseline dump: `{baseline.dump_dir}`\n")
    md_parts.append(f"Quantized runs: {', '.join(quant_map.keys()) or '_none_'}\n\n")
    md_parts.append(_markdown_section_distribution(dist_df))
    md_parts.append("\n")
    md_parts.append(_markdown_section_outliers(outlier_df))
    md_parts.append("\n")
    md_parts.append(_markdown_section_rotation(rotation_df))
    md_parts.append("\n")
    md_parts.append(_markdown_section_quant(quant_df))
    md_parts.append("\n")
    md_parts.append(_markdown_section_attention(attn_df))

    md_path = output_dir / "report.md"
    md_path.write_text("\n".join(md_parts), encoding="utf-8")

    return {
        "distribution_stats": dist_path,
        "outliers_per_layer": outlier_path,
        "quant_errors": quant_path,
        "quant_theoretical_comparison": theo_path,
        "rotation_analysis": rotation_path,
        "attention_analysis": attn_path,
        "report_md": md_path,
    }
