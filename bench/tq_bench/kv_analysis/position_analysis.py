"""Position-aware KV cache analysis.

This module examines how KV cache statistics vary across the *token
position* axis — the dimension that distinguishes vision tokens from
text tokens.  It answers questions like:

* Do vision tokens carry systematically different L2 norms?
* Do outlier channels concentrate in vision vs. text positions?
* Does quantization error vary by position (and token type)?

All functions operate on :class:`~tq_bench.kv_analysis.loader.KVDump`
objects and return either numpy arrays or pandas DataFrames.
Plotting helpers write 150 DPI PNGs and close figures after saving.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from .loader import KVDump


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_tensor(dump: KVDump, kind: str, layer: int) -> np.ndarray:
    """Return (T, H, D) tensor for ``kind`` ('K' or 'V')."""
    if kind == "K":
        return dump.get_K(layer)
    if kind == "V":
        return dump.get_V(layer)
    raise ValueError(f"kind must be 'K' or 'V', got {kind!r}")


def _flatten_to_features(tensor: np.ndarray) -> np.ndarray:
    """Reshape (T, H, D) -> (T, H*D)."""
    if tensor.ndim == 3:
        T, H, D = tensor.shape
        return tensor.reshape(T, H * D)
    if tensor.ndim == 2:
        return tensor
    raise ValueError(f"expected 2-D or 3-D tensor, got shape {tensor.shape}")


def _resolve_layers(dump: KVDump, layers: Sequence[int] | None) -> list[int]:
    """Return explicit layer list, defaulting to all layers."""
    if layers is not None:
        return list(layers)
    return list(range(dump.n_layers))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 1. Per-token norms
# ---------------------------------------------------------------------------


def compute_per_token_norms(
    dump: KVDump,
    kind: str = "K",
    layer: int = 0,
) -> np.ndarray:
    """L2 norm of each token's ``(H*D,)`` vector for one layer.

    Parameters
    ----------
    dump:
        Loaded KV dump.
    kind:
        ``'K'`` or ``'V'``.
    layer:
        Layer index.

    Returns
    -------
    np.ndarray
        Shape ``(n_tokens,)`` of float64 norms.
    """
    tensor = _get_tensor(dump, kind, layer)
    flat = _flatten_to_features(tensor).astype(np.float64, copy=False)
    return np.linalg.norm(flat, axis=1)


# ---------------------------------------------------------------------------
# 2. Token-index-vs-norm DataFrame
# ---------------------------------------------------------------------------


def token_index_vs_norm_df(
    dump: KVDump,
    kind: str = "K",
) -> pd.DataFrame:
    """DataFrame with ``(layer, token_idx, norm, is_vision)`` for all layers.

    Parameters
    ----------
    dump:
        Loaded KV dump.
    kind:
        ``'K'`` or ``'V'``.

    Returns
    -------
    pd.DataFrame
        Columns: ``layer``, ``token_idx``, ``norm``, ``is_vision``.
    """
    vision_mask = dump.vision_mask()
    rows: list[dict[str, int | float | bool]] = []

    for layer in range(dump.n_layers):
        norms = compute_per_token_norms(dump, kind=kind, layer=layer)
        for t_idx in range(dump.n_tokens):
            rows.append({
                "layer": layer,
                "token_idx": t_idx,
                "norm": float(norms[t_idx]),
                "is_vision": bool(vision_mask[t_idx]),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Plot token norm vs position
# ---------------------------------------------------------------------------


def plot_token_norm_vs_position(
    dump: KVDump,
    kind: str = "K",
    layers: Sequence[int] | None = None,
    out_dir: str | Path | None = None,
) -> list[Path]:
    """Line plot of L2 norm vs token index, vision tokens coloured differently.

    Parameters
    ----------
    dump:
        Loaded KV dump.
    kind:
        ``'K'`` or ``'V'``.
    layers:
        Layer indices to plot.  Defaults to all layers.
    out_dir:
        Directory for PNGs.  If ``None``, defaults to
        ``dump.dump_dir / 'plots' / 'position_norm'``.

    Returns
    -------
    list[Path]
        Paths to the saved PNG files.
    """
    resolved_layers = _resolve_layers(dump, layers)
    if out_dir is None:
        out_dir = dump.dump_dir / "plots" / "position_norm"
    out_dir = _ensure_dir(Path(out_dir))

    vision_mask = dump.vision_mask()
    has_vision = bool(vision_mask.any())
    token_indices = np.arange(dump.n_tokens)
    saved: list[Path] = []

    for layer in resolved_layers:
        norms = compute_per_token_norms(dump, kind=kind, layer=layer)

        fig, ax = plt.subplots(figsize=(10, 4))

        if has_vision:
            vis_idx = token_indices[vision_mask]
            txt_idx = token_indices[~vision_mask]
            if vis_idx.size > 0:
                ax.plot(
                    vis_idx, norms[vision_mask],
                    ".", markersize=3, color="tab:orange", label="vision", alpha=0.8,
                )
            if txt_idx.size > 0:
                ax.plot(
                    txt_idx, norms[~vision_mask],
                    ".", markersize=3, color="tab:blue", label="text", alpha=0.8,
                )
            ax.legend(fontsize=8)
        else:
            ax.plot(
                token_indices, norms,
                ".", markersize=3, color="tab:blue", alpha=0.8,
            )

        ax.set_title(f"{kind} token L2 norm vs position (layer {layer})")
        ax.set_xlabel("token index")
        ax.set_ylabel("L2 norm")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = out_dir / f"{kind}_layer_{layer}_norm_vs_pos.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# 4. Per-position outlier ratio
# ---------------------------------------------------------------------------


def per_position_outlier_ratio(
    dump: KVDump,
    kind: str = "K",
    threshold: float = 10.0,
) -> pd.DataFrame:
    """Per-token outlier ratio across all layers.

    For each token at each layer, compute how many of its ``H*D``
    channel values exceed ``threshold * median_channel_value`` where
    the median is taken across *all* tokens for that layer/channel.

    Parameters
    ----------
    dump:
        Loaded KV dump.
    kind:
        ``'K'`` or ``'V'``.
    threshold:
        Channel-value magnitude threshold relative to the
        per-channel median absolute value across tokens.

    Returns
    -------
    pd.DataFrame
        Columns: ``layer``, ``token_idx``, ``is_vision``, ``outlier_ratio``.
    """
    vision_mask = dump.vision_mask()
    rows: list[dict[str, int | float | bool]] = []

    for layer in range(dump.n_layers):
        tensor = _get_tensor(dump, kind, layer)
        flat = _flatten_to_features(tensor).astype(np.float64, copy=False)
        # flat shape: (T, F)
        T, F = flat.shape

        if T == 0 or F == 0:
            continue

        # Median absolute value per channel across all tokens.
        channel_medians = np.median(np.abs(flat), axis=0)  # (F,)

        for t_idx in range(T):
            token_vals = np.abs(flat[t_idx])  # (F,)
            # A channel is an outlier for this token if its absolute value
            # exceeds threshold * the median absolute value of that channel.
            safe_medians = np.where(channel_medians > 0, channel_medians, 1.0)
            outlier_count = int(np.sum(token_vals > threshold * safe_medians))
            rows.append({
                "layer": layer,
                "token_idx": t_idx,
                "is_vision": bool(vision_mask[t_idx]),
                "outlier_ratio": outlier_count / F,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Plot position outlier heatmap
# ---------------------------------------------------------------------------


def plot_position_outlier_heatmap(
    dump: KVDump,
    kind: str = "K",
    threshold: float = 10.0,
    out_path: str | Path | None = None,
) -> Path:
    """Heatmap of outlier ratio: x=token_idx, y=layer.

    Parameters
    ----------
    dump:
        Loaded KV dump.
    kind:
        ``'K'`` or ``'V'``.
    threshold:
        Channel outlier threshold.
    out_path:
        Output PNG path.  Defaults to
        ``dump.dump_dir / 'plots' / '{kind}_outlier_heatmap.png'``.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    df = per_position_outlier_ratio(dump, kind=kind, threshold=threshold)

    if out_path is None:
        out_dir = _ensure_dir(dump.dump_dir / "plots")
        out_path = out_dir / f"{kind}_outlier_heatmap.png"
    else:
        out_path = Path(out_path)
        _ensure_dir(out_path.parent)

    n_layers = dump.n_layers
    n_tokens = dump.n_tokens

    # Build matrix: rows=layers, cols=tokens.
    heatmap = np.zeros((n_layers, n_tokens), dtype=np.float64)
    if not df.empty:
        for _, row in df.iterrows():
            heatmap[int(row["layer"]), int(row["token_idx"])] = float(row["outlier_ratio"])

    fig, ax = plt.subplots(figsize=(max(8, n_tokens * 0.15), max(4, n_layers * 0.4)))
    im = ax.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="YlOrRd",
    )
    fig.colorbar(im, ax=ax, label="outlier ratio")

    # Mark vision/text boundary if applicable.
    vision_mask = dump.vision_mask()
    if vision_mask.any() and not vision_mask.all():
        # Find the last vision token index.
        vision_indices = np.where(vision_mask)[0]
        boundary = float(vision_indices.max()) + 0.5
        ax.axvline(boundary, color="cyan", linewidth=1.5, linestyle="--", label="vision/text")
        ax.legend(fontsize=8, loc="upper right")

    ax.set_title(f"{kind} per-position outlier ratio (threshold={threshold}x)")
    ax.set_xlabel("token index")
    ax.set_ylabel("layer")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 6. Per-position quantization error
# ---------------------------------------------------------------------------


def per_position_quant_error(
    baseline_dump: KVDump,
    quant_dump: KVDump,
    kind: str = "K",
) -> pd.DataFrame:
    """Per-token quantization error between baseline and quantized dumps.

    Parameters
    ----------
    baseline_dump:
        Reference (FP16) KV dump.
    quant_dump:
        Quantized KV dump.
    kind:
        ``'K'`` or ``'V'``.

    Returns
    -------
    pd.DataFrame
        Columns: ``layer``, ``token_idx``, ``is_vision``,
        ``per_coord_mse``, ``cosine_sim``.
    """
    if baseline_dump.shape != quant_dump.shape:
        raise ValueError(
            f"KV shape mismatch: baseline {baseline_dump.shape} "
            f"vs quantized {quant_dump.shape}"
        )

    vision_mask = baseline_dump.vision_mask()
    rows: list[dict[str, int | float | bool]] = []

    for layer in range(baseline_dump.n_layers):
        base_tensor = _get_tensor(baseline_dump, kind, layer)
        quant_tensor = _get_tensor(quant_dump, kind, layer)

        base_flat = _flatten_to_features(base_tensor).astype(np.float64, copy=False)
        quant_flat = _flatten_to_features(quant_tensor).astype(np.float64, copy=False)
        # (T, F)

        T, F = base_flat.shape
        if T == 0 or F == 0:
            continue

        diff = base_flat - quant_flat
        per_token_mse = np.mean(diff * diff, axis=1)  # (T,)

        # Per-token cosine similarity.
        base_norms = np.linalg.norm(base_flat, axis=1)
        quant_norms = np.linalg.norm(quant_flat, axis=1)
        denom = base_norms * quant_norms
        cosine = np.zeros(T, dtype=np.float64)
        valid = denom > 0
        if valid.any():
            cosine[valid] = np.einsum(
                "tf,tf->t", base_flat[valid], quant_flat[valid]
            ) / denom[valid]

        for t_idx in range(T):
            rows.append({
                "layer": layer,
                "token_idx": t_idx,
                "is_vision": bool(vision_mask[t_idx]),
                "per_coord_mse": float(per_token_mse[t_idx]),
                "cosine_sim": float(cosine[t_idx]),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Plot per-position quantization error
# ---------------------------------------------------------------------------


def plot_position_quant_error(
    baseline_dump: KVDump,
    quant_dump: KVDump,
    kind: str = "K",
    layers: Sequence[int] | None = None,
    out_dir: str | Path | None = None,
) -> list[Path]:
    """Line plot of per-token MSE vs position, vision/text coloured.

    Parameters
    ----------
    baseline_dump:
        Reference (FP16) KV dump.
    quant_dump:
        Quantized KV dump.
    kind:
        ``'K'`` or ``'V'``.
    layers:
        Layer indices to plot.  Defaults to all layers.
    out_dir:
        Directory for PNGs.  Defaults to
        ``baseline_dump.dump_dir / 'plots' / 'position_quant_error'``.

    Returns
    -------
    list[Path]
        Paths to the saved PNG files.
    """
    df = per_position_quant_error(baseline_dump, quant_dump, kind=kind)
    resolved_layers = _resolve_layers(baseline_dump, layers)

    if out_dir is None:
        out_dir = baseline_dump.dump_dir / "plots" / "position_quant_error"
    out_dir = _ensure_dir(Path(out_dir))

    vision_mask = baseline_dump.vision_mask()
    has_vision = bool(vision_mask.any())
    saved: list[Path] = []

    for layer in resolved_layers:
        layer_df = df[df["layer"] == layer]
        if layer_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 4))

        if has_vision:
            vis_df = layer_df[layer_df["is_vision"]]
            txt_df = layer_df[~layer_df["is_vision"]]
            if not vis_df.empty:
                ax.plot(
                    vis_df["token_idx"].values, vis_df["per_coord_mse"].values,
                    ".", markersize=3, color="tab:orange", label="vision", alpha=0.8,
                )
            if not txt_df.empty:
                ax.plot(
                    txt_df["token_idx"].values, txt_df["per_coord_mse"].values,
                    ".", markersize=3, color="tab:blue", label="text", alpha=0.8,
                )
            ax.legend(fontsize=8)
        else:
            ax.plot(
                layer_df["token_idx"].values, layer_df["per_coord_mse"].values,
                ".", markersize=3, color="tab:blue", alpha=0.8,
            )

        ax.set_title(f"{kind} per-token MSE vs position (layer {layer})")
        ax.set_xlabel("token index")
        ax.set_ylabel("MSE / coord")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = out_dir / f"{kind}_layer_{layer}_quant_error_vs_pos.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    return saved
