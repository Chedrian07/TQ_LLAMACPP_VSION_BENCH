"""Quantization-error metrics measured on real KV dumps.

The C++ tool dumps the FP16 (upcast to FP32) KV cache for both a
``baseline`` run and one or more quantized runs.  This module compares
those dumps directly:

* :func:`compare_tensors` — raw numpy primitive
* :func:`compare_dumps` — per-layer diff between two :class:`KVDump`
* :func:`compare_with_theoretical` — check against the paper's
  theoretical MSE per rotated coordinate

Theoretical MSE numbers from the TurboQuant paper (for 128-dim rotated
unit vectors following ``Beta((d-1)/2, (d-1)/2)``, 2/3/4-bit Lloyd-Max):

* ``turbo2``: ~0.117 per coord
* ``turbo3``: ~0.034 per coord
* ``turbo4``: ~0.009 per coord

``compare_with_theoretical`` uses these numbers as a sanity check and
returns the ratio of measured to theoretical.
"""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import pandas as pd

from .loader import KVDump


# ---------------------------------------------------------------------------
# Theoretical Lloyd-Max MSE (paper values, per rotated coord)
# ---------------------------------------------------------------------------

#: Maps bitwidth (int) -> paper MSE/coord for a Beta((d-1)/2,(d-1)/2)
#: distribution rotated from a unit vector.  Values are from the
#: TurboQuant paper (arXiv 2504.19874) Table 1 / Eq. 16.
THEORETICAL_MSE_PER_COORD: dict[int, float] = {
    2: 0.117,
    3: 0.034,
    4: 0.009,
}


def _as_tokens_features(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3:
        T, H, D = tensor.shape
        return tensor.reshape(T, H * D)
    raise ValueError(f"expected 2-D or 3-D tensor, got shape {tensor.shape}")


# ---------------------------------------------------------------------------
# Elementary tensor comparison
# ---------------------------------------------------------------------------


def compare_tensors(
    baseline: np.ndarray,
    quantized: np.ndarray,
    *,
    random_probe: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Return per-tensor error metrics.

    Metrics returned:

    * ``per_coord_mse`` – ``mean((a - b)^2)`` across all elements.
    * ``max_abs_error`` – worst-case absolute error.
    * ``cosine_sim`` – mean over tokens of ``cos(a_token, b_token)``.
    * ``inner_product_bias`` – ``E[<y, b>] / E[<y, a>]`` where
      ``y`` is a probe vector drawn from ``N(0, I)`` (rows).  Values
      near 1.0 indicate an unbiased estimator.
    * ``relative_error`` – ``||a - b||_F / ||a||_F`` (Frobenius).
    * ``mean_baseline_norm`` – average per-token L2 norm of
      ``baseline`` (useful to contextualise the error).
    """
    if baseline.shape != quantized.shape:
        raise ValueError(
            f"shape mismatch: baseline {baseline.shape} vs "
            f"quantized {quantized.shape}"
        )
    if baseline.size == 0:
        return {
            "per_coord_mse": float("nan"),
            "max_abs_error": float("nan"),
            "cosine_sim": float("nan"),
            "inner_product_bias": float("nan"),
            "relative_error": float("nan"),
            "mean_baseline_norm": float("nan"),
        }

    a = _as_tokens_features(baseline).astype(np.float64, copy=False)
    b = _as_tokens_features(quantized).astype(np.float64, copy=False)

    diff = a - b
    per_coord_mse = float(np.mean(diff * diff))
    max_abs = float(np.max(np.abs(diff)))

    a_norms = np.linalg.norm(a, axis=1)
    b_norms = np.linalg.norm(b, axis=1)
    mean_baseline_norm = float(a_norms.mean())
    denom = a_norms * b_norms
    cos = np.zeros_like(a_norms)
    valid = denom > 0
    if valid.any():
        cos[valid] = np.einsum("tf,tf->t", a[valid], b[valid]) / denom[valid]
    cos_mean = float(cos[valid].mean()) if valid.any() else float("nan")

    # Relative Frobenius error
    base_fro = float(np.linalg.norm(a))
    rel_error = float(np.linalg.norm(diff) / base_fro) if base_fro > 0 else float("nan")

    # Inner-product bias using a random probe vector (or caller-provided)
    T, F = a.shape
    if random_probe is None:
        rng_ = rng or np.random.default_rng(42)
        probe = rng_.standard_normal(F).astype(np.float64)
    else:
        probe = random_probe.astype(np.float64, copy=False).ravel()
        if probe.shape != (F,):
            raise ValueError(
                f"random_probe length {probe.shape} does not match F={F}"
            )
    ip_baseline = float(np.mean(a @ probe))
    ip_quant = float(np.mean(b @ probe))
    if abs(ip_baseline) > 1e-30:
        ip_bias = ip_quant / ip_baseline
    else:
        ip_bias = float("nan")

    return {
        "per_coord_mse": per_coord_mse,
        "max_abs_error": max_abs,
        "cosine_sim": cos_mean,
        "inner_product_bias": ip_bias,
        "relative_error": rel_error,
        "mean_baseline_norm": mean_baseline_norm,
    }


# ---------------------------------------------------------------------------
# Dump-level comparison
# ---------------------------------------------------------------------------


def _require_compatible(a: KVDump, b: KVDump) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"KV shape mismatch: baseline {a.shape} vs quantized {b.shape}"
        )


def compare_dumps(
    baseline: KVDump,
    quantized: KVDump,
    *,
    separate_vision_text: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-layer diff between two dumps.

    Returns one row per ``(layer, token_type, kind)`` where ``kind`` is
    ``'K'`` or ``'V'``.  The random probe used for
    ``inner_product_bias`` is drawn once at the start and reused for
    every layer so the columns are directly comparable.
    """
    _require_compatible(baseline, quantized)

    rng = np.random.default_rng(seed)
    probe_len = baseline.n_kv_head * baseline.head_dim
    probe = rng.standard_normal(probe_len).astype(np.float64)

    token_types: list[str] = ["all"]
    if separate_vision_text:
        if baseline.has_vision() and quantized.has_vision():
            token_types.append("vision")
        if baseline.has_text() and quantized.has_text():
            token_types.append("text")

    rows: list[dict[str, float | int | str]] = []
    for layer in range(baseline.n_layers):
        Kb = baseline.get_K(layer)
        Vb = baseline.get_V(layer)
        Kq = quantized.get_K(layer)
        Vq = quantized.get_V(layer)

        for tt in token_types:
            if tt == "all":
                mask = None
            elif tt == "vision":
                mask = baseline.vision_mask()
                if not mask.any():
                    continue
            else:  # text
                mask = baseline.text_mask()
                if not mask.any():
                    continue

            Kb_s = Kb if mask is None else Kb[mask]
            Vb_s = Vb if mask is None else Vb[mask]
            Kq_s = Kq if mask is None else Kq[mask]
            Vq_s = Vq if mask is None else Vq[mask]

            k_metrics = compare_tensors(Kb_s, Kq_s, random_probe=probe)
            v_metrics = compare_tensors(Vb_s, Vq_s, random_probe=probe)

            rows.append({"layer": layer, "token_type": tt, "kind": "K", **k_metrics})
            rows.append({"layer": layer, "token_type": tt, "kind": "V", **v_metrics})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Theoretical comparison
# ---------------------------------------------------------------------------


def compare_with_theoretical(
    measured_mse: float,
    bits: int,
    *,
    tol_ratio: float = 1.1,
) -> dict[str, float | bool]:
    """Compare measured MSE/coord against the paper's value.

    Notes
    -----
    The paper's MSE is measured on a *unit* rotated vector.  If the
    caller passes raw (non-normalised) MSE, the returned ``ratio``
    should be divided by ``mean_norm^2`` externally to get a fair
    comparison.  This function makes no assumption about scale — it
    just returns the ratio and a boolean flag.
    """
    if bits not in THEORETICAL_MSE_PER_COORD:
        raise ValueError(
            f"no theoretical MSE for {bits}-bit; "
            f"known: {sorted(THEORETICAL_MSE_PER_COORD)}"
        )
    theoretical = THEORETICAL_MSE_PER_COORD[bits]
    ratio = measured_mse / theoretical if theoretical > 0 else float("inf")
    return {
        "bits": bits,
        "measured_mse_per_coord": float(measured_mse),
        "theoretical_mse_per_coord": float(theoretical),
        "ratio": float(ratio),
        "within_tolerance": bool(1.0 / tol_ratio <= ratio <= tol_ratio),
    }


def summarize_against_theoretical(
    diff_frame: pd.DataFrame,
    *,
    bits_by_kind: Mapping[Literal["K", "V"], int] | int,
) -> pd.DataFrame:
    """Compare a diff frame (from :func:`compare_dumps`) against theory.

    ``bits_by_kind`` can be a single int (same bits on both K and V)
    or a mapping ``{"K": 4, "V": 2}``.  The result is restricted to
    rows where ``token_type == 'all'`` so the ratio is meaningful.
    """
    if diff_frame.empty:
        return pd.DataFrame()
    if isinstance(bits_by_kind, int):
        bits_map = {"K": bits_by_kind, "V": bits_by_kind}
    else:
        bits_map = dict(bits_by_kind)

    sub = diff_frame[diff_frame["token_type"] == "all"].copy()
    rows: list[dict[str, float | int | str]] = []
    for _, row in sub.iterrows():
        kind = row["kind"]
        bits = bits_map.get(kind)
        if bits is None or bits not in THEORETICAL_MSE_PER_COORD:
            continue
        cmp = compare_with_theoretical(float(row["per_coord_mse"]), bits)
        rows.append({
            "layer": int(row["layer"]),
            "kind": kind,
            "bits": bits,
            "measured_mse_per_coord": cmp["measured_mse_per_coord"],
            "theoretical_mse_per_coord": cmp["theoretical_mse_per_coord"],
            "ratio": cmp["ratio"],
            "within_tolerance": cmp["within_tolerance"],
        })
    return pd.DataFrame(rows)
