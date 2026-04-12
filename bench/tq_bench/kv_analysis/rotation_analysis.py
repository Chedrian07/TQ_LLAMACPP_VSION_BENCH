"""TurboQuant rotation-theory diagnostics.

TurboQuant's premise is that applying a random orthogonal rotation
(FWHT + deterministic sign flip) to a unit vector in ``R^d`` makes the
marginal distribution of every coordinate look like a
``Beta((d-1)/2, (d-1)/2)`` random variable on ``[-1, 1]`` and makes the
coordinates close to independent.  The Lloyd-Max codebook used by the
C++ implementation is optimal under exactly this assumption.

This module measures how well those assumptions actually hold on real
KV cache data:

1. :func:`apply_fwht` — the same Walsh-Hadamard transform + sign flip
   used by the CPU reference (seed=42).
2. :func:`beta_distribution_fit_test` — Kolmogorov-Smirnov test versus
   the theoretical ``Beta((d-1)/2, (d-1)/2)`` marginal (transformed to
   ``[0, 1]``).
3. :func:`coordinate_independence_test` — off-diagonal correlation
   norm of the rotated coordinates.
4. :func:`analyze_rotation_per_layer` — apply 1-3 to every layer.
5. :func:`vision_vs_text_rotation_analysis` — the core research
   metric: are rotated vision tokens "more Beta-like" than text?

The coordinate convention matches the C++ reference: the input vector
is normalised to unit length, the sign flip is applied, then the FWHT
is taken in place (without any ``1/sqrt(n)`` scaling).  The resulting
buffer has elements roughly on ``[-1, 1]``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from .loader import KVDump


# ---------------------------------------------------------------------------
# Constants matching the C++ reference
# ---------------------------------------------------------------------------

#: Seed used by the reference CPU quantizer (see ``TURBO_SEED`` in
#: ``ggml-quants.c``).  Changing this will break round-trip parity.
TURBO_SEED: int = 42

#: Native dim used by all Qwen3-VL TurboQuant types.
TURBO_DIM: int = 128


# ---------------------------------------------------------------------------
# Reference transforms
# ---------------------------------------------------------------------------


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def sign_flip_mask(n: int, seed: int = TURBO_SEED) -> np.ndarray:
    """Deterministic ±1 mask matching ``turbo_sign_flip``.

    The C++ reference uses::

        h = seed * 2654435761u + i * 2246822519u;
        if (h >> 31) { x[i] = -x[i]; }

    which is a 32-bit unsigned multiplication followed by a top-bit
    test.  We reproduce that in numpy using explicit uint32 math.
    """
    if n <= 0:
        return np.ones(0, dtype=np.int8)
    idx = np.arange(n, dtype=np.uint64)
    h = (np.uint64(seed) * np.uint64(2654435761)
         + idx * np.uint64(2246822519))
    h32 = (h & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    flip = (h32 >> np.uint32(31)).astype(bool)
    mask = np.where(flip, -1, 1).astype(np.int8)
    return mask


def _fwht_inplace_matrix(x: np.ndarray) -> np.ndarray:
    """In-place Walsh-Hadamard transform along the last axis.

    ``x`` must have shape ``(..., n)`` with ``n`` a power of two.  This
    mirrors the C++ reference and does NOT apply the ``1/sqrt(n)``
    scaling (that happens in the inverse path).
    """
    n = x.shape[-1]
    if not _is_power_of_two(n):
        raise ValueError(f"FWHT length must be power of two, got {n}")
    length = 1
    while length < n:
        # Reshape to ``(..., n/(2L), 2, L)`` so we can operate on the
        # two interleaved halves simultaneously.
        new_shape = x.shape[:-1] + (n // (2 * length), 2, length)
        view = x.reshape(new_shape)
        u = view[..., 0, :].copy()
        v = view[..., 1, :].copy()
        view[..., 0, :] = u + v
        view[..., 1, :] = u - v
        x = view.reshape(x.shape[:-1] + (n,))
        length <<= 1
    return x


def apply_fwht(
    x: np.ndarray,
    *,
    normalize: bool = True,
    seed: int = TURBO_SEED,
) -> np.ndarray:
    """Apply unit-normalise -> sign-flip -> FWHT to each row of ``x``.

    Parameters
    ----------
    x:
        Either a single vector of length ``d`` or a 2-D array of shape
        ``(n, d)``.  ``d`` must be a power of two.
    normalize:
        When ``True`` each row is L2-normalised before the transform,
        matching :c:func:`turbo_forward_rotate`.  Pass ``False`` if the
        caller has already normalised.
    seed:
        Sign-flip seed (default :data:`TURBO_SEED`).

    Returns
    -------
    np.ndarray
        A float64 copy of ``x`` with the rotation applied.  The shape
        matches the input.
    """
    orig = np.asarray(x)
    if orig.ndim == 1:
        batched = orig.reshape(1, -1)
        squeeze = True
    elif orig.ndim == 2:
        batched = orig
        squeeze = False
    else:
        raise ValueError(f"expected 1-D or 2-D array, got shape {orig.shape}")

    n = batched.shape[-1]
    if not _is_power_of_two(n):
        raise ValueError(f"FWHT length must be power of two, got {n}")

    out = batched.astype(np.float64, copy=True)

    if normalize:
        norms = np.linalg.norm(out, axis=-1, keepdims=True)
        safe = norms > 1e-30
        np.divide(out, norms, out=out, where=safe)
        out[~safe.squeeze(-1)] = 0.0

    mask = sign_flip_mask(n, seed=seed).astype(np.float64)
    out *= mask[np.newaxis, :]

    out = _fwht_inplace_matrix(out)

    if squeeze:
        return out.reshape(-1)
    return out


def fwht_round_trip(x: np.ndarray, seed: int = TURBO_SEED) -> np.ndarray:
    """Debug helper: apply FWHT forward and inverse to verify parity."""
    forward = apply_fwht(x, normalize=False, seed=seed)
    # FWHT is its own inverse up to scale factor n.
    inverse = _fwht_inplace_matrix(forward.copy())
    n = forward.shape[-1]
    inverse = inverse / n
    mask = sign_flip_mask(n, seed=seed).astype(np.float64)
    if inverse.ndim == 1:
        inverse *= mask
    else:
        inverse *= mask[np.newaxis, :]
    return inverse


# ---------------------------------------------------------------------------
# Beta distribution fit test
# ---------------------------------------------------------------------------


def _beta_params(n_dim: int) -> tuple[float, float]:
    alpha = (n_dim - 1) / 2.0
    return alpha, alpha


def beta_distribution_fit_test(
    rotated_coords: np.ndarray,
    *,
    n_dim: int = TURBO_DIM,
    scale: float | None = None,
) -> dict[str, float | str]:
    """KS test of rotated coordinates vs ``Beta((d-1)/2, (d-1)/2)``.

    The theoretical marginal of a coordinate of a random unit vector
    in ``R^d`` lives on ``[-1, 1]`` with density proportional to
    ``(1 - x^2)^((d-3)/2)``.  This is a Beta density after the affine
    change of variables ``u = (x + 1)/2``.

    Because the C++ forward rotation does **not** divide by
    ``sqrt(n)`` (it only divides on the inverse path), the raw FWHT
    output has coordinates on ``[-sqrt(n), sqrt(n)]`` if fed a unit
    vector.  We therefore rescale by ``1/sqrt(n_dim)`` before the fit
    unless the caller provides an explicit ``scale``.

    Parameters
    ----------
    rotated_coords:
        Flattened 1-D array of rotated coordinate samples.
    n_dim:
        Dimensionality used to derive the Beta shape parameters.
    scale:
        Optional explicit rescale factor.  Defaults to
        ``1 / sqrt(n_dim)``.

    Returns
    -------
    dict
        ``{'ks_statistic', 'p_value', 'alpha', 'beta', 'n', 'fit_quality'}``
        where ``fit_quality`` is a coarse label in
        ``{'excellent', 'good', 'poor', 'fail'}``.
    """
    flat = np.asarray(rotated_coords, dtype=np.float64).ravel()
    if flat.size < 10:
        return {
            "ks_statistic": float("nan"),
            "p_value": float("nan"),
            "alpha": float("nan"),
            "beta": float("nan"),
            "n": int(flat.size),
            "fit_quality": "insufficient",
        }

    scale_factor = scale if scale is not None else (1.0 / np.sqrt(n_dim))
    scaled = flat * scale_factor
    # Clip strictly inside (-1, 1) so the KS test does not blow up on
    # boundary samples.
    eps = 1e-9
    scaled = np.clip(scaled, -1.0 + eps, 1.0 - eps)
    u = (scaled + 1.0) / 2.0

    alpha, beta = _beta_params(n_dim)

    ks_stat, p_value = stats.kstest(u, stats.beta(alpha, beta).cdf)
    ks_stat = float(ks_stat)
    p_value = float(p_value)

    if p_value > 0.1 or ks_stat < 0.02:
        quality = "excellent"
    elif p_value > 0.01 or ks_stat < 0.05:
        quality = "good"
    elif ks_stat < 0.10:
        quality = "poor"
    else:
        quality = "fail"

    return {
        "ks_statistic": ks_stat,
        "p_value": p_value,
        "alpha": float(alpha),
        "beta": float(beta),
        "n": int(flat.size),
        "fit_quality": quality,
    }


# ---------------------------------------------------------------------------
# Coordinate independence
# ---------------------------------------------------------------------------


def coordinate_independence_test(
    rotated_coords: np.ndarray,
) -> dict[str, float | int]:
    """Compute summary statistics of the pairwise correlation matrix.

    Parameters
    ----------
    rotated_coords:
        2-D array of shape ``(n_samples, n_dim)``.  Each row is one
        rotated vector.

    Returns
    -------
    dict
        ``mean_abs_correlation``, ``max_correlation``,
        ``off_diagonal_norm`` (Frobenius norm of the off-diagonal
        correlation matrix divided by ``n * (n - 1)``), plus
        ``n_samples`` and ``n_dim``.
    """
    arr = np.asarray(rotated_coords, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {arr.shape}")
    n_samples, n_dim = arr.shape
    if n_samples < 2 or n_dim < 2:
        return {
            "mean_abs_correlation": float("nan"),
            "max_correlation": float("nan"),
            "off_diagonal_norm": float("nan"),
            "n_samples": n_samples,
            "n_dim": n_dim,
        }

    # np.corrcoef expects variables in rows (each row = one variable).
    # We transpose so that each column of ``arr`` becomes one variable.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(arr, rowvar=False)
    # Replace NaN (constant coordinates) with zero so they do not
    # dominate the summary.
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    off_diag = corr - np.diag(np.diag(corr))
    abs_off = np.abs(off_diag)
    n_off = n_dim * (n_dim - 1)
    mean_abs = float(abs_off.sum() / n_off) if n_off else float("nan")
    max_corr = float(abs_off.max()) if n_off else float("nan")
    fro_off = float(np.linalg.norm(off_diag) / np.sqrt(n_off)) if n_off else float("nan")

    return {
        "mean_abs_correlation": mean_abs,
        "max_correlation": max_corr,
        "off_diagonal_norm": fro_off,
        "n_samples": int(n_samples),
        "n_dim": int(n_dim),
    }


# ---------------------------------------------------------------------------
# Per-layer analysis
# ---------------------------------------------------------------------------


def _rotate_all_head_vectors(
    tensor: np.ndarray,
    *,
    seed: int = TURBO_SEED,
) -> np.ndarray:
    """Reshape a KV tensor to (T*H, D) and run :func:`apply_fwht`.

    The input is expected to be shape ``(T, H, D)``.  D must be a
    power of two (matches ``head_dim`` for Qwen3-VL).  Returns a 2-D
    array of shape ``(T*H, D)`` holding the rotated vectors.
    """
    if tensor.ndim != 3:
        raise ValueError(f"expected (T, H, D) tensor, got {tensor.shape}")
    T, H, D = tensor.shape
    flat = tensor.reshape(T * H, D)
    return apply_fwht(flat, normalize=True, seed=seed)


def analyze_rotation_per_layer(
    dump: KVDump,
    *,
    token_type: str = "all",
    kind: Literal["K", "V", "both"] = "both",
    seed: int = TURBO_SEED,
) -> pd.DataFrame:
    """Per-layer TurboQuant rotation diagnostics.

    Parameters
    ----------
    dump:
        Loaded :class:`KVDump`.
    token_type:
        ``'all'`` | ``'vision'`` | ``'text'``.  Filters the token axis.
    kind:
        Which tensor to analyse.  ``'both'`` produces one row per
        ``(layer, K/V)`` pair.
    seed:
        Sign-flip seed, defaults to :data:`TURBO_SEED`.

    Returns
    -------
    pd.DataFrame
        Columns: ``layer``, ``token_type``, ``kind``, ``n_vectors``,
        ``ks_statistic``, ``p_value``, ``fit_quality``,
        ``mean_abs_correlation``, ``max_correlation``,
        ``off_diagonal_norm``.
    """
    if kind not in ("K", "V", "both"):
        raise ValueError(f"kind must be K|V|both, got {kind!r}")

    if token_type == "all":
        mask = None
    elif token_type == "vision":
        mask = dump.vision_mask()
        if not mask.any():
            return pd.DataFrame()
    elif token_type == "text":
        mask = dump.text_mask()
        if not mask.any():
            return pd.DataFrame()
    else:
        raise ValueError(f"token_type must be all|vision|text, got {token_type!r}")

    n_dim = dump.head_dim
    kinds: list[str] = ["K", "V"] if kind == "both" else [kind]

    rows: list[dict[str, float | int | str]] = []
    for layer in range(dump.n_layers):
        tensors: dict[str, np.ndarray] = {}
        if "K" in kinds:
            tensors["K"] = dump.get_K(layer)
        if "V" in kinds:
            tensors["V"] = dump.get_V(layer)

        for knd, t in tensors.items():
            sub = t if mask is None else t[mask]
            if sub.size == 0:
                continue
            rotated = _rotate_all_head_vectors(sub, seed=seed)
            beta_res = beta_distribution_fit_test(rotated.ravel(), n_dim=n_dim)
            indep_res = coordinate_independence_test(rotated)

            rows.append({
                "layer": int(layer),
                "token_type": token_type,
                "kind": knd,
                "n_vectors": int(rotated.shape[0]),
                "ks_statistic": beta_res["ks_statistic"],
                "p_value": beta_res["p_value"],
                "fit_quality": beta_res["fit_quality"],
                "mean_abs_correlation": indep_res["mean_abs_correlation"],
                "max_correlation": indep_res["max_correlation"],
                "off_diagonal_norm": indep_res["off_diagonal_norm"],
            })

    return pd.DataFrame(rows)


def vision_vs_text_rotation_analysis(
    dump: KVDump,
    *,
    kind: Literal["K", "V", "both"] = "both",
    seed: int = TURBO_SEED,
) -> pd.DataFrame:
    """Core research metric: rotated-coord stats for vision vs text.

    Returns a long-format DataFrame with rows for ``all``, ``vision``
    and ``text`` (whichever are present).  Intended to be pivoted by
    the analysis notebook.
    """
    frames: list[pd.DataFrame] = []
    for tt in ("all", "vision", "text"):
        if tt == "vision" and not dump.has_vision():
            continue
        if tt == "text" and not dump.has_text():
            continue
        frame = analyze_rotation_per_layer(dump, token_type=tt, kind=kind, seed=seed)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
