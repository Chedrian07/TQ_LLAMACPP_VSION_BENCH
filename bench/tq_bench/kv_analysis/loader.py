"""KV cache dump loader.

The C++ side writes a directory containing ``meta.json`` plus one
``K_layer_{i}.bin`` / ``V_layer_{i}.bin`` per layer.  Each binary file
stores a little-endian float32 array with shape
``(n_tokens, n_kv_head, head_dim)``.

:class:`KVDump` loads everything lazily (on first access) and caches
the per-layer arrays.  :class:`KVDumpWriter` is the mirror used by the
unit tests to produce synthetic dumps when the C++ tool is not yet
available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KVDumpError(RuntimeError):
    """Raised when a KV dump directory cannot be loaded."""


# ---------------------------------------------------------------------------
# KVDump
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KVShape:
    n_tokens: int
    n_layers: int
    n_kv_head: int
    head_dim: int

    @property
    def per_layer_size(self) -> int:
        """Number of float32 elements in one ``K_layer_*.bin`` file."""
        return self.n_tokens * self.n_kv_head * self.head_dim

    @property
    def per_layer_bytes(self) -> int:
        return self.per_layer_size * 4


class KVDump:
    """Loaded KV cache dump from a single run.

    Parameters
    ----------
    dump_dir:
        Directory containing ``meta.json`` and per-layer ``.bin`` files.
    lazy:
        If ``True`` (default) individual ``.bin`` files are loaded only
        when requested via :meth:`get_K` / :meth:`get_V`.
    dtype:
        numpy dtype used for the loaded arrays.  Always ``float32``
        because that matches the C++ writer contract; the parameter is
        kept only so we can force an upcast in tests if needed.
    """

    def __init__(
        self,
        dump_dir: str | Path,
        *,
        lazy: bool = True,
        dtype: np.dtype | type = np.float32,
    ) -> None:
        self.dump_dir = Path(dump_dir)
        self._dtype = np.dtype(dtype)
        if not self.dump_dir.is_dir():
            raise KVDumpError(f"dump_dir does not exist: {self.dump_dir}")

        self.meta: dict[str, Any] = self._load_meta()
        self.shape: KVShape = self._shape_from_meta(self.meta)

        self._K: dict[int, np.ndarray] = {}
        self._V: dict[int, np.ndarray] = {}

        if not lazy:
            self._load_all()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _load_meta(self) -> dict[str, Any]:
        meta_path = self.dump_dir / "meta.json"
        if not meta_path.is_file():
            raise KVDumpError(f"missing meta.json in {self.dump_dir}")
        try:
            raw = meta_path.read_text(encoding="utf-8")
            return json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise KVDumpError(f"failed to read {meta_path}: {exc}") from exc

    @staticmethod
    def _shape_from_meta(meta: dict[str, Any]) -> KVShape:
        required = ("n_tokens", "n_layers", "n_kv_head", "head_dim")
        missing = [k for k in required if k not in meta]
        if missing:
            raise KVDumpError(f"meta.json missing required keys: {missing}")
        return KVShape(
            n_tokens=int(meta["n_tokens"]),
            n_layers=int(meta["n_layers"]),
            n_kv_head=int(meta["n_kv_head"]),
            head_dim=int(meta["head_dim"]),
        )

    def _load_all(self) -> None:
        for layer in range(self.shape.n_layers):
            self.get_K(layer)
            self.get_V(layer)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def n_tokens(self) -> int:
        return self.shape.n_tokens

    @property
    def n_layers(self) -> int:
        return self.shape.n_layers

    @property
    def n_kv_head(self) -> int:
        return self.shape.n_kv_head

    @property
    def head_dim(self) -> int:
        return self.shape.head_dim

    @property
    def run_name(self) -> str:
        return self.meta.get("run_name", self.dump_dir.name)

    @property
    def K(self) -> dict[int, np.ndarray]:
        """Dict view ``layer -> K tensor``. Loads missing layers on access."""
        for layer in range(self.shape.n_layers):
            if layer not in self._K:
                self.get_K(layer)
        return self._K

    @property
    def V(self) -> dict[int, np.ndarray]:
        for layer in range(self.shape.n_layers):
            if layer not in self._V:
                self.get_V(layer)
        return self._V

    # ------------------------------------------------------------------
    # Tensor access
    # ------------------------------------------------------------------

    def get_K(self, layer: int) -> np.ndarray:
        """Return K cache for ``layer`` with shape ``(T, H, D)``."""
        if layer in self._K:
            return self._K[layer]
        self._K[layer] = self._load_layer("K", layer)
        return self._K[layer]

    def get_V(self, layer: int) -> np.ndarray:
        if layer in self._V:
            return self._V[layer]
        self._V[layer] = self._load_layer("V", layer)
        return self._V[layer]

    def _load_layer(self, kind: str, layer: int) -> np.ndarray:
        if kind not in ("K", "V"):
            raise ValueError(f"kind must be 'K' or 'V', got {kind!r}")
        if not (0 <= layer < self.shape.n_layers):
            raise IndexError(f"layer {layer} out of range [0, {self.shape.n_layers})")

        path = self.dump_dir / f"{kind}_layer_{layer}.bin"
        if not path.is_file():
            raise KVDumpError(f"missing dump file: {path}")

        expected = self.shape.per_layer_bytes
        actual = path.stat().st_size
        if actual != expected:
            raise KVDumpError(
                f"{path} size mismatch: expected {expected} bytes "
                f"({self.shape.n_tokens}x{self.shape.n_kv_head}x{self.shape.head_dim}"
                f" fp32), got {actual}"
            )

        arr = np.fromfile(path, dtype=np.float32)
        arr = arr.reshape(self.shape.n_tokens, self.shape.n_kv_head, self.shape.head_dim)
        if self._dtype != np.float32:
            arr = arr.astype(self._dtype, copy=False)
        # Arrays live independently so callers can mutate freely.
        return arr

    # ------------------------------------------------------------------
    # Token masks
    # ------------------------------------------------------------------

    def vision_mask(self) -> np.ndarray:
        """Return boolean mask of length ``n_tokens`` marking vision tokens."""
        raw = self.meta.get("vision_token_mask")
        if raw is None:
            return np.zeros(self.shape.n_tokens, dtype=bool)
        mask = np.asarray(raw, dtype=bool)
        if mask.shape != (self.shape.n_tokens,):
            raise KVDumpError(
                f"vision_token_mask shape {mask.shape} does not match "
                f"n_tokens={self.shape.n_tokens}"
            )
        return mask

    def text_mask(self) -> np.ndarray:
        """Return boolean mask of length ``n_tokens`` marking text tokens."""
        return ~self.vision_mask()

    def has_vision(self) -> bool:
        return bool(self.vision_mask().any())

    def has_text(self) -> bool:
        return bool(self.text_mask().any())

    def token_indices(self, token_type: str) -> np.ndarray:
        """Return token indices for ``'all' | 'vision' | 'text'``."""
        if token_type == "all":
            return np.arange(self.shape.n_tokens)
        if token_type == "vision":
            return np.where(self.vision_mask())[0]
        if token_type == "text":
            return np.where(self.text_mask())[0]
        raise ValueError(f"token_type must be all|vision|text, got {token_type!r}")

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def iter_layers(self):
        for layer in range(self.shape.n_layers):
            yield layer, self.get_K(layer), self.get_V(layer)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"KVDump(dir={self.dump_dir}, n_tokens={self.n_tokens}, "
            f"n_layers={self.n_layers}, n_kv_head={self.n_kv_head}, "
            f"head_dim={self.head_dim})"
        )


# ---------------------------------------------------------------------------
# Writer (used by tests / mock data generation)
# ---------------------------------------------------------------------------


class KVDumpWriter:
    """Utility to produce a dump directory from numpy arrays.

    The C++ tool is the canonical producer in production; this writer
    exists so the Python test-suite can fabricate dumps with known
    statistics before the C++ tool is available.
    """

    def __init__(self, dump_dir: str | Path) -> None:
        self.dump_dir = Path(dump_dir)
        self.dump_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        K: dict[int, np.ndarray],
        V: dict[int, np.ndarray],
        vision_token_mask: list[bool] | np.ndarray | None = None,
        run_name: str | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        """Write a complete dump directory.

        ``K`` and ``V`` must share the same keys (0..n_layers-1) and the
        same ``(n_tokens, n_kv_head, head_dim)`` shape.
        """

        if not K or not V:
            raise ValueError("K and V must be non-empty dicts")
        layers = sorted(K.keys())
        if layers != sorted(V.keys()):
            raise ValueError("K and V must share the same layer indices")
        if layers != list(range(len(layers))):
            raise ValueError("Layer indices must be 0..N-1 with no gaps")

        first = K[layers[0]]
        shape = first.shape
        if len(shape) != 3:
            raise ValueError(f"tensors must be rank-3 (T,H,D), got {shape}")
        n_tokens, n_kv_head, head_dim = shape

        # Validate shapes and write binaries.
        for layer in layers:
            k = K[layer]
            v = V[layer]
            if k.shape != shape or v.shape != shape:
                raise ValueError(
                    f"layer {layer}: shape mismatch "
                    f"K={k.shape} V={v.shape} expected={shape}"
                )
            k.astype(np.float32, copy=False).tofile(self.dump_dir / f"K_layer_{layer}.bin")
            v.astype(np.float32, copy=False).tofile(self.dump_dir / f"V_layer_{layer}.bin")

        # Default mask: all text tokens.
        if vision_token_mask is None:
            mask_list = [False] * n_tokens
        else:
            mask = np.asarray(vision_token_mask, dtype=bool)
            if mask.shape != (n_tokens,):
                raise ValueError(
                    f"vision_token_mask length {mask.shape} does not match "
                    f"n_tokens={n_tokens}"
                )
            mask_list = mask.tolist()

        meta: dict[str, Any] = {
            "n_tokens": int(n_tokens),
            "n_layers": len(layers),
            "n_kv_head": int(n_kv_head),
            "head_dim": int(head_dim),
            "vision_token_mask": mask_list,
        }
        if run_name:
            meta["run_name"] = run_name
        if extra_meta:
            meta.update(extra_meta)

        meta_path = self.dump_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Functional helper
# ---------------------------------------------------------------------------


def load_dump(dump_dir: str | Path, *, lazy: bool = True) -> KVDump:
    """Convenience wrapper that returns a :class:`KVDump`."""
    return KVDump(dump_dir, lazy=lazy)
