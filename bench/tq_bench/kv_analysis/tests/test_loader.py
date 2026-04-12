"""Tests for :mod:`tq_bench.kv_analysis.loader`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tq_bench.kv_analysis.loader import (
    KVDump,
    KVDumpError,
    KVDumpWriter,
    load_dump,
)
from tq_bench.kv_analysis.tests.conftest import (
    SyntheticDumpSpec,
    build_synthetic_dump,
)


def test_writer_then_loader_round_trip(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(n_tokens=6, n_layers=2, n_kv_head=2, head_dim=16)
    dump, K, V = build_synthetic_dump(tmp_path / "rt", spec)

    assert dump.n_tokens == spec.n_tokens
    assert dump.n_layers == spec.n_layers
    assert dump.n_kv_head == spec.n_kv_head
    assert dump.head_dim == spec.head_dim

    for layer in range(spec.n_layers):
        Kl = dump.get_K(layer)
        Vl = dump.get_V(layer)
        assert Kl.shape == (spec.n_tokens, spec.n_kv_head, spec.head_dim)
        np.testing.assert_allclose(Kl, K[layer])
        np.testing.assert_allclose(Vl, V[layer])


def test_vision_and_text_masks(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=10,
        n_layers=1,
        n_kv_head=1,
        head_dim=8,
        vision_token_count=4,
    )
    dump, _, _ = build_synthetic_dump(tmp_path / "masks", spec)

    v = dump.vision_mask()
    t = dump.text_mask()
    assert v.shape == (spec.n_tokens,)
    assert t.shape == (spec.n_tokens,)
    assert int(v.sum()) == spec.vision_token_count
    assert int(t.sum()) == spec.n_tokens - spec.vision_token_count
    assert np.all(v != t)

    assert dump.has_vision()
    assert dump.has_text()

    np.testing.assert_array_equal(
        dump.token_indices("vision"), np.arange(spec.vision_token_count)
    )
    np.testing.assert_array_equal(
        dump.token_indices("text"),
        np.arange(spec.vision_token_count, spec.n_tokens),
    )


def test_lazy_loading(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(n_tokens=4, n_layers=2, n_kv_head=1, head_dim=8)
    dump, _, _ = build_synthetic_dump(tmp_path / "lazy", spec)

    # Before calling anything, internal caches should be empty.
    assert dump._K == {}
    assert dump._V == {}

    dump.get_K(0)
    assert 0 in dump._K
    assert 1 not in dump._K

    # Accessing .K should fill everything.
    _ = dump.K
    assert sorted(dump._K.keys()) == [0, 1]


def test_missing_meta_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(KVDumpError, match="missing meta.json"):
        KVDump(tmp_path / "empty")


def test_wrong_shape_file_raises(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(n_tokens=4, n_layers=1, n_kv_head=1, head_dim=8)
    dump_dir = tmp_path / "bad"
    _ = build_synthetic_dump(dump_dir, spec)

    # Truncate the binary to force a size mismatch.
    bin_path = dump_dir / "K_layer_0.bin"
    data = bin_path.read_bytes()
    bin_path.write_bytes(data[: len(data) // 2])

    dump = KVDump(dump_dir)
    with pytest.raises(KVDumpError, match="size mismatch"):
        dump.get_K(0)


def test_load_dump_helper(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(n_tokens=2, n_layers=1, n_kv_head=1, head_dim=4)
    build_synthetic_dump(tmp_path / "helper", spec)

    dump = load_dump(tmp_path / "helper")
    assert isinstance(dump, KVDump)
    assert dump.n_tokens == 2


def test_writer_rejects_bad_mask(tmp_path: Path) -> None:
    writer = KVDumpWriter(tmp_path / "bad_mask")
    K = {0: np.zeros((3, 1, 4), dtype=np.float32)}
    V = {0: np.zeros((3, 1, 4), dtype=np.float32)}
    with pytest.raises(ValueError, match="vision_token_mask"):
        writer.write(K=K, V=V, vision_token_mask=[True, False])


def test_writer_rejects_layer_gaps(tmp_path: Path) -> None:
    writer = KVDumpWriter(tmp_path / "gap")
    K = {0: np.zeros((2, 1, 4), dtype=np.float32), 2: np.zeros((2, 1, 4), dtype=np.float32)}
    V = {0: np.zeros((2, 1, 4), dtype=np.float32), 2: np.zeros((2, 1, 4), dtype=np.float32)}
    with pytest.raises(ValueError, match="0..N-1"):
        writer.write(K=K, V=V)


def test_meta_json_fields(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(n_tokens=5, n_layers=1, n_kv_head=2, head_dim=8)
    dump_dir = tmp_path / "meta"
    build_synthetic_dump(dump_dir, spec, run_name="unit_test")

    meta = json.loads((dump_dir / "meta.json").read_text())
    assert meta["n_tokens"] == spec.n_tokens
    assert meta["n_layers"] == spec.n_layers
    assert meta["run_name"] == "unit_test"
    assert len(meta["vision_token_mask"]) == spec.n_tokens
    assert meta["cache_type_k"] == "f16"
