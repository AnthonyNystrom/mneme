"""Phase-9 mmap-shared multi-process tests.

These exercise ``MmapSharedCoordinator`` directly: file-format header
validation, append/search round-trip, soft-delete, cross-process visibility.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pytest

from mneme._exceptions import (
    EmbedderDimensionError,
    EmbedderMismatchError,
    QuantizationError,
)
from mneme._multiproc import (
    MmapSharedCoordinator,
    _bitmap_get,
    _bitmap_set,
    _header_pack,
    _header_unpack,
    _ns_to_token,
    _token_to_ns,
)


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n).astype(np.float32)


def _rand(rng: np.random.Generator, dim: int = 8) -> np.ndarray:
    return _l2(rng.standard_normal(dim).astype(np.float32))


# --- Header pack/unpack round-trip ---


def test_header_pack_unpack_round_trip():
    fp = b"x" * 32
    packed = _header_pack(dim=128, count=10, capacity=100, dtype="float16", fp_hash=fp)
    assert len(packed) == 64
    dim, count, cap, dtype, fp_out = _header_unpack(packed)
    assert dim == 128
    assert count == 10
    assert cap == 100
    assert dtype == "float16"
    assert fp_out == fp


def test_header_unpack_rejects_bad_magic():
    bad = b"BAD!MAGI" + b"\x00" * 56
    from mneme._exceptions import MultiProcessLockError

    with pytest.raises(MultiProcessLockError, match="magic"):
        _header_unpack(bad)


def test_header_unpack_rejects_short_buffer():
    from mneme._exceptions import MultiProcessLockError

    with pytest.raises(MultiProcessLockError, match="short"):
        _header_unpack(b"too short")


# --- Namespace token encoding ---


def test_ns_token_round_trip_short():
    token = _ns_to_token("tenant_a")
    assert len(token) == 16
    assert _token_to_ns(token) == "tenant_a"


def test_ns_token_handles_long_namespace():
    long_ns = "a" * 100
    t1 = _ns_to_token(long_ns)
    t2 = _ns_to_token("a" * 99 + "b")
    assert len(t1) == 16
    assert len(t2) == 16
    assert t1 != t2  # different long namespaces shouldn't collide


# --- Bitmap helpers ---


def test_bitmap_set_and_get():
    buf = bytearray(8)
    _bitmap_set(buf, 0, True)
    _bitmap_set(buf, 17, True)
    assert _bitmap_get(bytes(buf), 0) is True
    assert _bitmap_get(bytes(buf), 1) is False
    assert _bitmap_get(bytes(buf), 17) is True
    _bitmap_set(buf, 0, False)
    assert _bitmap_get(bytes(buf), 0) is False


def test_bitmap_get_out_of_range_returns_false():
    assert _bitmap_get(b"\x00\x00", 999) is False


# --- File creation and validation ---


def test_creates_file_on_first_attach(tmp_path: Path):
    base = tmp_path / "cache"
    coord = MmapSharedCoordinator(base, dim=8, embedder_fingerprint="fp:v1")
    try:
        assert (tmp_path / "cache.vectors").exists()
        assert (tmp_path / "cache.vectors.lock").exists()
        # File mode for the data file should be owner-only.
        if os.name == "posix":
            mode = (tmp_path / "cache.vectors").stat().st_mode & 0o077
            assert mode == 0
    finally:
        coord.close()


def test_attach_validates_dim_mismatch(tmp_path: Path):
    base = tmp_path / "cache"
    c1 = MmapSharedCoordinator(base, dim=8, embedder_fingerprint="fp:v1")
    c1.close()
    with pytest.raises(EmbedderDimensionError, match="dim"):
        MmapSharedCoordinator(base, dim=16, embedder_fingerprint="fp:v1")


def test_attach_validates_dtype_mismatch(tmp_path: Path):
    base = tmp_path / "cache"
    c1 = MmapSharedCoordinator(base, dim=8, dtype="float32", embedder_fingerprint="fp")
    c1.close()
    with pytest.raises(QuantizationError, match="dtype"):
        MmapSharedCoordinator(base, dim=8, dtype="int8", embedder_fingerprint="fp")


def test_attach_validates_fingerprint_mismatch(tmp_path: Path):
    base = tmp_path / "cache"
    c1 = MmapSharedCoordinator(base, dim=8, embedder_fingerprint="fp:original")
    c1.close()
    with pytest.raises(EmbedderMismatchError, match="fingerprint"):
        MmapSharedCoordinator(base, dim=8, embedder_fingerprint="fp:different")


# --- Append + search ---


def test_append_then_search(tmp_path: Path):
    base = tmp_path / "cache"
    coord = MmapSharedCoordinator(base, dim=8, embedder_fingerprint="fp:v1")
    try:
        rng = np.random.default_rng(0)
        target = _l2(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        coord.append(1, target, "default")
        for i in range(5):
            coord.append(i + 2, _rand(rng, 8), "default")
        hits = coord.search(target, "default", k=1)
        assert len(hits) == 1
        assert hits[0][0] == 1
        assert hits[0][1] == pytest.approx(1.0, abs=1e-3)
    finally:
        coord.close()


def test_namespace_isolation(tmp_path: Path):
    base = tmp_path / "cache"
    coord = MmapSharedCoordinator(base, dim=8, embedder_fingerprint="fp:v1")
    try:
        a = _l2(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        coord.append(1, a, "tenant_a")
        coord.append(2, a, "tenant_b")
        hits_a = coord.search(a, "tenant_a", k=10)
        hits_b = coord.search(a, "tenant_b", k=10)
        assert {row_id for row_id, _ in hits_a} == {1}
        assert {row_id for row_id, _ in hits_b} == {2}
    finally:
        coord.close()


def test_search_empty_returns_empty_list(tmp_path: Path):
    base = tmp_path / "cache"
    coord = MmapSharedCoordinator(base, dim=8, embedder_fingerprint="fp:v1")
    try:
        rng = np.random.default_rng(0)
        assert coord.search(_rand(rng, 8), "default", k=5) == []
    finally:
        coord.close()


def test_remove_excludes_from_search(tmp_path: Path):
    base = tmp_path / "cache"
    coord = MmapSharedCoordinator(base, dim=8, embedder_fingerprint="fp:v1")
    try:
        target = _l2(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        other = _l2(np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        coord.append(1, target, "default")
        coord.append(2, other, "default")
        assert coord.remove(1) is True
        hits = coord.search(target, "default", k=10)
        assert 1 not in {row_id for row_id, _ in hits}
    finally:
        coord.close()


def test_remove_missing_id_returns_false(tmp_path: Path):
    base = tmp_path / "cache"
    coord = MmapSharedCoordinator(base, dim=8, embedder_fingerprint="fp:v1")
    try:
        assert coord.remove(99999) is False
    finally:
        coord.close()


def test_search_with_zero_k_returns_empty(tmp_path: Path):
    base = tmp_path / "cache"
    coord = MmapSharedCoordinator(base, dim=8, embedder_fingerprint="fp:v1")
    try:
        rng = np.random.default_rng(0)
        coord.append(1, _rand(rng, 8), "default")
        assert coord.search(_rand(rng, 8), "default", k=0) == []
    finally:
        coord.close()


def test_search_with_wrong_dim_raises(tmp_path: Path):
    base = tmp_path / "cache"
    coord = MmapSharedCoordinator(base, dim=8, embedder_fingerprint="fp:v1")
    try:
        with pytest.raises(ValueError, match="shape"):
            coord.search(np.zeros(16, dtype=np.float32), "default", k=1)
    finally:
        coord.close()


def test_append_with_wrong_dim_raises(tmp_path: Path):
    base = tmp_path / "cache"
    coord = MmapSharedCoordinator(base, dim=8, embedder_fingerprint="fp:v1")
    try:
        with pytest.raises(ValueError, match="shape"):
            coord.append(1, np.zeros(16, dtype=np.float32), "default")
    finally:
        coord.close()


# --- Header reflection ---


def test_read_header_reflects_appends(tmp_path: Path):
    base = tmp_path / "cache"
    coord = MmapSharedCoordinator(base, dim=8, embedder_fingerprint="fp:v1")
    try:
        dim0, count0, cap0, dtype0, _fp = coord.read_header()
        assert count0 == 0
        assert dim0 == 8
        assert dtype0 == "float32"
        coord.append(1, _l2(np.ones(8, dtype=np.float32)), "default")
        coord.append(2, _l2(np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)), "default")
        _, count1, cap1, _, _ = coord.read_header()
        assert count1 == 2
        assert cap1 >= cap0
    finally:
        coord.close()


# --- Cross-process visibility ---


def _writer_worker(base_path: str, fp: str) -> None:  # type: ignore[no-untyped-def]
    coord = MmapSharedCoordinator(base_path, dim=8, embedder_fingerprint=fp)
    try:
        target = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        coord.append(101, target, "default")
        coord.append(102, np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32), "default")
    finally:
        coord.close()


def test_cross_process_visibility(tmp_path: Path):
    """A writer process appends; a reader process sees the appended rows."""
    base = str(tmp_path / "shared")
    fp = "shared:fp:v1"

    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_writer_worker, args=(base, fp))
    p.start()
    p.join(timeout=10)
    assert p.exitcode == 0, f"writer exit {p.exitcode}"

    # Reader (this process) opens the same file.
    reader = MmapSharedCoordinator(base, dim=8, embedder_fingerprint=fp)
    try:
        _, count, _, _, _ = reader.read_header()
        assert count == 2
        target = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        hits = reader.search(target, "default", k=2)
        ids = {row_id for row_id, _ in hits}
        assert ids == {101, 102}
    finally:
        reader.close()


# --- Capacity growth ---


def test_grow_capacity_via_many_appends(tmp_path: Path):
    base = tmp_path / "cache"
    coord = MmapSharedCoordinator(base, dim=4, embedder_fingerprint="fp:v1", initial_capacity=2)
    try:
        rng = np.random.default_rng(0)
        for i in range(10):
            coord.append(i + 1, _rand(rng, 4), "default")
        _, count, capacity, _, _ = coord.read_header()
        assert count == 10
        assert capacity >= 10
    finally:
        coord.close()
