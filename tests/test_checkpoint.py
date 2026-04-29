"""Phase-10 checkpoint tests: round-trip dumps/loads."""

from __future__ import annotations

import json
import tarfile
import time
from pathlib import Path

import pytest

from mneme import (
    AsyncSemanticCache,
    CheckpointError,
    EmbedderDimensionError,
    EmbedderMismatchError,
    MemoryStore,
    SemanticCache,
)
from mneme._checkpoint import CHECKPOINT_FORMAT_VERSION, restore

from .fakes import FakeAsyncEmbedder, FakeEmbedder, HighDimEmbedder

# --- Sync round-trip ---


def test_dumps_creates_tar_gz_archive(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    with SemanticCache(path=tmp_path / "src.db", embedder=e) as cache:
        cache.put("hi", "there")
        archive = tmp_path / "snap.tar.gz"
        cache.dumps(archive)
    assert archive.exists()
    # Owner-only mode (POSIX best-effort).
    import os

    if os.name == "posix":
        mode = archive.stat().st_mode & 0o077
        assert mode == 0


def test_dumps_archive_contains_manifest_and_store(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    with SemanticCache(path=tmp_path / "src.db", embedder=e) as cache:
        cache.put("hi", "there")
        archive = tmp_path / "snap.tar.gz"
        cache.dumps(archive)
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    assert "manifest.json" in names
    assert any(n.startswith("store") for n in names)


def test_round_trip_preserves_entries(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    src_db = tmp_path / "src.db"
    archive = tmp_path / "snap.tar.gz"
    with SemanticCache(path=src_db, embedder=e) as cache:
        cache.put("q1", "r1")
        cache.put("q2", "r2", namespace="tenant_a")
        cache.put("q3", "r3", metadata={"k": "v"})
        cache.dumps(archive)

    dst_db = tmp_path / "dst.db"
    with SemanticCache.loads(archive, dst_db, e) as restored:
        # Same entries
        assert restored.stats().entries == 3
        h1 = restored.get("q1")
        assert h1 is not None
        assert h1.response == "r1"
        # Namespace preserved
        h2 = restored.get("q2", namespace="tenant_a")
        assert h2 is not None
        assert h2.response == "r2"
        # Metadata preserved
        h3 = restored.get("q3")
        assert h3 is not None
        assert h3.metadata == {"k": "v"}


def test_round_trip_preserves_counters(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    src_db = tmp_path / "src.db"
    archive = tmp_path / "snap.tar.gz"
    with SemanticCache(path=src_db, embedder=e) as cache:
        cache.put("hi", "there")
        cache.get("hi")
        cache.get("hi")
        cache.get("missing")  # 1 miss
        cache.dumps(archive)
    dst_db = tmp_path / "dst.db"
    with SemanticCache.loads(archive, dst_db, e) as restored:
        s = restored.stats()
        assert s.hits_exact == 2
        assert s.misses == 1


def test_round_trip_preserves_namespaces(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    archive = tmp_path / "snap.tar.gz"
    with SemanticCache(path=tmp_path / "src.db", embedder=e) as cache:
        cache.put("q", "r", namespace="alpha")
        cache.put("q", "r", namespace="beta")
        cache.put("q", "r", namespace="gamma")
        cache.dumps(archive)
    with SemanticCache.loads(archive, tmp_path / "dst.db", e) as restored:
        assert restored.list_namespaces() == ["alpha", "beta", "gamma"]


def test_round_trip_preserves_vector_dtype_via_manifest(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    archive = tmp_path / "snap.tar.gz"
    with SemanticCache(path=tmp_path / "src.db", embedder=e, vector_dtype="int8") as cache:
        cache.put("hi", "there")
        cache.dumps(archive)
    with SemanticCache.loads(archive, tmp_path / "dst.db", e) as restored:
        # Manifest's int8 used as default unless caller overrides.
        assert restored.health().vector_dtype == "int8"


def test_loads_kwargs_override_manifest_defaults(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    archive = tmp_path / "snap.tar.gz"
    with SemanticCache(path=tmp_path / "src.db", embedder=e, vector_dtype="int8") as cache:
        cache.put("hi", "there")
        cache.dumps(archive)
    with SemanticCache.loads(archive, tmp_path / "dst.db", e, vector_dtype="float32") as restored:
        assert restored.health().vector_dtype == "float32"


# --- Manifest validation ---


def test_loads_fingerprint_mismatch_raises_no_force(tmp_path: Path):
    """No force flag — fingerprint mismatch always raises."""
    e1 = FakeEmbedder(dim=8, fingerprint="fp:original")
    e2 = FakeEmbedder(dim=8, fingerprint="fp:different")
    archive = tmp_path / "snap.tar.gz"
    with SemanticCache(path=tmp_path / "src.db", embedder=e1) as cache:
        cache.put("q", "r")
        cache.dumps(archive)
    with pytest.raises(EmbedderMismatchError, match="reembed"):
        SemanticCache.loads(archive, tmp_path / "dst.db", e2)


def test_loads_dim_mismatch_raises(tmp_path: Path):
    """Same fingerprint but different dim still raises (defensive)."""
    fp = "stable:fp"
    e1 = HighDimEmbedder(dim=8, fingerprint=fp)
    e2 = HighDimEmbedder(dim=16, fingerprint=fp)
    archive = tmp_path / "snap.tar.gz"
    with SemanticCache(path=tmp_path / "src.db", embedder=e1) as cache:
        cache.put("q", "r")
        cache.dumps(archive)
    with pytest.raises(EmbedderDimensionError):
        SemanticCache.loads(archive, tmp_path / "dst.db", e2)


# --- Failure modes ---


def test_dumps_on_memory_store_raises_checkpoint_error(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    with (
        SemanticCache(store=MemoryStore(), embedder=e) as cache,
        pytest.raises(CheckpointError),
    ):
        cache.dumps(tmp_path / "snap.tar.gz")


def test_dumps_on_closed_cache_raises(tmp_path: Path):
    from mneme import CacheClosedError

    e = FakeEmbedder(dim=8)
    cache = SemanticCache(path=tmp_path / "c.db", embedder=e)
    cache.put("hi", "there")
    cache.close()
    # Closed cache => CacheClosedError fires inside the lock guard before
    # _checkpoint.dumps runs (we wrap dumps with _check_open).
    with pytest.raises(CacheClosedError):
        cache.dumps(tmp_path / "snap.tar.gz")


def test_loads_missing_source_raises(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    with pytest.raises(CheckpointError, match="does not exist"):
        SemanticCache.loads(tmp_path / "nope.tar.gz", tmp_path / "dst.db", e)


def test_loads_corrupt_archive_raises(tmp_path: Path):
    archive = tmp_path / "bad.tar.gz"
    archive.write_bytes(b"not a tar.gz file at all")
    e = FakeEmbedder(dim=8)
    with pytest.raises(CheckpointError):
        SemanticCache.loads(archive, tmp_path / "dst.db", e)


def test_loads_missing_manifest_raises(tmp_path: Path):
    archive = tmp_path / "no_manifest.tar.gz"
    # Make a tar.gz with only a 'store' subdir, no manifest.
    fake_store = tmp_path / "fake_store"
    fake_store.write_bytes(b"\x00" * 16)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(fake_store, arcname="store")
    e = FakeEmbedder(dim=8)
    with pytest.raises(CheckpointError, match="manifest"):
        SemanticCache.loads(archive, tmp_path / "dst.db", e)


def test_loads_unsupported_store_backend_raises(tmp_path: Path):
    """Manifest with store_backend=postgres is rejected in v1."""
    archive = tmp_path / "pg_archive.tar.gz"
    e = FakeEmbedder(dim=8)
    with tarfile.open(archive, "w:gz") as tar:
        manifest = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "schema_version": 1,
            "embedder_fingerprint": e.fingerprint,
            "embedder_dim": e.dim,
            "index_backend": "numpy",
            "store_backend": "postgres",  # not supported by loads
            "vector_dtype": "float32",
            "created_at": int(time.time()),
            "namespaces": [],
        }
        manifest_bytes = json.dumps(manifest).encode()
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        tar.add(manifest_path, arcname="manifest.json")
        # Add a placeholder store/ dir
        store_dir = tmp_path / "_dummy_store"
        store_dir.write_bytes(b"\x00")
        tar.add(store_dir, arcname="store")
    with pytest.raises(CheckpointError, match="postgres"):
        SemanticCache.loads(archive, tmp_path / "dst.db", e)


# --- Manifest format ---


def test_manifest_has_documented_fields(tmp_path: Path):
    e = FakeEmbedder(dim=8, fingerprint="fp:v1")
    archive = tmp_path / "snap.tar.gz"
    with SemanticCache(path=tmp_path / "src.db", embedder=e) as cache:
        cache.put("hi", "there", namespace="ns_a")
        cache.dumps(archive)
    with (
        tarfile.open(archive, "r:gz") as tar,
        tar.extractfile("manifest.json") as f,  # type: ignore[union-attr]
    ):
        manifest = json.loads(f.read().decode())  # type: ignore[union-attr]
    expected_keys = {
        "format_version",
        "schema_version",
        "embedder_fingerprint",
        "embedder_dim",
        "index_backend",
        "store_backend",
        "vector_dtype",
        "created_at",
        "namespaces",
    }
    assert expected_keys <= set(manifest.keys())
    assert manifest["embedder_fingerprint"] == "fp:v1"
    assert manifest["embedder_dim"] == 8
    assert "ns_a" in manifest["namespaces"]


# --- restore() returns manifest ---


def test_restore_returns_manifest(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    archive = tmp_path / "snap.tar.gz"
    with SemanticCache(path=tmp_path / "src.db", embedder=e) as cache:
        cache.put("hi", "there")
        cache.dumps(archive)
    manifest = restore(archive, tmp_path / "dst.db", e)
    assert manifest["embedder_dim"] == 8
    assert manifest["embedder_fingerprint"] == e.fingerprint
    assert (tmp_path / "dst.db").exists()


# --- Async ---


async def test_async_round_trip_preserves_entries(tmp_path: Path):
    e = FakeAsyncEmbedder(dim=8)
    archive = tmp_path / "snap.tar.gz"
    async with AsyncSemanticCache(path=tmp_path / "src.db", embedder=e) as cache:
        await cache.put("hi", "there")
        await cache.put("bye", "later", namespace="t1")
        await cache.dumps(archive)
    cache2 = await AsyncSemanticCache.loads(archive, tmp_path / "dst.db", e)
    try:
        assert cache2.stats().entries == 2
        h = await cache2.get("hi")
        assert h is not None
        assert h.response == "there"
        h2 = await cache2.get("bye", namespace="t1")
        assert h2 is not None
        assert h2.response == "later"
    finally:
        await cache2.close()


async def test_async_loads_fingerprint_mismatch_raises(tmp_path: Path):
    e1 = FakeAsyncEmbedder(dim=8, fingerprint="fp:a")
    e2 = FakeAsyncEmbedder(dim=8, fingerprint="fp:b")
    archive = tmp_path / "snap.tar.gz"
    async with AsyncSemanticCache(path=tmp_path / "src.db", embedder=e1) as cache:
        await cache.put("q", "r")
        await cache.dumps(archive)
    with pytest.raises(EmbedderMismatchError):
        await AsyncSemanticCache.loads(archive, tmp_path / "dst.db", e2)
