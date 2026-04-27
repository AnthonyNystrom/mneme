"""Conformance battery: every shipped Store impl plus a reference custom one
must pass the same contract. Phase 3b will add Redis/Postgres parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mneme._exceptions import (
    CacheClosedError,
    CheckpointError,
    EmbedderDimensionError,
    EmbedderMismatchError,
)
from mneme._store_memory import MemoryStore
from mneme._store_sqlite import SQLiteStore
from mneme._types import Store, StoredEntry

from .stores.inmemory_store import InMemoryStore

# --- Store factories: every shipped Store + reference InMemoryStore ---


def _factory_memory(tmp_path: Path) -> Store:
    del tmp_path
    return MemoryStore()


def _factory_sqlite(tmp_path: Path) -> Store:
    return SQLiteStore(tmp_path / "cache.db")


def _factory_sqlite_memory(tmp_path: Path) -> Store:
    del tmp_path
    return SQLiteStore(":memory:")


def _factory_inmemory_ref(tmp_path: Path) -> Store:
    del tmp_path
    return InMemoryStore()


_FACTORIES: list[tuple[str, Callable[[Path], Store]]] = [
    ("MemoryStore", _factory_memory),
    ("SQLiteStore[file]", _factory_sqlite),
    ("SQLiteStore[:memory:]", _factory_sqlite_memory),
    ("InMemoryStore[ref]", _factory_inmemory_ref),
]


@pytest.fixture(params=_FACTORIES, ids=[name for name, _ in _FACTORIES])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    _name, factory = request.param
    s = factory(tmp_path)
    s.open(embedder_fingerprint="fake:embedder:v1", embedder_dim=4)
    yield s
    s.close()


# --- Helpers ---


def _make_entry(
    *,
    namespace: str = "default",
    query_hash: str = "h" * 64,
    query: str = "hello",
    response: str = "world",
    embedding: bytes | None = None,
    metadata: dict[str, Any] | None = None,
    created_at: int = 1_700_000_000,
    last_accessed_at: int | None = None,
    ttl: int | None = None,
    access_count: int = 0,
) -> StoredEntry:
    if embedding is None:
        embedding = np.zeros(4, dtype=np.float32).tobytes()
    if last_accessed_at is None:
        last_accessed_at = created_at
    return StoredEntry(
        id=0,  # ignored; assigned by store
        namespace=namespace,
        query_hash=query_hash,
        query=query,
        response=response,
        embedding=embedding,
        metadata=metadata if metadata is not None else {},
        created_at=created_at,
        last_accessed_at=last_accessed_at,
        ttl=ttl,
        access_count=access_count,
    )


# --- Protocol structural conformance ---


def test_factories_produce_store_protocol_instances(tmp_path: Path):
    for name, factory in _FACTORIES:
        inst = factory(tmp_path)
        assert isinstance(inst, Store), f"{name} does not satisfy Store protocol"


# --- Lifecycle ---


def test_open_validates_fingerprint(tmp_path: Path):
    for name, factory in _FACTORIES:
        s = factory(tmp_path)
        s.open(embedder_fingerprint="fp:a", embedder_dim=4)
        s.close()
        # Reopen with mismatched fingerprint
        if name in ("MemoryStore", "InMemoryStore[ref]", "SQLiteStore[:memory:]"):
            # Memory-only stores forget across close/new instance.
            continue
        s2 = factory(tmp_path)
        with pytest.raises(EmbedderMismatchError):
            s2.open(embedder_fingerprint="fp:b", embedder_dim=4)
        s2.close()


def test_open_validates_dim(tmp_path: Path):
    for name, factory in _FACTORIES:
        if name in ("MemoryStore", "InMemoryStore[ref]", "SQLiteStore[:memory:]"):
            continue
        s = factory(tmp_path)
        s.open(embedder_fingerprint="fp:a", embedder_dim=4)
        s.close()
        s2 = factory(tmp_path)
        with pytest.raises(EmbedderDimensionError):
            s2.open(embedder_fingerprint="fp:a", embedder_dim=8)
        s2.close()


def test_close_makes_subsequent_ops_raise(store: Store):
    store.close()
    with pytest.raises(CacheClosedError):
        store.get_by_hash("default", "x")


# --- Insert / read round-trip ---


def test_insert_returns_assigned_id(store: Store):
    e = _make_entry(query_hash="a" * 64)
    new_id = store.insert(e)
    assert isinstance(new_id, int)
    assert new_id > 0


def test_insert_round_trip_via_get_by_hash(store: Store):
    e = _make_entry(query_hash="b" * 64, query="hi", response="bye", metadata={"k": "v"})
    new_id = store.insert(e)
    fetched = store.get_by_hash("default", "b" * 64)
    assert fetched is not None
    assert fetched.id == new_id
    assert fetched.query == "hi"
    assert fetched.response == "bye"
    assert fetched.metadata == {"k": "v"}


def test_insert_round_trip_via_get_by_id(store: Store):
    e = _make_entry(query_hash="c" * 64)
    new_id = store.insert(e)
    fetched = store.get_by_id(new_id)
    assert fetched is not None
    assert fetched.query_hash == "c" * 64


def test_get_by_hash_miss_returns_none(store: Store):
    assert store.get_by_hash("default", "nope" * 16) is None


def test_get_by_id_miss_returns_none(store: Store):
    assert store.get_by_id(99999) is None


def test_insert_replaces_on_hash_collision(store: Store):
    e1 = _make_entry(query_hash="d" * 64, response="first")
    store.insert(e1)
    e2 = _make_entry(query_hash="d" * 64, response="second")
    id2 = store.insert(e2)
    # Same logical entry, may or may not get a new id depending on impl, but
    # the row's response must be the latest write.
    fetched = store.get_by_hash("default", "d" * 64)
    assert fetched is not None
    assert fetched.response == "second"
    assert id2 == fetched.id


def test_embedding_bytes_round_trip(store: Store):
    vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    e = _make_entry(query_hash="e" * 64, embedding=vec.tobytes())
    store.insert(e)
    fetched = store.get_by_hash("default", "e" * 64)
    assert fetched is not None
    rt = np.frombuffer(fetched.embedding, dtype=np.float32)
    np.testing.assert_array_equal(rt, vec)


def test_metadata_json_round_trip(store: Store):
    meta = {"nested": {"a": 1, "b": [1, 2, 3]}, "s": "hello"}
    e = _make_entry(query_hash="f" * 64, metadata=meta)
    store.insert(e)
    fetched = store.get_by_hash("default", "f" * 64)
    assert fetched is not None
    assert fetched.metadata == meta


# --- Counts & namespaces ---


def test_count_global_and_per_namespace(store: Store):
    store.insert(_make_entry(namespace="a", query_hash="1" * 64))
    store.insert(_make_entry(namespace="a", query_hash="2" * 64))
    store.insert(_make_entry(namespace="b", query_hash="3" * 64))
    assert store.count() == 3
    assert store.count("a") == 2
    assert store.count("b") == 1
    assert store.count("c") == 0


def test_list_namespaces_returns_sorted_distinct(store: Store):
    store.insert(_make_entry(namespace="z", query_hash="1" * 64))
    store.insert(_make_entry(namespace="a", query_hash="2" * 64))
    store.insert(_make_entry(namespace="m", query_hash="3" * 64))
    assert store.list_namespaces() == ["a", "m", "z"]


def test_namespace_isolation_on_get_by_hash(store: Store):
    h = "x" * 64
    store.insert(_make_entry(namespace="a", query_hash=h, response="A"))
    store.insert(_make_entry(namespace="b", query_hash=h, response="B"))
    assert store.get_by_hash("a", h).response == "A"  # type: ignore[union-attr]
    assert store.get_by_hash("b", h).response == "B"  # type: ignore[union-attr]
    assert store.get_by_hash("c", h) is None


# --- LRU & ordering ---


def test_iter_lru_ids_returns_oldest_first(store: Store):
    store.insert(_make_entry(query_hash="1" * 64, last_accessed_at=300))
    store.insert(_make_entry(query_hash="2" * 64, last_accessed_at=100))
    store.insert(_make_entry(query_hash="3" * 64, last_accessed_at=200))
    ids = list(store.iter_lru_ids(2))
    # Oldest two: ts=100 then ts=200
    e0 = store.get_by_id(ids[0])
    e1 = store.get_by_id(ids[1])
    assert e0 is not None
    assert e1 is not None
    assert e0.last_accessed_at <= e1.last_accessed_at


def test_iter_lru_ids_namespace_scoped(store: Store):
    store.insert(_make_entry(namespace="a", query_hash="1" * 64, last_accessed_at=100))
    store.insert(_make_entry(namespace="b", query_hash="2" * 64, last_accessed_at=50))
    ids = list(store.iter_lru_ids(10, namespace="a"))
    assert len(ids) == 1
    e = store.get_by_id(ids[0])
    assert e is not None
    assert e.namespace == "a"


def test_iter_all_yields_in_id_order(store: Store):
    ids = []
    for i in range(5):
        ids.append(store.insert(_make_entry(query_hash=str(i) * 64)))
    seen = [e.id for e in store.iter_all()]
    assert seen == sorted(seen)
    assert set(seen) == set(ids)


def test_iter_since_returns_post_id(store: Store):
    ids = []
    for i in range(5):
        ids.append(store.insert(_make_entry(query_hash=str(i) * 64)))
    cutoff = ids[2]
    seen = [e.id for e in store.iter_since(cutoff)]
    assert all(eid > cutoff for eid in seen)
    assert set(seen) == {ids[3], ids[4]}


# --- Update / delete ---


def test_update_access_advances_timestamp_and_count(store: Store):
    new_id = store.insert(_make_entry(query_hash="u" * 64, last_accessed_at=100))
    store.update_access(new_id, now=500)
    fetched = store.get_by_id(new_id)
    assert fetched is not None
    assert fetched.last_accessed_at == 500
    assert fetched.access_count == 1


def test_update_access_on_missing_id_is_noop(store: Store):
    store.update_access(99999, now=500)  # must not raise


def test_delete_by_id_returns_true_when_existing(store: Store):
    new_id = store.insert(_make_entry(query_hash="d" * 64))
    assert store.delete_by_id(new_id) is True
    assert store.get_by_id(new_id) is None


def test_delete_by_id_returns_false_when_missing(store: Store):
    assert store.delete_by_id(99999) is False


def test_delete_expired_removes_only_expired(store: Store):
    now = 1_000_000
    store.insert(_make_entry(query_hash="1" * 64, created_at=now - 100, ttl=50))  # expired
    store.insert(_make_entry(query_hash="2" * 64, created_at=now - 100, ttl=200))  # alive
    store.insert(_make_entry(query_hash="3" * 64, created_at=now, ttl=None))  # no ttl
    deleted = store.delete_expired(now=now)
    assert deleted == 1
    assert store.count() == 2


def test_delete_expired_namespace_scoped(store: Store):
    now = 1_000_000
    store.insert(_make_entry(namespace="a", query_hash="1" * 64, created_at=now - 100, ttl=50))
    store.insert(_make_entry(namespace="b", query_hash="2" * 64, created_at=now - 100, ttl=50))
    deleted = store.delete_expired(now=now, namespace="a")
    assert deleted == 1
    assert store.count("a") == 0
    assert store.count("b") == 1


def test_clear_namespace_removes_only_that_namespace(store: Store):
    store.insert(_make_entry(namespace="a", query_hash="1" * 64))
    store.insert(_make_entry(namespace="a", query_hash="2" * 64))
    store.insert(_make_entry(namespace="b", query_hash="3" * 64))
    deleted = store.clear_namespace("a")
    assert deleted == 2
    assert store.count("a") == 0
    assert store.count("b") == 1


# --- Quotas ---


def test_quota_set_then_get(store: Store):
    assert store.get_namespace_quota("tenant") is None
    store.set_namespace_quota("tenant", 1000)
    assert store.get_namespace_quota("tenant") == 1000


def test_quota_overwrite(store: Store):
    store.set_namespace_quota("tenant", 100)
    store.set_namespace_quota("tenant", 200)
    assert store.get_namespace_quota("tenant") == 200


# --- Coordination (version_counter, meta) ---


def test_version_counter_starts_at_zero(store: Store):
    # On a freshly opened store, the counter must be zero.
    assert store.read_version_counter() == 0


def test_version_counter_increments_on_insert(store: Store):
    before = store.read_version_counter()
    store.insert(_make_entry(query_hash="1" * 64))
    after = store.read_version_counter()
    assert after > before


def test_version_counter_increments_on_delete(store: Store):
    new_id = store.insert(_make_entry(query_hash="1" * 64))
    before = store.read_version_counter()
    store.delete_by_id(new_id)
    after = store.read_version_counter()
    assert after > before


def test_meta_round_trip(store: Store):
    assert store.read_meta("custom_key") is None
    store.write_meta("custom_key", "custom_value")
    assert store.read_meta("custom_key") == "custom_value"


def test_meta_overwrite(store: Store):
    store.write_meta("k", "v1")
    store.write_meta("k", "v2")
    assert store.read_meta("k") == "v2"


# --- Health ---


def test_integrity_check_on_fresh_store(store: Store):
    assert store.integrity_check() is True


# --- Snapshot / restore (only stores that support persistence) ---


def test_snapshot_supported_or_raises_clearly(store: Store, tmp_path: Path):
    """SQLiteStore (file or :memory:) supports snapshot via SQLite's backup API.
    MemoryStore and the reference InMemoryStore raise CheckpointError."""
    store.insert(_make_entry(query_hash="1" * 64, response="snapshot-data"))
    dest = tmp_path / "snap.db"
    if isinstance(store, SQLiteStore):
        store.snapshot_to(dest)
        assert dest.exists()
        # Restore round-trip works only when source is on-disk (so we have a
        # real file to copy via shutil.copy2 inside restore_from). For :memory:
        # snapshots the file exists but restore is exercised via a fresh open.
        restored = SQLiteStore.restore_from(dest, tmp_path / "restored.db")
        restored.open(embedder_fingerprint="fake:embedder:v1", embedder_dim=4)
        try:
            fetched = restored.get_by_hash("default", "1" * 64)
            assert fetched is not None
            assert fetched.response == "snapshot-data"
        finally:
            restored.close()
    else:
        with pytest.raises(CheckpointError):
            store.snapshot_to(dest)
