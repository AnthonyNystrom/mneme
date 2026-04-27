"""``MemoryStore``-specific tests: capacity, no persistence, threading."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from mneme._exceptions import (
    CacheClosedError,
    CheckpointError,
    StoreBackendError,
)
from mneme._store_memory import MemoryStore
from mneme._types import StoredEntry


def _entry(h: str, ns: str = "default") -> StoredEntry:
    return StoredEntry(
        id=0,
        namespace=ns,
        query_hash=h,
        query="q",
        response="r",
        embedding=np.zeros(4, dtype=np.float32).tobytes(),
        metadata={},
        created_at=0,
        last_accessed_at=0,
        ttl=None,
        access_count=0,
    )


def test_max_entries_blocks_new_inserts_at_capacity():
    s = MemoryStore(max_entries=2)
    s.open("fp:v1", 4)
    s.insert(_entry("1" * 64))
    s.insert(_entry("2" * 64))
    with pytest.raises(StoreBackendError, match="capacity"):
        s.insert(_entry("3" * 64))
    s.close()


def test_max_entries_allows_replacing_existing_at_capacity():
    s = MemoryStore(max_entries=2)
    s.open("fp:v1", 4)
    s.insert(_entry("1" * 64))
    s.insert(_entry("2" * 64))
    # Replacement (same hash) does not exceed capacity.
    e = StoredEntry(
        id=0,
        namespace="default",
        query_hash="1" * 64,
        query="updated",
        response="updated",
        embedding=np.zeros(4, dtype=np.float32).tobytes(),
        metadata={},
        created_at=0,
        last_accessed_at=0,
        ttl=None,
        access_count=0,
    )
    s.insert(e)
    fetched = s.get_by_hash("default", "1" * 64)
    assert fetched is not None
    assert fetched.query == "updated"
    s.close()


def test_data_lost_on_close():
    s = MemoryStore()
    s.open("fp:v1", 4)
    s.insert(_entry("1" * 64))
    s.close()
    s2 = MemoryStore()
    s2.open("fp:v1", 4)
    assert s2.count() == 0
    s2.close()


def test_dumps_raises_checkpoint_error(tmp_path):
    s = MemoryStore()
    s.open("fp:v1", 4)
    with pytest.raises(CheckpointError):
        s.snapshot_to(tmp_path / "snap.db")
    s.close()


def test_loads_raises_checkpoint_error(tmp_path):
    with pytest.raises(CheckpointError):
        MemoryStore.restore_from(tmp_path / "src.db", tmp_path / "dst.db")


def test_thread_safety_concurrent_inserts():
    s = MemoryStore()
    s.open("fp:v1", 4)
    n_threads = 8
    n_per_thread = 100

    def worker(tid: int) -> None:
        for i in range(n_per_thread):
            s.insert(_entry(f"{tid:02d}{i:062d}"))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert s.count() == n_threads * n_per_thread
    assert s.integrity_check() is True
    s.close()


def test_close_is_idempotent():
    s = MemoryStore()
    s.open("fp:v1", 4)
    s.close()
    s.close()  # second close must not raise


def test_use_after_close_raises():
    s = MemoryStore()
    s.open("fp:v1", 4)
    s.close()
    with pytest.raises(CacheClosedError):
        s.count()


def test_metadata_is_deep_copied_on_insert():
    """Mutating the caller's metadata dict after insert must not bleed in."""
    s = MemoryStore()
    s.open("fp:v1", 4)
    meta = {"k": [1, 2, 3]}
    e = StoredEntry(
        id=0,
        namespace="default",
        query_hash="m" * 64,
        query="q",
        response="r",
        embedding=np.zeros(4, dtype=np.float32).tobytes(),
        metadata=meta,
        created_at=0,
        last_accessed_at=0,
        ttl=None,
        access_count=0,
    )
    s.insert(e)
    meta["k"].append(99)  # mutate original
    fetched = s.get_by_hash("default", "m" * 64)
    assert fetched is not None
    assert fetched.metadata == {"k": [1, 2, 3]}
    s.close()
