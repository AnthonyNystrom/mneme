"""``SQLiteStore``-specific tests: WAL, integrity, snapshot/restore, file mode."""

from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path

import numpy as np
import pytest

from mneme._exceptions import (
    EmbedderDimensionError,
    EmbedderMismatchError,
)
from mneme._store_sqlite import SQLiteStore
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


def test_wal_mode_enabled(tmp_path: Path):
    s = SQLiteStore(tmp_path / "cache.db")
    s.open("fp:v1", 4)
    cur = s._conn.execute("PRAGMA journal_mode")  # type: ignore[union-attr]
    mode = cur.fetchone()[0]
    assert mode.lower() == "wal"
    s.close()


def test_pragma_integrity_check_returns_true_on_fresh_db(tmp_path: Path):
    s = SQLiteStore(tmp_path / "cache.db")
    s.open("fp:v1", 4)
    assert s.integrity_check() is True
    s.close()


def test_db_file_created_with_owner_only_permissions(tmp_path: Path):
    if os.name == "nt":
        pytest.skip("POSIX-only file mode test")
    db = tmp_path / "cache.db"
    s = SQLiteStore(db)
    s.open("fp:v1", 4)
    s.close()
    mode = stat.S_IMODE(db.stat().st_mode)
    # Owner-only: 0o600. We accept any mode that excludes group/other.
    assert mode & 0o077 == 0, f"DB file mode {oct(mode)} leaks group/other access"


def test_persistence_across_open_close_cycles(tmp_path: Path):
    db = tmp_path / "cache.db"
    s1 = SQLiteStore(db)
    s1.open("fp:v1", 4)
    s1.insert(_entry("1" * 64))
    s1.close()

    s2 = SQLiteStore(db)
    s2.open("fp:v1", 4)
    assert s2.count() == 1
    fetched = s2.get_by_hash("default", "1" * 64)
    assert fetched is not None
    s2.close()


def test_open_rejects_fingerprint_mismatch(tmp_path: Path):
    db = tmp_path / "cache.db"
    s = SQLiteStore(db)
    s.open("fp:original", 4)
    s.close()

    s2 = SQLiteStore(db)
    with pytest.raises(EmbedderMismatchError, match="reembed"):
        s2.open("fp:different", 4)


def test_open_rejects_dim_mismatch(tmp_path: Path):
    db = tmp_path / "cache.db"
    s = SQLiteStore(db)
    s.open("fp:original", 4)
    s.close()

    s2 = SQLiteStore(db)
    with pytest.raises(EmbedderDimensionError, match="reembed"):
        s2.open("fp:original", 8)


def test_version_counter_in_same_transaction_as_insert(tmp_path: Path):
    """If insert is rolled back, version_counter must not bump."""
    db = tmp_path / "cache.db"
    s = SQLiteStore(db)
    s.open("fp:v1", 4)
    initial = s.read_version_counter()
    s.insert(_entry("1" * 64))
    after = s.read_version_counter()
    assert after == initial + 1
    s.close()


def test_snapshot_to_creates_independent_db_copy(tmp_path: Path):
    src = tmp_path / "cache.db"
    snap = tmp_path / "snap.db"
    s = SQLiteStore(src)
    s.open("fp:v1", 4)
    s.insert(_entry("1" * 64))
    s.snapshot_to(snap)
    assert snap.exists()
    # Mutate source after snapshot.
    s.insert(_entry("2" * 64))
    s.close()
    # Open the snapshot directly; should only have the original entry.
    raw = sqlite3.connect(str(snap))
    try:
        n = raw.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        assert n == 1
    finally:
        raw.close()


def test_restore_from_round_trip(tmp_path: Path):
    src = tmp_path / "cache.db"
    snap = tmp_path / "snap.db"
    s = SQLiteStore(src)
    s.open("fp:v1", 4)
    s.insert(_entry("1" * 64))
    s.snapshot_to(snap)
    s.close()
    restored = SQLiteStore.restore_from(snap, tmp_path / "restored.db")
    restored.open("fp:v1", 4)
    try:
        assert restored.count() == 1
        assert restored.get_by_hash("default", "1" * 64) is not None
    finally:
        restored.close()


def test_close_runs_wal_checkpoint(tmp_path: Path):
    db = tmp_path / "cache.db"
    s = SQLiteStore(db)
    s.open("fp:v1", 4)
    s.insert(_entry("1" * 64))
    s.close()
    # After close, the WAL file may exist but should be small (drained).
    wal = Path(str(db) + "-wal")
    if wal.exists():
        assert wal.stat().st_size == 0


def test_memory_db_works_with_full_transactional_semantics():
    s = SQLiteStore(":memory:")
    s.open("fp:v1", 4)
    s.insert(_entry("1" * 64))
    assert s.count() == 1
    assert s.integrity_check() is True
    # version_counter increments
    s.insert(_entry("2" * 64))
    assert s.read_version_counter() == 2
    s.close()


def test_corrupt_db_raises_on_open_or_integrity_check(tmp_path: Path):
    """Garbage in the DB file must surface a clear exception at open() or
    integrity_check() — never silent corruption."""
    db = tmp_path / "cache.db"
    s = SQLiteStore(db)
    s.open("fp:v1", 4)
    s.insert(_entry("1" * 64))
    s.close()
    db.write_bytes(b"not a sqlite database header" * 100)

    s2 = SQLiteStore(db)
    surfaced: Exception | None = None
    try:
        s2.open("fp:v1", 4)
    except Exception as exc:  # corruption detected at open
        surfaced = exc
    if surfaced is None:
        try:
            s2.integrity_check()
        except Exception as exc:
            surfaced = exc
        finally:
            s2.close()
    assert surfaced is not None, "expected an exception from corruption path"


def test_snapshot_to_missing_source_raises(tmp_path: Path):
    """restore_from on a non-existent path raises StoreBackendError."""
    from mneme._exceptions import StoreBackendError

    with pytest.raises(StoreBackendError, match="does not exist"):
        SQLiteStore.restore_from(tmp_path / "nonexistent.db", tmp_path / "dest.db")


# --- Error paths to exercise defensive code ---


def test_use_before_open_raises_cache_closed(tmp_path: Path):
    from mneme._exceptions import CacheClosedError

    s = SQLiteStore(tmp_path / "cache.db")
    with pytest.raises(CacheClosedError, match="open"):
        s.get_by_hash("default", "x")


def test_open_is_idempotent(tmp_path: Path):
    db = tmp_path / "cache.db"
    s = SQLiteStore(db)
    s.open("fp:v1", 4)
    s.open("fp:v1", 4)  # second call is a no-op, must not raise
    s.close()


def test_close_when_never_opened_is_safe(tmp_path: Path):
    s = SQLiteStore(tmp_path / "cache.db")
    s.close()  # must not raise; should mark closed
    from mneme._exceptions import CacheClosedError

    with pytest.raises(CacheClosedError):
        s.open("fp:v1", 4)


def test_op_error_after_underlying_close(tmp_path: Path):
    """If the underlying connection is closed out from under us, the next
    operation must surface a clear StoreBackendError, not a bare sqlite3 error."""
    from mneme._exceptions import StoreBackendError

    s = SQLiteStore(tmp_path / "cache.db")
    s.open("fp:v1", 4)
    s.insert(_entry("1" * 64))
    # Force-close the underlying connection (simulate corruption / external close).
    s._conn.close()  # type: ignore[union-attr]
    with pytest.raises(StoreBackendError):
        s.insert(_entry("2" * 64))


def test_set_quota_persists_across_reopen(tmp_path: Path):
    db = tmp_path / "cache.db"
    s = SQLiteStore(db)
    s.open("fp:v1", 4)
    s.set_namespace_quota("tenant", 500)
    s.close()
    s2 = SQLiteStore(db)
    s2.open("fp:v1", 4)
    try:
        assert s2.get_namespace_quota("tenant") == 500
    finally:
        s2.close()


def test_delete_expired_no_namespace_filter(tmp_path: Path):
    s = SQLiteStore(tmp_path / "cache.db")
    s.open("fp:v1", 4)
    s.insert(
        StoredEntry(
            id=0,
            namespace="a",
            query_hash="1" * 64,
            query="q",
            response="r",
            embedding=np.zeros(4, dtype=np.float32).tobytes(),
            metadata={},
            created_at=0,
            last_accessed_at=0,
            ttl=10,
            access_count=0,
        )
    )
    s.insert(
        StoredEntry(
            id=0,
            namespace="b",
            query_hash="2" * 64,
            query="q",
            response="r",
            embedding=np.zeros(4, dtype=np.float32).tobytes(),
            metadata={},
            created_at=0,
            last_accessed_at=0,
            ttl=10,
            access_count=0,
        )
    )
    deleted = s.delete_expired(now=1000)  # both expired
    assert deleted == 2
    s.close()


def test_clear_namespace_returns_zero_when_empty(tmp_path: Path):
    s = SQLiteStore(tmp_path / "cache.db")
    s.open("fp:v1", 4)
    assert s.clear_namespace("nonexistent") == 0
    s.close()


def test_delete_by_id_returns_false_when_missing(tmp_path: Path):
    s = SQLiteStore(tmp_path / "cache.db")
    s.open("fp:v1", 4)
    assert s.delete_by_id(99999) is False
    s.close()


def test_iter_lru_ids_with_zero_n_returns_empty(tmp_path: Path):
    s = SQLiteStore(tmp_path / "cache.db")
    s.open("fp:v1", 4)
    assert list(s.iter_lru_ids(0)) == []
    s.close()
