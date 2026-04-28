"""``PostgresStore``-specific tests. Skipped unless MNEME_PG_URL is set or
RUN_POSTGRES_INTEGRATION=1 spins up a testcontainer."""

from __future__ import annotations

import time

import numpy as np
import pytest

# Skip the whole module if the optional extra is not installed.
pytest.importorskip("psycopg")

from mneme._exceptions import (
    CheckpointError,
    EmbedderDimensionError,
    EmbedderMismatchError,
)
from mneme._store_postgres import PostgresStore
from mneme._types import StoredEntry


def _entry(h: str, ns: str = "default", ttl: int | None = None) -> StoredEntry:
    return StoredEntry(
        id=0,
        namespace=ns,
        query_hash=h,
        query="q",
        response="r",
        embedding=np.zeros(4, dtype=np.float32).tobytes(),
        metadata={},
        created_at=int(time.time()),
        last_accessed_at=int(time.time()),
        ttl=ttl,
        access_count=0,
    )


def _drop_schema(dsn: str, schema: str) -> None:
    import psycopg

    with psycopg.connect(dsn) as conn, conn.transaction():
        conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


def test_construct_with_dsn_and_schema(pg_url: str, pg_schema: str):
    s = PostgresStore(dsn=pg_url, schema=pg_schema)
    s.open("fp:v1", 4)
    try:
        assert s.count() == 0
    finally:
        s.close()
        _drop_schema(pg_url, pg_schema)


def test_requires_exactly_one_connection_source():
    with pytest.raises(ValueError, match="exactly one"):
        PostgresStore()  # nothing supplied
    with pytest.raises(ValueError, match="exactly one"):
        PostgresStore(dsn="x", connection=object(), pool=object())  # too many


def test_invalid_schema_name_rejected():
    with pytest.raises(ValueError, match="alphanumeric"):
        PostgresStore(dsn="postgresql://x", schema="bad-schema; DROP TABLE")
    with pytest.raises(ValueError, match="alphanumeric"):
        PostgresStore(dsn="postgresql://x", schema="")


def test_persistence_across_reopen_same_schema(pg_url: str, pg_schema: str):
    s1 = PostgresStore(dsn=pg_url, schema=pg_schema)
    s1.open("fp:v1", 4)
    try:
        s1.insert(_entry("1" * 64))
    finally:
        s1.close()

    s2 = PostgresStore(dsn=pg_url, schema=pg_schema)
    s2.open("fp:v1", 4)
    try:
        assert s2.count() == 1
        assert s2.get_by_hash("default", "1" * 64) is not None
    finally:
        s2.close()
        _drop_schema(pg_url, pg_schema)


def test_fingerprint_mismatch_on_reopen(pg_url: str, pg_schema: str):
    s1 = PostgresStore(dsn=pg_url, schema=pg_schema)
    s1.open("fp:original", 4)
    s1.close()
    s2 = PostgresStore(dsn=pg_url, schema=pg_schema)
    try:
        with pytest.raises(EmbedderMismatchError, match="reembed"):
            s2.open("fp:different", 4)
    finally:
        s2.close()
        _drop_schema(pg_url, pg_schema)


def test_dim_mismatch_on_reopen(pg_url: str, pg_schema: str):
    s1 = PostgresStore(dsn=pg_url, schema=pg_schema)
    s1.open("fp:original", 4)
    s1.close()
    s2 = PostgresStore(dsn=pg_url, schema=pg_schema)
    try:
        with pytest.raises(EmbedderDimensionError, match="reembed"):
            s2.open("fp:original", 8)
    finally:
        s2.close()
        _drop_schema(pg_url, pg_schema)


def test_jsonb_metadata_round_trip(pg_url: str, pg_schema: str):
    s = PostgresStore(dsn=pg_url, schema=pg_schema)
    s.open("fp:v1", 4)
    try:
        nested = {"a": 1, "b": [1, 2, 3], "c": {"d": True}}
        e = StoredEntry(
            id=0,
            namespace="default",
            query_hash="m" * 64,
            query="q",
            response="r",
            embedding=np.zeros(4, dtype=np.float32).tobytes(),
            metadata=nested,
            created_at=0,
            last_accessed_at=0,
            ttl=None,
            access_count=0,
        )
        s.insert(e)
        fetched = s.get_by_hash("default", "m" * 64)
        assert fetched is not None
        assert fetched.metadata == nested
    finally:
        s.close()
        _drop_schema(pg_url, pg_schema)


def test_snapshot_to_raises_checkpoint_error(pg_url: str, pg_schema: str, tmp_path):
    s = PostgresStore(dsn=pg_url, schema=pg_schema)
    s.open("fp:v1", 4)
    try:
        with pytest.raises(CheckpointError, match="pg_dump"):
            s.snapshot_to(tmp_path / "snap")
    finally:
        s.close()
        _drop_schema(pg_url, pg_schema)


def test_restore_from_raises_checkpoint_error(tmp_path):
    with pytest.raises(CheckpointError):
        PostgresStore.restore_from(tmp_path / "src", tmp_path / "dst")


def test_invalid_dsn_raises_clear_error():
    from mneme._exceptions import StoreBackendError

    s = PostgresStore(
        dsn="postgresql://nonexistent:9999/nope",
        schema="mneme_test",
    )
    with pytest.raises(StoreBackendError, match="connect"):
        s.open("fp:v1", 4)


def test_pool_supplied_externally_not_closed(pg_url: str, pg_schema: str):
    """When the caller supplies a pool, PostgresStore must not close the pool."""
    import psycopg_pool

    pool = psycopg_pool.ConnectionPool(pg_url, min_size=1, max_size=2, open=True)
    try:
        s = PostgresStore(pool=pool, schema=pg_schema)
        s.open("fp:v1", 4)
        s.close()
        # pool should still be usable
        with pool.connection() as conn:
            cur = conn.execute("SELECT 1")
            assert cur.fetchone()[0] == 1
    finally:
        pool.close()
        _drop_schema(pg_url, pg_schema)


def test_close_when_never_opened_is_safe(pg_url: str, pg_schema: str):
    s = PostgresStore(dsn=pg_url, schema=pg_schema)
    s.close()  # connection was never opened
    from mneme._exceptions import CacheClosedError

    with pytest.raises(CacheClosedError):
        s.open("fp:v1", 4)


def test_use_before_open_raises(pg_url: str, pg_schema: str):
    from mneme._exceptions import CacheClosedError

    s = PostgresStore(dsn=pg_url, schema=pg_schema)
    with pytest.raises(CacheClosedError, match="open"):
        s.get_by_hash("default", "x")


def test_iter_lru_ids_zero_n_returns_empty(pg_url: str, pg_schema: str):
    s = PostgresStore(dsn=pg_url, schema=pg_schema)
    s.open("fp:v1", 4)
    try:
        assert list(s.iter_lru_ids(0)) == []
    finally:
        s.close()
        _drop_schema(pg_url, pg_schema)


def test_clear_namespace_empty_returns_zero(pg_url: str, pg_schema: str):
    s = PostgresStore(dsn=pg_url, schema=pg_schema)
    s.open("fp:v1", 4)
    try:
        assert s.clear_namespace("nonexistent") == 0
    finally:
        s.close()
        _drop_schema(pg_url, pg_schema)


def test_delete_by_id_missing_returns_false(pg_url: str, pg_schema: str):
    s = PostgresStore(dsn=pg_url, schema=pg_schema)
    s.open("fp:v1", 4)
    try:
        assert s.delete_by_id(99999) is False
    finally:
        s.close()
        _drop_schema(pg_url, pg_schema)


def test_set_quota_persists_across_reopen(pg_url: str, pg_schema: str):
    s = PostgresStore(dsn=pg_url, schema=pg_schema)
    s.open("fp:v1", 4)
    s.set_namespace_quota("tenant", 500)
    s.close()
    s2 = PostgresStore(dsn=pg_url, schema=pg_schema)
    s2.open("fp:v1", 4)
    try:
        assert s2.get_namespace_quota("tenant") == 500
    finally:
        s2.close()
        _drop_schema(pg_url, pg_schema)


def test_op_error_after_underlying_close(pg_url: str, pg_schema: str):
    """If the connection is closed underneath us, the next op surfaces a clear
    StoreBackendError rather than a bare psycopg error."""
    from mneme._exceptions import StoreBackendError

    s = PostgresStore(dsn=pg_url, schema=pg_schema)
    s.open("fp:v1", 4)
    try:
        s.insert(_entry("1" * 64))
        s._conn.close()  # type: ignore[union-attr]
        with pytest.raises(StoreBackendError):
            s.insert(_entry("2" * 64))
    finally:
        # Drop schema using a fresh connection since ours is closed.
        _drop_schema(pg_url, pg_schema)
