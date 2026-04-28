"""``RedisStore``-specific tests. Skipped unless MNEME_REDIS_URL is set or
RUN_REDIS_INTEGRATION=1 spins up a testcontainer."""

from __future__ import annotations

import time

import numpy as np
import pytest

# Skip the whole module if the optional extra is not installed.
pytest.importorskip("redis")

from mneme._exceptions import (
    CheckpointError,
    EmbedderDimensionError,
    EmbedderMismatchError,
)
from mneme._store_redis import RedisStore
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


def test_construct_with_url_and_prefix(redis_url: str, redis_prefix: str):
    s = RedisStore(url=redis_url, key_prefix=redis_prefix)
    s.open("fp:v1", 4)
    try:
        assert s.count() == 0
    finally:
        s.close()


def test_requires_exactly_one_of_url_or_client():
    with pytest.raises(ValueError, match="exactly one"):
        RedisStore()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="exactly one"):
        # Both supplied is also an error.
        RedisStore(url="redis://localhost:6379", client=object())  # type: ignore[arg-type]


def test_user_supplied_client_not_closed_by_store(redis_url: str, redis_prefix: str):
    """If the caller passes their own Redis client, RedisStore must not close it."""
    import redis

    client = redis.Redis.from_url(redis_url, decode_responses=False)
    try:
        s = RedisStore(client=client, key_prefix=redis_prefix)
        s.open("fp:v1", 4)
        s.close()
        # client should still be usable
        assert client.ping() is True
    finally:
        client.close()


def test_native_ttl_expires_keys(redis_url: str, redis_prefix: str):
    s = RedisStore(url=redis_url, key_prefix=redis_prefix, use_native_ttl=True)
    s.open("fp:v1", 4)
    try:
        # Insert an entry with a TTL that's already expired
        e = _entry("a" * 64, ttl=1)
        # Backdate created_at so EXPIREAT is in the past
        backdated = StoredEntry(
            id=0,
            namespace=e.namespace,
            query_hash=e.query_hash,
            query=e.query,
            response=e.response,
            embedding=e.embedding,
            metadata=e.metadata,
            created_at=int(time.time()) - 10,
            last_accessed_at=int(time.time()) - 10,
            ttl=1,
            access_count=0,
        )
        s.insert(backdated)
        # Redis-level TTL should have evicted the key already (EXPIREAT in past).
        # Allow a brief window for Redis to apply.
        time.sleep(0.1)
        assert s.get_by_hash("default", "a" * 64) is None
    finally:
        s.close()


def test_persistence_across_reopen_same_prefix(redis_url: str, redis_prefix: str):
    s1 = RedisStore(url=redis_url, key_prefix=redis_prefix)
    s1.open("fp:v1", 4)
    s1.insert(_entry("1" * 64))
    s1.close()

    s2 = RedisStore(url=redis_url, key_prefix=redis_prefix)
    s2.open("fp:v1", 4)
    try:
        assert s2.count() == 1
        assert s2.get_by_hash("default", "1" * 64) is not None
    finally:
        s2.close()
        # cleanup
        import redis

        client = redis.Redis.from_url(redis_url, decode_responses=False)
        try:
            for key in client.scan_iter(match=f"{redis_prefix}:*"):
                client.delete(key)
        finally:
            client.close()


def test_fingerprint_mismatch_on_reopen(redis_url: str, redis_prefix: str):
    s1 = RedisStore(url=redis_url, key_prefix=redis_prefix)
    s1.open("fp:original", 4)
    s1.close()
    s2 = RedisStore(url=redis_url, key_prefix=redis_prefix)
    try:
        with pytest.raises(EmbedderMismatchError, match="reembed"):
            s2.open("fp:different", 4)
    finally:
        s2.close()
        import redis

        client = redis.Redis.from_url(redis_url, decode_responses=False)
        try:
            for key in client.scan_iter(match=f"{redis_prefix}:*"):
                client.delete(key)
        finally:
            client.close()


def test_dim_mismatch_on_reopen(redis_url: str, redis_prefix: str):
    s1 = RedisStore(url=redis_url, key_prefix=redis_prefix)
    s1.open("fp:original", 4)
    s1.close()
    s2 = RedisStore(url=redis_url, key_prefix=redis_prefix)
    try:
        with pytest.raises(EmbedderDimensionError, match="reembed"):
            s2.open("fp:original", 8)
    finally:
        s2.close()
        import redis

        client = redis.Redis.from_url(redis_url, decode_responses=False)
        try:
            for key in client.scan_iter(match=f"{redis_prefix}:*"):
                client.delete(key)
        finally:
            client.close()


def test_snapshot_to_raises_checkpoint_error(redis_url: str, redis_prefix: str, tmp_path):
    s = RedisStore(url=redis_url, key_prefix=redis_prefix)
    s.open("fp:v1", 4)
    try:
        with pytest.raises(CheckpointError, match="redis-cli"):
            s.snapshot_to(tmp_path / "snap.rdb")
    finally:
        s.close()


def test_restore_from_raises_checkpoint_error(tmp_path):
    with pytest.raises(CheckpointError):
        RedisStore.restore_from(tmp_path / "src", tmp_path / "dst")


def test_invalid_url_raises_clear_error():
    from mneme._exceptions import StoreBackendError

    s = RedisStore(url="redis://nonexistent-host-xyz:9999", key_prefix="x")
    with pytest.raises(StoreBackendError, match="connect"):
        s.open("fp:v1", 4)
