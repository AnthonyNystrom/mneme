"""``RedisStore``: Redis-backed Store. Optional ``[redis]`` extra.

Reference implementation per PRD §8.13.3. Multi-host coordination is provided
by Redis itself: each process maintains its own in-memory index but reads the
authoritative entries from Redis. ``snapshot_to`` / ``restore_from`` raise
``CheckpointError`` — use ``redis-cli BGSAVE`` and the resulting RDB file
externally if you need backups.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._exceptions import (
    CacheClosedError,
    CheckpointError,
    EmbedderDimensionError,
    EmbedderMismatchError,
    StoreBackendError,
)
from ._migrations import CURRENT_SCHEMA_VERSION
from ._types import StoredEntry

if TYPE_CHECKING:
    import redis as _redis_t


def _import_redis() -> Any:
    try:
        import redis
    except ImportError as exc:  # pragma: no cover - extras-gated
        raise StoreBackendError(
            "RedisStore requires the optional 'redis' extra. Remediation: pip install mneme[redis]"
        ) from exc
    return redis


class RedisStore:
    """Redis-backed Store. Provide a connection ``url`` or a pre-built ``client``.

    Key layout under ``key_prefix``:

    - ``{p}:meta`` HASH — schema_meta (fingerprint, dim, schema_version)
    - ``{p}:next_id`` STRING — INCR-backed monotonic id
    - ``{p}:version`` STRING — INCR-backed version_counter
    - ``{p}:entry:{id}`` HASH — full entry fields
    - ``{p}:hash:{ns}:{query_hash}`` STRING — id (exact-lookup index)
    - ``{p}:lru:{ns}`` ZSET — member=id, score=last_accessed_at
    - ``{p}:by_id`` ZSET — member=id, score=id (for ``iter_since`` / ``iter_all``)
    - ``{p}:by_ttl`` ZSET — member=id, score=created_at+ttl (for ``delete_expired``)
    - ``{p}:quota:{ns}`` STRING — namespace quota
    - ``{p}:namespaces`` SET — known namespaces

    When ``use_native_ttl=True`` (default), the entry HASH and its dependent
    keys also receive Redis-level ``EXPIREAT``. Application-level TTL stored
    in the entry remains authoritative.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        client: _redis_t.Redis | None = None,
        key_prefix: str = "mneme",
        use_native_ttl: bool = True,
    ) -> None:
        if (url is None) == (client is None):
            raise ValueError(
                "RedisStore: provide exactly one of `url` or `client`. "
                "Remediation: pass a connection URL or a pre-built redis.Redis."
            )
        self._url = url
        self._client_user_supplied = client is not None
        self._client: _redis_t.Redis | None = client
        self._prefix = key_prefix
        self._use_native_ttl = use_native_ttl
        self._closed = False

    # --- key helpers ---

    def _k(self, *parts: str) -> str:
        return ":".join((self._prefix, *parts))

    # --- Lifecycle ---

    def open(self, embedder_fingerprint: str, embedder_dim: int) -> None:
        if self._closed:
            raise CacheClosedError("RedisStore was closed. Remediation: create a new instance.")
        if self._client is None:
            redis = _import_redis()
            try:
                self._client = redis.Redis.from_url(self._url, decode_responses=False)
                self._client.ping()
            except Exception as exc:
                raise StoreBackendError(
                    f"Failed to connect to Redis at {self._url!r}. "
                    f"Remediation: verify the URL and that Redis is running."
                ) from exc

        client = self._client
        meta_key = self._k("meta")
        existing_fp_b = client.hget(meta_key, "embedder_fingerprint")
        if existing_fp_b is None:
            client.hset(
                meta_key,
                mapping={
                    "embedder_fingerprint": embedder_fingerprint,
                    "embedder_dim": str(embedder_dim),
                    "schema_version": str(CURRENT_SCHEMA_VERSION),
                },
            )
        else:
            existing_fp = existing_fp_b.decode("utf-8")
            if existing_fp != embedder_fingerprint:
                raise EmbedderMismatchError(
                    f"Stored fingerprint {existing_fp!r} does not match "
                    f"supplied {embedder_fingerprint!r}. Remediation: open "
                    f"with the original embedder, or use "
                    f"mneme.tools.migrate.reembed() to migrate."
                )
            stored_dim_b = client.hget(meta_key, "embedder_dim")
            stored_dim = int(stored_dim_b.decode("utf-8")) if stored_dim_b else 0
            if stored_dim != embedder_dim:
                raise EmbedderDimensionError(
                    f"Stored embedder_dim={stored_dim} does not match "
                    f"supplied {embedder_dim}. Remediation: use reembed() "
                    f"to change dimension."
                )
        if not client.exists(self._k("version")):
            client.set(self._k("version"), 0)
        if not client.exists(self._k("next_id")):
            client.set(self._k("next_id"), 0)

    def close(self) -> None:
        if self._client is not None and not self._client_user_supplied:
            with contextlib.suppress(Exception):
                self._client.close()
        self._client = None
        self._closed = True

    def _client_or_fail(self) -> _redis_t.Redis:
        if self._closed:
            raise CacheClosedError("RedisStore is closed.")
        if self._client is None:
            raise CacheClosedError("RedisStore not opened. Remediation: call open() first.")
        return self._client

    # --- Read ---

    def _entry_from_hash(self, entry_id: int, raw: dict[bytes, bytes]) -> StoredEntry:
        if not raw:
            raise StoreBackendError(
                f"Redis entry {entry_id} hash is empty (race or corruption). "
                f"Remediation: re-open or re-warm the cache."
            )

        def _b(field: str) -> bytes:
            return raw[field.encode("utf-8")]

        return StoredEntry(
            id=entry_id,
            namespace=_b("namespace").decode("utf-8"),
            query_hash=_b("query_hash").decode("utf-8"),
            query=_b("query").decode("utf-8"),
            response=_b("response").decode("utf-8"),
            embedding=_b("embedding"),
            metadata=json.loads(_b("metadata").decode("utf-8")),
            created_at=int(_b("created_at")),
            last_accessed_at=int(_b("last_accessed_at")),
            ttl=None if _b("ttl") == b"" else int(_b("ttl")),
            access_count=int(_b("access_count")),
        )

    def get_by_hash(self, namespace: str, query_hash: str) -> StoredEntry | None:
        client = self._client_or_fail()
        id_b = client.get(self._k("hash", namespace, query_hash))
        if id_b is None:
            return None
        entry_id = int(id_b)
        raw = client.hgetall(self._k("entry", str(entry_id)))
        if not raw:
            return None
        return self._entry_from_hash(entry_id, raw)

    def get_by_id(self, id: int) -> StoredEntry | None:
        client = self._client_or_fail()
        raw = client.hgetall(self._k("entry", str(id)))
        if not raw:
            return None
        return self._entry_from_hash(id, raw)

    def count(self, namespace: str | None = None) -> int:
        client = self._client_or_fail()
        if namespace is None:
            return int(client.zcard(self._k("by_id")))
        return int(client.zcard(self._k("lru", namespace)))

    def list_namespaces(self) -> list[str]:
        client = self._client_or_fail()
        members = client.smembers(self._k("namespaces"))
        return sorted(m.decode("utf-8") for m in members)

    def iter_lru_ids(self, n: int, namespace: str | None = None) -> Iterator[int]:
        client = self._client_or_fail()
        if n <= 0:
            return iter(())
        if namespace is not None:
            ids = client.zrange(self._k("lru", namespace), 0, n - 1)
        else:
            # Aggregate across namespaces: collect from all and sort.
            all_namespaces = client.smembers(self._k("namespaces"))
            scored: list[tuple[int, float]] = []
            for ns_b in all_namespaces:
                ns = ns_b.decode("utf-8")
                pairs = client.zrange(self._k("lru", ns), 0, n - 1, withscores=True)
                for member, score in pairs:
                    scored.append((int(member), float(score)))
            scored.sort(key=lambda p: (p[1], p[0]))
            return iter([sid for sid, _ in scored[:n]])
        return iter([int(m) for m in ids])

    def _all_ids_sorted(self) -> list[int]:
        client = self._client_or_fail()
        members = client.zrange(self._k("by_id"), 0, -1)
        return [int(m) for m in members]

    def iter_all(self) -> Iterator[StoredEntry]:
        for sid in self._all_ids_sorted():
            entry = self.get_by_id(sid)
            if entry is not None:
                yield entry

    def iter_since(self, last_id: int) -> Iterator[StoredEntry]:
        client = self._client_or_fail()
        members = client.zrangebyscore(self._k("by_id"), f"({last_id}", "+inf")
        for m in members:
            entry = self.get_by_id(int(m))
            if entry is not None:
                yield entry

    # --- Write ---

    @staticmethod
    def _ttl_member(namespace: str, entry_id: int) -> str:
        """Compose the ``by_ttl`` ZSET member as ``ns|id`` so we can recover
        the namespace later even if the entry HASH has been TTL-evicted."""
        return f"{namespace}|{entry_id}"

    @staticmethod
    def _parse_ttl_member(member: bytes | str) -> tuple[str, int]:
        s = member.decode("utf-8") if isinstance(member, bytes) else member
        ns, _, sid = s.rpartition("|")
        return ns, int(sid)

    def insert(self, entry: StoredEntry) -> int:
        client = self._client_or_fail()
        hash_key = self._k("hash", entry.namespace, entry.query_hash)
        existing_id_b = client.get(hash_key)
        if existing_id_b is not None:
            new_id = int(existing_id_b)
        else:
            new_id = int(client.incr(self._k("next_id")))

        entry_key = self._k("entry", str(new_id))
        ttl_field = "" if entry.ttl is None else str(entry.ttl)

        pipe = client.pipeline(transaction=True)
        pipe.hset(
            entry_key,
            mapping={
                "namespace": entry.namespace,
                "query_hash": entry.query_hash,
                "query": entry.query,
                "response": entry.response,
                "embedding": entry.embedding,
                "metadata": json.dumps(entry.metadata),
                "created_at": str(entry.created_at),
                "last_accessed_at": str(entry.last_accessed_at),
                "ttl": ttl_field,
                "access_count": str(entry.access_count),
            },
        )
        pipe.set(hash_key, str(new_id))
        pipe.zadd(self._k("lru", entry.namespace), {str(new_id): entry.last_accessed_at})
        pipe.zadd(self._k("by_id"), {str(new_id): new_id})
        if entry.ttl is not None:
            ttl_member = self._ttl_member(entry.namespace, new_id)
            pipe.zadd(self._k("by_ttl"), {ttl_member: entry.created_at + entry.ttl})
            if self._use_native_ttl:
                # Apply EXPIREAT only to the entry HASH. Auxiliary indexes
                # (by_ttl, by_id, lru, hash mapping, namespaces SET) are
                # cleaned up by ``delete_expired`` / ``_delete_atomic``.
                pipe.expireat(entry_key, entry.created_at + entry.ttl)
        pipe.sadd(self._k("namespaces"), entry.namespace)
        pipe.incr(self._k("version"))
        pipe.execute()
        return new_id

    def update_access(self, id: int, now: int) -> None:
        client = self._client_or_fail()
        entry_key = self._k("entry", str(id))
        if not client.exists(entry_key):
            return
        ns_b = client.hget(entry_key, "namespace")
        if ns_b is None:
            return
        ns = ns_b.decode("utf-8")
        pipe = client.pipeline(transaction=True)
        pipe.hset(entry_key, "last_accessed_at", str(now))
        pipe.hincrby(entry_key, "access_count", 1)
        pipe.zadd(self._k("lru", ns), {str(id): now})
        pipe.execute()

    def _delete_atomic(self, entry_id: int, *, namespace_hint: str | None = None) -> bool:
        """Remove an entry across all auxiliary indexes.

        If the entry HASH is still present, namespace and query_hash are read
        from it. If the HASH has been TTL-evicted by Redis, ``namespace_hint``
        (if supplied) is used to clean up ``lru:{ns}`` and the namespaces SET;
        the ``hash:{ns}:{query_hash}`` mapping leaks but is harmless (its key
        also expired via EXPIREAT) and is overwritten on any future insert
        with the same query_hash.
        """
        client = self._client_or_fail()
        entry_key = self._k("entry", str(entry_id))
        raw = client.hgetall(entry_key)
        ns = raw[b"namespace"].decode("utf-8") if raw else namespace_hint
        if not raw and ns is None:
            # Entry HASH gone and we have no namespace hint — best-effort
            # cleanup of namespace-agnostic indexes only.
            removed = client.zrem(self._k("by_id"), str(entry_id)) > 0
            return removed

        pipe = client.pipeline(transaction=True)
        pipe.delete(entry_key)
        if raw:
            h = raw[b"query_hash"].decode("utf-8")
            pipe.delete(self._k("hash", ns, h))
        if ns is not None:
            pipe.zrem(self._k("lru", ns), str(entry_id))
            pipe.zrem(self._k("by_ttl"), self._ttl_member(ns, entry_id))
        pipe.zrem(self._k("by_id"), str(entry_id))
        pipe.incr(self._k("version"))
        pipe.execute()
        if ns is not None and client.zcard(self._k("lru", ns)) == 0:
            client.srem(self._k("namespaces"), ns)
        return True

    def delete_by_id(self, id: int) -> bool:
        return self._delete_atomic(id)

    def delete_expired(self, now: int, namespace: str | None = None) -> int:
        client = self._client_or_fail()
        candidates = client.zrangebyscore(self._k("by_ttl"), "-inf", now)
        deleted = 0
        for member in candidates:
            ns, sid = self._parse_ttl_member(member)
            if namespace is not None and ns != namespace:
                continue
            if self._delete_atomic(sid, namespace_hint=ns):
                deleted += 1
        return deleted

    def clear_namespace(self, namespace: str) -> int:
        client = self._client_or_fail()
        ids = [int(m) for m in client.zrange(self._k("lru", namespace), 0, -1)]
        for sid in ids:
            self._delete_atomic(sid, namespace_hint=namespace)
        return len(ids)

    # --- Quotas ---

    def set_namespace_quota(self, namespace: str, max_entries: int) -> None:
        self._client_or_fail().set(self._k("quota", namespace), str(max_entries))

    def get_namespace_quota(self, namespace: str) -> int | None:
        v = self._client_or_fail().get(self._k("quota", namespace))
        return None if v is None else int(v)

    # --- Coordination ---

    def read_version_counter(self) -> int:
        v = self._client_or_fail().get(self._k("version"))
        return 0 if v is None else int(v)

    def read_meta(self, key: str) -> str | None:
        v = self._client_or_fail().hget(self._k("meta"), key)
        return None if v is None else v.decode("utf-8")

    def write_meta(self, key: str, value: str) -> None:
        self._client_or_fail().hset(self._k("meta"), key, value)

    # --- Health ---

    def integrity_check(self) -> bool:
        try:
            self._client_or_fail().ping()
        except Exception:
            return False
        return True

    # --- Backup ---

    def snapshot_to(self, dest_path: str | Path) -> None:
        del dest_path
        raise CheckpointError(
            "RedisStore.snapshot_to is not implemented in v1; the library does "
            "not subprocess to redis-cli. Remediation: run `redis-cli "
            "--rdb path/to/dump.rdb` externally and copy the file. The "
            "Cache.dumps() entry point is unaffected for SQLiteStore."
        )

    @classmethod
    def restore_from(cls, source_path: str | Path, dest_path: str | Path) -> RedisStore:
        del source_path, dest_path
        raise CheckpointError(
            "RedisStore.restore_from is not implemented in v1. Remediation: "
            "load the RDB into Redis externally, then construct RedisStore "
            "pointing at it."
        )


__all__ = ["RedisStore"]
