"""Reference custom Store impl exposed to the conformance battery.

Demonstrates that the ``Store`` Protocol is implementable without using the
shipped ``MemoryStore`` or ``SQLiteStore`` base. Exists only for tests.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

from mneme._exceptions import (
    CacheClosedError,
    CheckpointError,
    EmbedderDimensionError,
    EmbedderMismatchError,
)
from mneme._types import StoredEntry


class InMemoryStore:
    """Pure-Python reference Store. Independent implementation from MemoryStore."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._closed = False
        self._entries: dict[int, StoredEntry] = {}
        self._namespaces: set[str] = set()
        self._quotas: dict[str, int] = {}
        self._meta: dict[str, str] = {}
        self._next_id = 1
        self._version_counter = 0

    def _check(self) -> None:
        if self._closed:
            raise CacheClosedError("InMemoryStore is closed.")

    def open(self, embedder_fingerprint: str, embedder_dim: int) -> None:
        with self._lock:
            self._check()
            existing_fp = self._meta.get("embedder_fingerprint")
            if existing_fp is None:
                self._meta["embedder_fingerprint"] = embedder_fingerprint
                self._meta["embedder_dim"] = str(embedder_dim)
            else:
                if existing_fp != embedder_fingerprint:
                    raise EmbedderMismatchError(
                        f"fingerprint mismatch: {existing_fp!r} vs "
                        f"{embedder_fingerprint!r}. Remediation: reembed."
                    )
                if int(self._meta["embedder_dim"]) != embedder_dim:
                    raise EmbedderDimensionError("dim mismatch. Remediation: reembed.")

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def get_by_hash(self, namespace: str, query_hash: str) -> StoredEntry | None:
        with self._lock:
            self._check()
            for entry in self._entries.values():
                if entry.namespace == namespace and entry.query_hash == query_hash:
                    return entry
            return None

    def get_by_id(self, id: int) -> StoredEntry | None:
        with self._lock:
            self._check()
            return self._entries.get(id)

    def count(self, namespace: str | None = None) -> int:
        with self._lock:
            self._check()
            if namespace is None:
                return len(self._entries)
            return sum(1 for e in self._entries.values() if e.namespace == namespace)

    def list_namespaces(self) -> list[str]:
        with self._lock:
            self._check()
            return sorted(self._namespaces)

    def iter_lru_ids(self, n: int, namespace: str | None = None) -> Iterator[int]:
        with self._lock:
            self._check()
            entries = list(self._entries.values())
            if namespace is not None:
                entries = [e for e in entries if e.namespace == namespace]
            entries.sort(key=lambda e: (e.last_accessed_at, e.id))
            return iter([e.id for e in entries[: max(0, n)]])

    def iter_all(self) -> Iterator[StoredEntry]:
        with self._lock:
            self._check()
            return iter(sorted(self._entries.values(), key=lambda e: e.id))

    def iter_since(self, last_id: int) -> Iterator[StoredEntry]:
        with self._lock:
            self._check()
            return iter(
                sorted(
                    (e for e in self._entries.values() if e.id > last_id),
                    key=lambda e: e.id,
                )
            )

    def insert(self, entry: StoredEntry) -> int:
        with self._lock:
            self._check()
            existing_id = None
            for eid, e in self._entries.items():
                if e.namespace == entry.namespace and e.query_hash == entry.query_hash:
                    existing_id = eid
                    break
            new_id = existing_id if existing_id is not None else self._next_id
            if existing_id is None:
                self._next_id += 1
            self._entries[new_id] = StoredEntry(
                id=new_id,
                namespace=entry.namespace,
                query_hash=entry.query_hash,
                query=entry.query,
                response=entry.response,
                embedding=entry.embedding,
                metadata=deepcopy(entry.metadata),
                created_at=entry.created_at,
                last_accessed_at=entry.last_accessed_at,
                ttl=entry.ttl,
                access_count=entry.access_count,
            )
            self._namespaces.add(entry.namespace)
            self._version_counter += 1
            return new_id

    def update_access(self, id: int, now: int) -> None:
        with self._lock:
            self._check()
            entry = self._entries.get(id)
            if entry is None:
                return
            self._entries[id] = StoredEntry(
                id=entry.id,
                namespace=entry.namespace,
                query_hash=entry.query_hash,
                query=entry.query,
                response=entry.response,
                embedding=entry.embedding,
                metadata=entry.metadata,
                created_at=entry.created_at,
                last_accessed_at=now,
                ttl=entry.ttl,
                access_count=entry.access_count + 1,
            )

    def delete_by_id(self, id: int) -> bool:
        with self._lock:
            self._check()
            entry = self._entries.pop(id, None)
            if entry is None:
                return False
            if not any(e.namespace == entry.namespace for e in self._entries.values()):
                self._namespaces.discard(entry.namespace)
            self._version_counter += 1
            return True

    def delete_expired(self, now: int, namespace: str | None = None) -> int:
        with self._lock:
            self._check()
            ids = []
            for entry in list(self._entries.values()):
                if entry.ttl is None:
                    continue
                if namespace is not None and entry.namespace != namespace:
                    continue
                if entry.created_at + entry.ttl <= now:
                    ids.append(entry.id)
            for eid in ids:
                e = self._entries.pop(eid, None)
                if e and not any(x.namespace == e.namespace for x in self._entries.values()):
                    self._namespaces.discard(e.namespace)
            if ids:
                self._version_counter += 1
            return len(ids)

    def clear_namespace(self, namespace: str) -> int:
        with self._lock:
            self._check()
            ids = [eid for eid, e in self._entries.items() if e.namespace == namespace]
            for eid in ids:
                self._entries.pop(eid, None)
            self._namespaces.discard(namespace)
            if ids:
                self._version_counter += 1
            return len(ids)

    def set_namespace_quota(self, namespace: str, max_entries: int) -> None:
        with self._lock:
            self._check()
            self._quotas[namespace] = max_entries

    def get_namespace_quota(self, namespace: str) -> int | None:
        with self._lock:
            self._check()
            return self._quotas.get(namespace)

    def read_version_counter(self) -> int:
        with self._lock:
            self._check()
            return self._version_counter

    def read_meta(self, key: str) -> str | None:
        with self._lock:
            self._check()
            return self._meta.get(key)

    def write_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._check()
            self._meta[key] = value

    def integrity_check(self) -> bool:
        with self._lock:
            self._check()
            return True

    def snapshot_to(self, dest_path: str | Path) -> None:
        del dest_path
        raise CheckpointError("InMemoryStore: snapshots not supported.")

    @classmethod
    def restore_from(cls, source_path: str | Path, dest_path: str | Path) -> InMemoryStore:
        del source_path, dest_path
        raise CheckpointError("InMemoryStore: restore not supported.")


__all__ = ["InMemoryStore"]
