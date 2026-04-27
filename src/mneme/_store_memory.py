"""``MemoryStore``: dict-backed Store. No persistence; single-process multi-thread safe.

Use for tests, ephemeral runtimes (Lambda), local development, or scenarios
where SQLite overhead is unwanted. Data is lost on close. ``dumps``/``loads``
raise ``CheckpointError``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

from ._exceptions import (
    CacheClosedError,
    CheckpointError,
    EmbedderDimensionError,
    EmbedderMismatchError,
    StoreBackendError,
)
from ._migrations import CURRENT_SCHEMA_VERSION
from ._types import StoredEntry


class MemoryStore:
    """In-memory Store implementation. Thread-safe via ``threading.RLock``."""

    def __init__(self, *, max_entries: int | None = None) -> None:
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self._closed = False
        self._opened = False
        self._entries: dict[int, StoredEntry] = {}
        self._hash_index: dict[tuple[str, str], int] = {}
        self._namespaces: set[str] = set()
        self._quotas: dict[str, int] = {}
        self._meta: dict[str, str] = {}
        self._next_id: int = 1
        self._version_counter: int = 0

    # --- Lifecycle ---

    def open(self, embedder_fingerprint: str, embedder_dim: int) -> None:
        with self._lock:
            self._check_not_closed()
            existing_fp = self._meta.get("embedder_fingerprint")
            if existing_fp is None:
                self._meta["embedder_fingerprint"] = embedder_fingerprint
                self._meta["embedder_dim"] = str(embedder_dim)
                self._meta["schema_version"] = str(CURRENT_SCHEMA_VERSION)
            else:
                if existing_fp != embedder_fingerprint:
                    raise EmbedderMismatchError(
                        f"Stored fingerprint {existing_fp!r} does not match "
                        f"supplied {embedder_fingerprint!r}. Remediation: open "
                        f"with the original embedder, or use "
                        f"mneme.tools.migrate.reembed() to migrate."
                    )
                stored_dim = int(self._meta.get("embedder_dim", "0"))
                if stored_dim != embedder_dim:
                    raise EmbedderDimensionError(
                        f"Stored embedder_dim={stored_dim} does not match "
                        f"supplied {embedder_dim}. Remediation: use reembed() "
                        f"to change dimension."
                    )
            self._opened = True

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._opened = False

    def _check_not_closed(self) -> None:
        if self._closed:
            raise CacheClosedError("MemoryStore is closed. Remediation: create a new instance.")

    # --- Read ---

    def get_by_hash(self, namespace: str, query_hash: str) -> StoredEntry | None:
        with self._lock:
            self._check_not_closed()
            entry_id = self._hash_index.get((namespace, query_hash))
            if entry_id is None:
                return None
            return self._entries[entry_id]

    def get_by_id(self, id: int) -> StoredEntry | None:
        with self._lock:
            self._check_not_closed()
            return self._entries.get(id)

    def count(self, namespace: str | None = None) -> int:
        with self._lock:
            self._check_not_closed()
            if namespace is None:
                return len(self._entries)
            return sum(1 for e in self._entries.values() if e.namespace == namespace)

    def list_namespaces(self) -> list[str]:
        with self._lock:
            self._check_not_closed()
            return sorted(self._namespaces)

    def iter_lru_ids(self, n: int, namespace: str | None = None) -> Iterator[int]:
        with self._lock:
            self._check_not_closed()
            entries = list(self._entries.values())
            if namespace is not None:
                entries = [e for e in entries if e.namespace == namespace]
            entries.sort(key=lambda e: (e.last_accessed_at, e.id))
            return iter([e.id for e in entries[: max(0, n)]])

    def iter_all(self) -> Iterator[StoredEntry]:
        with self._lock:
            self._check_not_closed()
            return iter(sorted(self._entries.values(), key=lambda e: e.id))

    def iter_since(self, last_id: int) -> Iterator[StoredEntry]:
        with self._lock:
            self._check_not_closed()
            return iter(
                sorted(
                    (e for e in self._entries.values() if e.id > last_id),
                    key=lambda e: e.id,
                )
            )

    # --- Write ---

    def insert(self, entry: StoredEntry) -> int:
        with self._lock:
            self._check_not_closed()
            existing = self._hash_index.get((entry.namespace, entry.query_hash))
            if existing is None and (
                self._max_entries is not None and len(self._entries) >= self._max_entries
            ):
                raise StoreBackendError(
                    f"MemoryStore is at capacity (max_entries={self._max_entries}). "
                    f"Remediation: increase max_entries, evict entries, or use "
                    f"a persistent Store backend."
                )
            new_id = existing if existing is not None else self._next_id
            if existing is None:
                self._next_id += 1
            new_entry = StoredEntry(
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
            self._entries[new_id] = new_entry
            self._hash_index[(entry.namespace, entry.query_hash)] = new_id
            self._namespaces.add(entry.namespace)
            self._version_counter += 1
            return new_id

    def update_access(self, id: int, now: int) -> None:
        with self._lock:
            self._check_not_closed()
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
            self._check_not_closed()
            return self._delete_by_id_locked(id)

    def _delete_by_id_locked(self, id: int) -> bool:
        entry = self._entries.pop(id, None)
        if entry is None:
            return False
        del self._hash_index[(entry.namespace, entry.query_hash)]
        if not any(e.namespace == entry.namespace for e in self._entries.values()):
            self._namespaces.discard(entry.namespace)
        self._version_counter += 1
        return True

    def delete_expired(self, now: int, namespace: str | None = None) -> int:
        with self._lock:
            self._check_not_closed()
            ids: list[int] = []
            for entry in self._entries.values():
                if entry.ttl is None:
                    continue
                if namespace is not None and entry.namespace != namespace:
                    continue
                if entry.created_at + entry.ttl <= now:
                    ids.append(entry.id)
            for id_ in ids:
                self._delete_by_id_locked(id_)
            return len(ids)

    def clear_namespace(self, namespace: str) -> int:
        with self._lock:
            self._check_not_closed()
            ids = [eid for eid, e in self._entries.items() if e.namespace == namespace]
            for id_ in ids:
                self._delete_by_id_locked(id_)
            return len(ids)

    # --- Quotas ---

    def set_namespace_quota(self, namespace: str, max_entries: int) -> None:
        with self._lock:
            self._check_not_closed()
            self._quotas[namespace] = max_entries

    def get_namespace_quota(self, namespace: str) -> int | None:
        with self._lock:
            self._check_not_closed()
            return self._quotas.get(namespace)

    # --- Coordination ---

    def read_version_counter(self) -> int:
        with self._lock:
            self._check_not_closed()
            return self._version_counter

    def read_meta(self, key: str) -> str | None:
        with self._lock:
            self._check_not_closed()
            return self._meta.get(key)

    def write_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._check_not_closed()
            self._meta[key] = value

    # --- Health ---

    def integrity_check(self) -> bool:
        with self._lock:
            self._check_not_closed()
            for (ns, h), entry_id in self._hash_index.items():
                entry = self._entries.get(entry_id)
                if entry is None or entry.namespace != ns or entry.query_hash != h:
                    return False
            return True

    # --- Backup (unsupported for MemoryStore) ---

    def snapshot_to(self, dest_path: str | Path) -> None:
        del dest_path
        raise CheckpointError(
            "MemoryStore does not support snapshots. Remediation: use SQLiteStore "
            "or another persistent Store backend for checkpointing."
        )

    @classmethod
    def restore_from(cls, source_path: str | Path, dest_path: str | Path) -> MemoryStore:
        del source_path, dest_path
        raise CheckpointError(
            "MemoryStore does not support restore. Remediation: use SQLiteStore "
            "or another persistent Store backend for checkpointing."
        )


__all__ = ["MemoryStore"]
