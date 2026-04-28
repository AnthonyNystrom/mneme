"""Implement a custom ``Store`` backend.

The ``Store`` protocol from ``mneme._types`` is the contract every backend
must satisfy. The included MemoryStore / SQLiteStore / Redis / Postgres /
DynamoDB stores all run against the same conformance battery in
``tests/test_store_protocol_compliance.py``; the same battery is the
yardstick for any custom backend you write.

This example shows a minimal *append-only* DictStore wrapping a Python
dict — useful as scaffolding when you're prototyping a new backend. It is
NOT production-ready: no version_counter atomicity, no namespace quotas,
no integrity checks.

Run:
    python examples/custom_store.py
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from mneme import (
    CacheClosedError,
    CheckpointError,
    EmbedderDimensionError,
    EmbedderMismatchError,
    SemanticCache,
    StoredEntry,
)


class DictStore:
    """Tiny ``Store`` impl. Persists nothing; loses state on process exit."""

    def __init__(self) -> None:
        self._entries: dict[int, StoredEntry] = {}
        self._next_id: int = 1
        self._version: int = 0
        self._meta: dict[str, str] = {}
        self._fp: str | None = None
        self._dim: int | None = None
        self._quotas: dict[str, int] = {}
        self._closed: bool = False

    # --- Lifecycle ---

    def open(self, embedder_fingerprint: str, embedder_dim: int) -> None:
        if self._closed:
            raise CacheClosedError("DictStore was closed.")
        if self._fp is None:
            self._fp = embedder_fingerprint
            self._dim = embedder_dim
        else:
            if self._fp != embedder_fingerprint:
                raise EmbedderMismatchError(
                    f"stored fp {self._fp!r} != supplied {embedder_fingerprint!r}"
                )
            if self._dim != embedder_dim:
                raise EmbedderDimensionError(
                    f"stored dim {self._dim} != supplied {embedder_dim}"
                )

    def close(self) -> None:
        self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise CacheClosedError("DictStore is closed.")

    # --- Read ---

    def get_by_hash(self, namespace: str, query_hash: str) -> StoredEntry | None:
        self._check_open()
        for entry in self._entries.values():
            if entry.namespace == namespace and entry.query_hash == query_hash:
                return entry
        return None

    def get_by_id(self, id: int) -> StoredEntry | None:
        self._check_open()
        return self._entries.get(id)

    def count(self, namespace: str | None = None) -> int:
        self._check_open()
        if namespace is None:
            return len(self._entries)
        return sum(1 for e in self._entries.values() if e.namespace == namespace)

    def list_namespaces(self) -> list[str]:
        self._check_open()
        return sorted({e.namespace for e in self._entries.values()})

    def iter_lru_ids(
        self, n: int, namespace: str | None = None
    ) -> Iterable[int]:
        self._check_open()
        candidates = [
            e for e in self._entries.values()
            if namespace is None or e.namespace == namespace
        ]
        candidates.sort(key=lambda e: (e.last_accessed_at, e.id))
        return iter([e.id for e in candidates[:n]])

    def iter_all(self) -> Iterable[StoredEntry]:
        self._check_open()
        return iter(sorted(self._entries.values(), key=lambda e: e.id))

    def iter_since(self, last_id: int) -> Iterable[StoredEntry]:
        self._check_open()
        return iter(e for e in self.iter_all() if e.id > last_id)

    # --- Write ---

    def insert(self, entry: StoredEntry) -> int:
        self._check_open()
        existing = self.get_by_hash(entry.namespace, entry.query_hash)
        if existing is not None:
            row_id = existing.id
        else:
            row_id = self._next_id
            self._next_id += 1
        # Frozen dataclass; rebuild with the assigned id.
        self._entries[row_id] = StoredEntry(
            id=row_id,
            namespace=entry.namespace,
            query_hash=entry.query_hash,
            query=entry.query,
            response=entry.response,
            embedding=entry.embedding,
            metadata=entry.metadata,
            created_at=entry.created_at,
            last_accessed_at=entry.last_accessed_at,
            ttl=entry.ttl,
            access_count=entry.access_count,
        )
        self._version += 1
        return row_id

    def update_access(self, id: int, now: int) -> None:
        self._check_open()
        existing = self._entries.get(id)
        if existing is None:
            return
        self._entries[id] = StoredEntry(
            id=existing.id,
            namespace=existing.namespace,
            query_hash=existing.query_hash,
            query=existing.query,
            response=existing.response,
            embedding=existing.embedding,
            metadata=existing.metadata,
            created_at=existing.created_at,
            last_accessed_at=now,
            ttl=existing.ttl,
            access_count=existing.access_count + 1,
        )
        self._version += 1

    def delete_by_id(self, id: int) -> bool:
        self._check_open()
        if id in self._entries:
            del self._entries[id]
            self._version += 1
            return True
        return False

    def delete_expired(self, now: int, namespace: str | None = None) -> int:
        self._check_open()
        to_delete = [
            e.id for e in self._entries.values()
            if e.ttl is not None and e.created_at + e.ttl <= now
            and (namespace is None or e.namespace == namespace)
        ]
        for id_ in to_delete:
            del self._entries[id_]
        if to_delete:
            self._version += 1
        return len(to_delete)

    def clear_namespace(self, namespace: str) -> int:
        self._check_open()
        to_delete = [
            id_ for id_, e in self._entries.items() if e.namespace == namespace
        ]
        for id_ in to_delete:
            del self._entries[id_]
        if to_delete:
            self._version += 1
        return len(to_delete)

    # --- Quotas ---

    def set_namespace_quota(self, namespace: str, max_entries: int) -> None:
        self._check_open()
        self._quotas[namespace] = max_entries

    def get_namespace_quota(self, namespace: str) -> int | None:
        self._check_open()
        return self._quotas.get(namespace)

    # --- Coordination ---

    def read_version_counter(self) -> int:
        self._check_open()
        return self._version

    def read_meta(self, key: str) -> str | None:
        self._check_open()
        return self._meta.get(key)

    def write_meta(self, key: str, value: str) -> None:
        self._check_open()
        self._meta[key] = value

    # --- Health ---

    def integrity_check(self) -> bool:
        return not self._closed

    # --- Backup ---

    def snapshot_to(self, dest_path: str | Path) -> None:
        del dest_path
        raise CheckpointError(
            "DictStore.snapshot_to is not implemented. Remediation: pickle "
            "the dict yourself, or use SQLiteStore for built-in snapshots."
        )

    @classmethod
    def restore_from(cls, source_path: str | Path, dest_path: str | Path) -> DictStore:
        del source_path, dest_path
        raise CheckpointError("DictStore.restore_from is not implemented.")


# Demo using the custom store with a real cache.


class ToyEmbedder:
    dim = 8
    fingerprint = "toy:demo:v1"

    def embed(self, text: str):  # type: ignore[no-untyped-def]
        import hashlib

        import numpy as np

        digest = hashlib.sha256(text.encode("utf-8")).digest()[: self.dim]
        v = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) - 128.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


def main() -> None:
    store = DictStore()
    with SemanticCache(store=store, embedder=ToyEmbedder()) as cache:
        cache.put("hello", "world")
        hit = cache.get("hello")
        assert hit is not None
        assert hit.response == "world"
        print(f"DictStore round-trip OK: {hit.response!r}")
        print(f"version_counter: {store.read_version_counter()}")


if __name__ == "__main__":
    main()
