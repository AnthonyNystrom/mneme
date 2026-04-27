"""Public dataclasses, Protocols, and type aliases for mneme."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

# --- Type aliases ---

VectorDtype = Literal["float32", "float16", "int8"]
HitLayer = Literal["exact", "semantic"]
IndexBackend = Literal["numpy", "hnsw", "auto"]
MultiProcessMode = Literal["single", "stale-tolerant", "mmap-shared"]

ConfidenceFn = Callable[[float, int, dict[str, Any]], float]
Validator = Callable[[str], bool]
MetricsHook = Callable[[str, dict[str, Any]], None]


# --- Public dataclasses ---


@dataclass(frozen=True, slots=True)
class Hit:
    response: str
    similarity: float
    confidence: float
    age_seconds: int
    layer: HitLayer
    namespace: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Stats:
    namespace: str | None
    entries: int
    hits_exact: int
    hits_semantic: int
    misses: int
    evictions: int
    expirations: int
    embedder_fingerprint: str
    vector_dtype: str
    memory_bytes_estimate: int


@dataclass(frozen=True, slots=True)
class Health:
    healthy: bool
    schema_version: int
    integrity_ok: bool
    embedder_fingerprint_match: bool
    entries: int
    namespaces: int
    oldest_entry_age_seconds: int | None
    index_backend: str
    store_backend: str
    vector_dtype: str
    multi_process_mode: str


@dataclass(frozen=True, slots=True)
class StoredEntry:
    id: int
    namespace: str
    query_hash: str
    query: str
    response: str
    embedding: bytes
    metadata: dict[str, Any]
    created_at: int
    last_accessed_at: int
    ttl: int | None
    access_count: int


# --- Embedder protocols ---


@runtime_checkable
class Embedder(Protocol):
    """Sync embedder. Returns 1-D float32 vectors of length ``dim``."""

    @property
    def dim(self) -> int: ...

    @property
    def fingerprint(self) -> str: ...

    def embed(self, text: str) -> npt.NDArray[np.float32]: ...


@runtime_checkable
class AsyncEmbedder(Protocol):
    """Async embedder. Returns 1-D float32 vectors of length ``dim``."""

    @property
    def dim(self) -> int: ...

    @property
    def fingerprint(self) -> str: ...

    async def embed(self, text: str) -> npt.NDArray[np.float32]: ...


# --- Index protocol ---


@runtime_checkable
class Index(Protocol):
    """In-memory vector index over L2-normalized float32 vectors."""

    @property
    def dim(self) -> int: ...

    @property
    def size(self) -> int: ...

    @property
    def dtype(self) -> VectorDtype: ...

    def append(self, row_id: int, vec: npt.NDArray[Any], namespace: str) -> None: ...

    def remove(self, row_id: int) -> None: ...

    def search(
        self,
        query: npt.NDArray[Any],
        namespace: str,
        *,
        k: int = 1,
    ) -> list[tuple[int, float]]: ...

    def rebuild_from(
        self,
        rows: Iterable[tuple[int, npt.NDArray[Any], str]],
    ) -> None: ...

    def compact(self) -> None: ...

    def requantize(self, dtype: VectorDtype) -> None: ...


# --- Store protocol ---


@runtime_checkable
class Store(Protocol):
    """Persistent backend. Source of truth for entries; vectors persisted as float32.

    Implementations must provide write atomicity for ``insert`` (vector and
    metadata committed together) and durable reads for ``get_by_hash``.
    """

    # --- Lifecycle ---

    def open(self, embedder_fingerprint: str, embedder_dim: int) -> None:
        """Initialize storage. Validate fingerprint/dim against any existing data;
        raise ``EmbedderMismatchError`` or ``EmbedderDimensionError`` on mismatch."""

    def close(self) -> None: ...

    # --- Read ---

    def get_by_hash(self, namespace: str, query_hash: str) -> StoredEntry | None: ...

    def get_by_id(self, id: int) -> StoredEntry | None: ...

    def count(self, namespace: str | None = None) -> int: ...

    def list_namespaces(self) -> list[str]: ...

    def iter_lru_ids(self, n: int, namespace: str | None = None) -> Iterable[int]: ...

    def iter_all(self) -> Iterable[StoredEntry]: ...

    def iter_since(self, last_id: int) -> Iterable[StoredEntry]: ...

    # --- Write (must be transactional) ---

    def insert(self, entry: StoredEntry) -> int:
        """Returns the assigned id and bumps ``version_counter`` in the same txn."""

    def update_access(self, id: int, now: int) -> None: ...

    def delete_by_id(self, id: int) -> bool: ...

    def delete_expired(self, now: int, namespace: str | None = None) -> int: ...

    def clear_namespace(self, namespace: str) -> int: ...

    # --- Quotas ---

    def set_namespace_quota(self, namespace: str, max_entries: int) -> None: ...

    def get_namespace_quota(self, namespace: str) -> int | None: ...

    # --- Coordination ---

    def read_version_counter(self) -> int: ...

    def read_meta(self, key: str) -> str | None: ...

    def write_meta(self, key: str, value: str) -> None: ...

    # --- Health ---

    def integrity_check(self) -> bool: ...

    # --- Backup ---

    def snapshot_to(self, dest_path: str | Path) -> None:
        """Copy the entire store to ``dest_path``. Format is implementation-defined."""

    @classmethod
    def restore_from(cls, source_path: str | Path, dest_path: str | Path) -> Store: ...


__all__ = [
    "AsyncEmbedder",
    "ConfidenceFn",
    "Embedder",
    "Health",
    "Hit",
    "HitLayer",
    "Index",
    "IndexBackend",
    "MetricsHook",
    "MultiProcessMode",
    "Stats",
    "Store",
    "StoredEntry",
    "Validator",
    "VectorDtype",
]
