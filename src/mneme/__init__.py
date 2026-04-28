"""mneme: a layered semantic cache for LLM applications.

Public API per PRD §8. Async types and adapter helpers join in Phase 8 and
Phase 12 respectively.
"""

from ._cache import SemanticCache
from ._exceptions import (
    CacheClosedError,
    CheckpointError,
    CorruptCacheError,
    EmbedderDimensionError,
    EmbedderMismatchError,
    IndexBackendUnavailableError,
    MnemeError,
    MultiProcessLockError,
    NamespaceQuotaExceededError,
    QuantizationError,
    SchemaMigrationError,
    StoreBackendError,
)
from ._store_memory import MemoryStore
from ._store_sqlite import SQLiteStore
from ._types import (
    AsyncEmbedder,
    ConfidenceFn,
    Embedder,
    Health,
    Hit,
    HitLayer,
    Index,
    IndexBackend,
    MetricsHook,
    MultiProcessMode,
    Stats,
    Store,
    StoredEntry,
    Validator,
    VectorDtype,
)

__version__ = "0.1.0"

__all__ = [
    # Protocols (re-exported for type checking + structural conformance)
    "AsyncEmbedder",
    # Exception hierarchy
    "CacheClosedError",
    "CheckpointError",
    # Type aliases
    "ConfidenceFn",
    "CorruptCacheError",
    "Embedder",
    "EmbedderDimensionError",
    "EmbedderMismatchError",
    # Public dataclasses
    "Health",
    "Hit",
    "HitLayer",
    "Index",
    "IndexBackend",
    "IndexBackendUnavailableError",
    # Stores
    "MemoryStore",
    "MetricsHook",
    "MnemeError",
    "MultiProcessLockError",
    "MultiProcessMode",
    "NamespaceQuotaExceededError",
    "QuantizationError",
    "SQLiteStore",
    "SchemaMigrationError",
    # Cache
    "SemanticCache",
    "Stats",
    "Store",
    "StoreBackendError",
    "StoredEntry",
    "Validator",
    "VectorDtype",
    "__version__",
]
