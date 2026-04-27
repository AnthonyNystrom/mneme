"""Exception hierarchy for mneme.

All public exceptions inherit from ``MnemeError``. Every exception raised by the
library carries an actionable remediation message at the call site.
"""

from __future__ import annotations


class MnemeError(Exception):
    """Base class for all mneme exceptions."""


class EmbedderMismatchError(MnemeError):
    """Stored embedder fingerprint does not match the supplied embedder.

    Remediation: reopen with the original embedder, or run
    ``mneme.tools.migrate.reembed()`` to migrate the cache to a new embedder.
    """


class EmbedderDimensionError(MnemeError):
    """Embedder dimensionality does not match the cache schema.

    Remediation: verify ``embedder.dim`` against the dim recorded in the cache;
    use ``reembed()`` to change dimension.
    """


class CorruptCacheError(MnemeError):
    """The store has failed an integrity check.

    Remediation: restore from a checkpoint via ``SemanticCache.loads()``, or
    delete the cache files and warm fresh.
    """


class SchemaMigrationError(MnemeError):
    """A forward migration failed.

    Remediation: inspect logs for the failing migration; fix the underlying
    issue or restore from a pre-upgrade checkpoint.
    """


class CacheClosedError(MnemeError):
    """A cache method was called after ``close()``.

    Remediation: open a new cache instance.
    """


class CheckpointError(MnemeError):
    """Checkpoint dump or load failed.

    Remediation: check disk space, file permissions, archive integrity, and
    embedder fingerprint consistency.
    """


class IndexBackendUnavailableError(MnemeError):
    """The requested index backend is not installed.

    Remediation: install the optional extra (e.g. ``pip install mneme[hnsw]``)
    or fall back to ``index_backend='numpy'``.
    """


class StoreBackendError(MnemeError):
    """An operational error from a Store backend.

    Remediation: see the wrapped exception (``__cause__``) for backend-specific
    details (disk full, network error, permissions, etc.).
    """


class NamespaceQuotaExceededError(MnemeError):
    """A namespace put would exceed its quota and eviction cannot reclaim space.

    Remediation: raise the quota via ``set_namespace_quota``, call ``vacuum()``
    to remove expired entries, or delete obsolete entries.
    """


class MultiProcessLockError(MnemeError):
    """Multi-process coordination failed.

    Remediation: check process permissions on the lock file and verify all
    processes use the same ``multi_process_mode``.
    """


class QuantizationError(MnemeError):
    """Incompatible vector dtype across processes or invalid input for quantization.

    Remediation: ensure all processes use the same ``vector_dtype``; for int8
    quantization, inputs must be L2-normalized to the range [-1, 1].
    """
