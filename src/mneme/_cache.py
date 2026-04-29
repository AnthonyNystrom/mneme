"""``SemanticCache``: layered cache (exact match -> semantic match).

Public API. The cache wires a ``Store`` (persistence boundary),
an ``Index`` (in-memory vector matrix), eviction, metrics, and
a single ``threading.RLock`` that guards every public method.

Algorithm:

- Layer 1: hash the normalized query, ``store.get_by_hash``. Hit -> Hit
  with ``layer="exact"``, ``similarity=1.0``.
- Layer 2: embed, L2-normalize, ``index.search``. For each candidate above
  ``similarity_threshold``, validate freshness, run ``validator``, score
  via ``confidence_fn``, accept if ``confidence >= 0.7``.

Invariants enforced here:

- ``_cache.py`` depends only on the ``Store`` Protocol; never imports
  concrete Store classes other than the default ``SQLiteStore`` at the
  ``path`` boundary.
- L2-normalization happens at the cache layer before any vector reaches
  the Index.
- Embedder failure during ``get`` -> miss + WARNING. During ``put`` -> raise.
- On open, ``store.count()`` is cross-checked against ``index.size``;
  divergence triggers a rebuild from ``store.iter_all()``.
- Counters are persisted to ``store.write_meta("counters", ...)`` on close
  and restored on open.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ._eviction import maybe_evict
from ._exceptions import (
    CacheClosedError,
    EmbedderDimensionError,
    IndexBackendUnavailableError,
)
from ._index import NumpyIndex
from ._metrics import MetricsDispatcher
from ._normalize import hash_query, normalize
from ._quantization import memory_bytes_estimate
from ._scoring import default_confidence, default_validator
from ._store_sqlite import SQLiteStore
from ._types import (
    ConfidenceFn,
    Embedder,
    Health,
    Hit,
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

logger = logging.getLogger("mneme.cache")

# Auto-select hnsw above 500k entries.
_AUTO_HNSW_THRESHOLD = 500_000
# Confidence cutoff is 0.7 — fixed by the spec.
_CONFIDENCE_CUTOFF = 0.7
_COUNTER_META_KEY = "counters"


def _l2_normalize(vec: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """Cosine semantics require unit-length vectors. Zero-vector returns as-is."""
    v32 = vec.astype(np.float32, copy=False)
    n = float(np.linalg.norm(v32))
    if n == 0.0:
        return v32
    return (v32 / n).astype(np.float32, copy=False)


def _backend_label(obj: object, suffix: str) -> str:
    name = type(obj).__name__
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name.lower()


class SemanticCache:
    """Synchronous layered semantic cache.

    Construct with either ``path`` (creates a default ``SQLiteStore``) or
    ``store`` (any ``Store`` impl). Every public method is RLock-guarded.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        embedder: Embedder | None = None,
        *,
        store: Store | None = None,
        similarity_threshold: float = 0.85,
        default_ttl: int | None = None,
        max_entries: int | None = None,
        namespace_quotas: dict[str, int] | None = None,
        confidence_fn: ConfidenceFn | None = None,
        validator: Validator | None = None,
        metrics_hook: MetricsHook | None = None,
        normalize: bool = True,
        index_backend: IndexBackend = "auto",
        index_options: dict[str, Any] | None = None,
        vector_dtype: VectorDtype = "float32",
        multi_process_mode: MultiProcessMode = "single",
        stale_check_interval: float = 0.0,
        max_query_bytes: int = 32_768,
        max_response_bytes: int = 4 * 1_048_576,
        max_metadata_bytes: int = 65_536,
    ) -> None:
        # Exactly one of `path` or `store`. None+None or both
        # are configuration errors.
        if (path is None) == (store is None):
            raise ValueError(
                "SemanticCache: provide exactly one of `path` or `store`. "
                "Remediation: pass `path='cache.db'` for the default "
                "SQLiteStore, or `store=YourStore(...)` for a custom backend."
            )
        if embedder is None:
            raise ValueError(
                "SemanticCache: `embedder` is required. Remediation: pass an "
                "object implementing the Embedder Protocol; see "
                "examples/reference_embedders/ for sentence-transformers, "
                "OpenAI, Bedrock, and Ollama wrappers."
            )

        self._embedder = embedder
        self._lock = threading.RLock()
        self._closed = False

        self._store: Store = store if store is not None else SQLiteStore(path)  # type: ignore[arg-type]
        self._store.open(embedder.fingerprint, embedder.dim)

        # Settings
        self._similarity_threshold = float(similarity_threshold)
        self._default_ttl = default_ttl
        self._max_entries = max_entries
        self._namespace_quotas = dict(namespace_quotas or {})
        self._confidence_fn: ConfidenceFn = confidence_fn or default_confidence
        self._validator: Validator = validator or default_validator
        self._normalize_queries = normalize
        self._index_backend_choice: IndexBackend = index_backend
        self._index_options = dict(index_options or {})
        self._vector_dtype: VectorDtype = vector_dtype
        self._multi_process_mode: MultiProcessMode = multi_process_mode
        self._stale_check_interval = stale_check_interval
        self._max_query_bytes = max_query_bytes
        self._max_response_bytes = max_response_bytes
        self._max_metadata_bytes = max_metadata_bytes

        self._metrics = MetricsDispatcher(metrics_hook)
        self._restore_counters()

        # Mirror namespace quotas into the store for cross-process visibility.
        for ns, quota in self._namespace_quotas.items():
            self._store.set_namespace_quota(ns, quota)

        self._index: Index = self._build_index()
        self._rebuild_index_from_store()

        # Multi-process coordination. Only stale-tolerant gets a runtime
        # coordinator hook here; mmap-shared mode is a separately-instantiable
        # primitive (see ``_multiproc.MmapSharedCoordinator``) used directly by
        # advanced users — wiring it into the cache happens via index_options.
        self._coordinator: Any = None
        if multi_process_mode == "stale-tolerant":
            from ._multiproc import StaleTolerantCoordinator

            self._coordinator = StaleTolerantCoordinator(
                self._store, self._index, stale_check_interval=stale_check_interval
            )

    # --- Construction helpers ---

    def _build_index(self) -> Index:
        backend = self._index_backend_choice
        if backend == "auto":
            count = self._store.count()
            if count >= _AUTO_HNSW_THRESHOLD:
                hnsw = self._try_make_hnsw(strict=False)
                if hnsw is not None:
                    return hnsw
            backend = "numpy"
        if backend == "hnsw":
            hnsw = self._try_make_hnsw(strict=True)
            if hnsw is not None:
                return hnsw
            # strict=True raised; unreachable
        return NumpyIndex(
            self._embedder.dim,
            dtype=self._vector_dtype,
            **self._index_options,
        )

    def _try_make_hnsw(self, *, strict: bool) -> Index | None:
        try:
            from ._index_hnsw import HnswIndex
        except IndexBackendUnavailableError:
            if strict:
                raise
            logger.warning(
                "index_backend='auto' would prefer hnsw, but the [hnsw] extra "
                "is not installed; falling back to NumPy. Install with "
                "`pip install mneme[hnsw]` for sub-1ms search at >500k entries."
            )
            return None
        try:
            return HnswIndex(
                self._embedder.dim,
                dtype=self._vector_dtype,
                index_options=self._index_options,
            )
        except IndexBackendUnavailableError:
            if strict:
                raise
            return None

    def _rebuild_index_from_store(self) -> None:
        """Cross-check store/index sizes; rebuild index from store on divergence.

        When the store exposes ``iter_index_rows`` (a bulk-read of just the
        id/embedding/namespace triples needed for index rebuild), use it to
        skip ``StoredEntry`` construction + JSON metadata parse per row.
        That fast path keeps the Phase-14 open-time targets in reach.
        """
        rows: list[tuple[int, npt.NDArray[Any], str]] = []
        fast_iter = getattr(self._store, "iter_index_rows", None)
        if callable(fast_iter):
            for row_id, emb_bytes, ns in fast_iter():
                vec = np.frombuffer(emb_bytes, dtype=np.float32).copy()
                rows.append((row_id, vec, ns))
        else:
            for entry in self._store.iter_all():
                vec = np.frombuffer(entry.embedding, dtype=np.float32).copy()
                rows.append((entry.id, vec, entry.namespace))
        store_count = self._store.count()
        self._index.rebuild_from(rows)
        if self._index.size != store_count:
            logger.info(
                "Index size %d differs from store count %d after rebuild.",
                self._index.size,
                store_count,
            )

    def _restore_counters(self) -> None:
        try:
            data = self._store.read_meta(_COUNTER_META_KEY)
        except Exception:
            return
        if not data:
            return
        try:
            self._metrics.counters.restore(json.loads(data))
        except (ValueError, TypeError):
            logger.warning(
                "Could not restore counters from store meta; starting fresh.",
                exc_info=True,
            )

    def _persist_counters(self) -> None:
        try:
            self._store.write_meta(
                _COUNTER_META_KEY,
                json.dumps(self._metrics.counters.serialize()),
            )
        except Exception:
            logger.warning("Failed to persist counters to store meta", exc_info=True)

    # --- Validation helpers ---

    def _check_open(self) -> None:
        if self._closed:
            raise CacheClosedError("SemanticCache is closed. Remediation: open a new instance.")

    def _validate_sizes(self, query: str, response: str, metadata: dict[str, Any]) -> str:
        if len(query.encode("utf-8")) > self._max_query_bytes:
            raise ValueError(
                f"Query exceeds max_query_bytes={self._max_query_bytes}. "
                f"Remediation: shorten the query or raise the limit."
            )
        if len(response.encode("utf-8")) > self._max_response_bytes:
            raise ValueError(
                f"Response exceeds max_response_bytes={self._max_response_bytes}. "
                f"Remediation: shorten the response or raise the limit."
            )
        meta_json = json.dumps(metadata)
        if len(meta_json.encode("utf-8")) > self._max_metadata_bytes:
            raise ValueError(
                f"Metadata JSON exceeds max_metadata_bytes={self._max_metadata_bytes}. "
                f"Remediation: shrink metadata or raise the limit."
            )
        return meta_json

    def _normalize_query(self, query: str) -> str:
        return normalize(query) if self._normalize_queries else query

    def _validate_embedding(self, embedding: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        if embedding.shape != (self._embedder.dim,):
            raise EmbedderDimensionError(
                f"Embedding shape {embedding.shape} does not match "
                f"embedder.dim={self._embedder.dim}. Remediation: ensure the "
                f"embedder returns a 1-D array of the expected length."
            )
        return _l2_normalize(embedding)

    # --- Internal locked helpers (also used by AsyncSemanticCache) ---
    #
    # The `_locked` suffix marks helpers that assume the caller already holds
    # ``self._lock``. The `_async_*` shims acquire the lock and then delegate.
    # AsyncSemanticCache calls the `_async_*` shims via ``asyncio.to_thread``
    # so each lock acquisition is bounded to a single sync call.

    def _refresh_coordinator(self) -> None:
        """If a multi-process coordinator is attached, sync index from store
        before the layered lookup. No-op in single-process mode."""
        if self._coordinator is not None:
            self._coordinator.refresh()

    def _layer1_locked(self, query: str, namespace: str, bypass: bool) -> tuple[Hit | None, bool]:
        """Layer-1 lookup. Returns ``(hit_or_None, need_embedder)``.

        ``need_embedder=True`` means layer-1 missed and the caller should
        embed the query and pass it to ``_layer2_locked``.
        """
        self._refresh_coordinator()
        if bypass:
            self._metrics.emit_miss(namespace, reason="bypass")
            return None, False

        normalized = self._normalize_query(query)
        qhash = hash_query(normalized)
        now = int(time.time())

        entry = self._store.get_by_hash(namespace, qhash)
        if entry is None:
            return None, True
        age = now - entry.created_at
        if entry.ttl is not None and age >= entry.ttl:
            self._store.delete_by_id(entry.id)
            self._index.remove(entry.id)
            self._metrics.emit_expired(namespace, count=1)
            return None, True
        self._store.update_access(entry.id, now)
        confidence = self._confidence_fn(1.0, age, entry.metadata)
        hit = Hit(
            response=entry.response,
            similarity=1.0,
            confidence=float(confidence),
            age_seconds=age,
            layer="exact",
            namespace=namespace,
            metadata=dict(entry.metadata),
        )
        self._metrics.emit_hit(namespace, "exact", 1.0, float(confidence), age)
        return hit, False

    def _layer2_locked(
        self,
        embedding: npt.NDArray[Any],
        namespace: str,
    ) -> Hit | None:
        """Layer-2 lookup. Caller must already hold the lock and have a
        pre-computed (still-raw) embedding."""
        try:
            qvec = self._validate_embedding(embedding)
        except EmbedderDimensionError:
            self._metrics.emit_miss(namespace, reason="dim_mismatch")
            return None

        now = int(time.time())
        results = self._index.search(qvec, namespace, k=3)
        for row_id, sim in results:
            if sim < self._similarity_threshold:
                self._metrics.emit_miss(namespace, reason="below_threshold")
                return None
            cand = self._store.get_by_id(row_id)
            if cand is None:
                # Stale index pointer — clean up and try the next candidate.
                self._index.remove(row_id)
                continue
            age = now - cand.created_at
            if cand.ttl is not None and age >= cand.ttl:
                self._store.delete_by_id(cand.id)
                self._index.remove(cand.id)
                self._metrics.emit_expired(namespace, count=1)
                continue
            if not self._validator(cand.response):
                self._store.delete_by_id(cand.id)
                self._index.remove(cand.id)
                continue
            confidence = self._confidence_fn(sim, age, cand.metadata)
            if confidence < _CONFIDENCE_CUTOFF:
                continue
            self._store.update_access(cand.id, now)
            hit = Hit(
                response=cand.response,
                similarity=float(sim),
                confidence=float(confidence),
                age_seconds=age,
                layer="semantic",
                namespace=namespace,
                metadata=dict(cand.metadata),
            )
            self._metrics.emit_hit(namespace, "semantic", float(sim), float(confidence), age)
            return hit

        self._metrics.emit_miss(namespace, reason="no_match")
        return None

    # --- `_async_*` shims (entry points for AsyncSemanticCache) ---

    def _async_layer1(self, query: str, namespace: str, bypass: bool) -> tuple[Hit | None, bool]:
        with self._lock:
            self._check_open()
            return self._layer1_locked(query, namespace, bypass)

    def _async_layer2(self, embedding: npt.NDArray[Any], namespace: str) -> Hit | None:
        with self._lock:
            self._check_open()
            return self._layer2_locked(embedding, namespace)

    def _async_record_embedder_failure(self, namespace: str) -> None:
        with self._lock:
            self._check_open()
            self._metrics.emit_miss(namespace, reason="embedder_failure")

    def _async_put(
        self,
        query: str,
        response: str,
        embedding: npt.NDArray[Any],
        namespace: str,
        metadata: dict[str, Any] | None,
        ttl: int | None,
    ) -> None:
        """Locked put with a pre-computed embedding (used by async layer)."""
        with self._lock:
            self._check_open()
            self._put_locked(query, response, embedding, namespace, metadata, ttl)

    def _put_locked(
        self,
        query: str,
        response: str,
        embedding: npt.NDArray[Any],
        namespace: str,
        metadata: dict[str, Any] | None,
        ttl: int | None,
    ) -> None:
        meta = dict(metadata) if metadata else {}
        self._validate_sizes(query, response, meta)
        normalized = self._normalize_query(query)
        qhash = hash_query(normalized)
        qvec = self._validate_embedding(embedding)
        now = int(time.time())
        effective_ttl = ttl if ttl is not None else self._default_ttl
        entry = StoredEntry(
            id=0,
            namespace=namespace,
            query_hash=qhash,
            query=query,
            response=response,
            embedding=qvec.tobytes(),
            metadata=meta,
            created_at=now,
            last_accessed_at=now,
            ttl=effective_ttl,
            access_count=0,
        )
        row_id = self._store.insert(entry)
        self._index.append(row_id, qvec, namespace)
        evicted = maybe_evict(
            self._store,
            namespace,
            namespace_quotas=self._namespace_quotas,
            max_entries=self._max_entries,
            on_evicted=self._index.remove,
        )
        for ns, count in evicted.items():
            self._metrics.emit_eviction(ns, count)

    def _normalize_query_locked(self, query: str) -> str:
        # Public-ish view of normalize for the async layer; doesn't take the
        # lock (pure function over self._normalize_queries).
        return self._normalize_query(query)

    # --- Public API ---

    def get(
        self,
        query: str,
        *,
        embedding: npt.NDArray[Any] | None = None,
        namespace: str = "default",
        bypass: bool = False,
    ) -> Hit | None:
        """Layered lookup: exact match (Layer 1), then semantic match (Layer 2).

        Args:
            query: The query string. Normalized per ``normalize=`` constructor flag.
            embedding: Optional precomputed embedding. If supplied, the cache
                skips its own embedder call. Useful when you've already embedded
                the query for another purpose (RAG retrieval, etc.).
            namespace: Multi-tenant scope. Layer-1 hashes are namespace-scoped;
                Layer-2 search is restricted to the namespace's vectors.
            bypass: If ``True``, **force a miss** — skip both Layer 1 and Layer 2,
                emit a ``miss`` metric with ``reason="bypass"``, and return
                ``None``. Useful for forcing the caller to invoke the underlying
                LLM (e.g. to refresh a stale-but-not-yet-TTL'd answer, or to
                A/B test cached vs. fresh responses).

        Returns:
            A ``Hit`` if Layer 1 or Layer 2 found a match passing the validator
            and confidence cutoff; ``None`` otherwise (cache miss, embedder
            failure, or ``bypass=True``).
        """
        # The sync get holds the lock for the full duration,
        # including any embedder call. AsyncSemanticCache.get drops the lock
        # around the embedder.
        with self._lock:
            self._check_open()
            hit, need_embedder = self._layer1_locked(query, namespace, bypass)
            if not need_embedder:
                return hit
            if embedding is None:
                try:
                    embedding = self._embedder.embed(self._normalize_query(query))
                except Exception:
                    logger.warning(
                        "Embedder failed during get; treating as miss",
                        exc_info=True,
                    )
                    self._metrics.emit_miss(namespace, reason="embedder_failure")
                    return None
            return self._layer2_locked(embedding, namespace)

    def put(
        self,
        query: str,
        response: str,
        *,
        embedding: npt.NDArray[Any] | None = None,
        namespace: str = "default",
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> None:
        with self._lock:
            self._check_open()
            # Embedder failure during put propagates.
            if embedding is None:
                embedding = self._embedder.embed(self._normalize_query(query))
            self._put_locked(query, response, embedding, namespace, metadata, ttl)

    def delete(self, query: str, *, namespace: str = "default") -> bool:
        with self._lock:
            self._check_open()
            normalized = self._normalize_query(query)
            qhash = hash_query(normalized)
            entry = self._store.get_by_hash(namespace, qhash)
            if entry is None:
                return False
            removed = self._store.delete_by_id(entry.id)
            if removed:
                self._index.remove(entry.id)
            return removed

    def vacuum(self, *, namespace: str | None = None, compact: bool = True) -> int:
        """Sweep TTL-expired entries and (by default) compact the index.

        Args:
            namespace: Limit the sweep to a single namespace. ``None`` sweeps all.
            compact: If ``True`` (default), call :meth:`compact` after the sweep
                so the in-memory index actually releases the deleted rows'
                memory. Set ``False`` if you want to schedule compaction
                separately (e.g. less often than vacuum).

        Returns:
            The number of expired entries removed.
        """
        with self._lock:
            self._check_open()
            now = int(time.time())
            # Two-pass: collect ids to expire so we can update the index too.
            expired_ids: list[tuple[int, str]] = []
            for entry in self._store.iter_all():
                if entry.ttl is None:
                    continue
                if namespace is not None and entry.namespace != namespace:
                    continue
                if entry.created_at + entry.ttl <= now:
                    expired_ids.append((entry.id, entry.namespace))
            count = 0
            per_ns: dict[str, int] = {}
            for id_, ns in expired_ids:
                if self._store.delete_by_id(id_):
                    self._index.remove(id_)
                    count += 1
                    per_ns[ns] = per_ns.get(ns, 0) + 1
            for ns, n in per_ns.items():
                self._metrics.emit_expired(ns, n)
            if compact:
                self._index.compact()
            return count

    def compact(self) -> int:
        """Reclaim memory occupied by tombstoned (soft-deleted) index rows.

        ``remove``, TTL expiry, and LRU eviction all *mark* index rows deleted
        without freeing their underlying matrix bytes. Long-running caches
        with churn accumulate tombstone memory; calling ``compact`` rebuilds
        the in-memory matrix at the live size and releases the rest.

        Cheap when there are no tombstones (early-return). The store is not
        touched — entries already deleted from the store remain deleted.

        Returns:
            The number of tombstones reclaimed.
        """
        with self._lock:
            self._check_open()
            before = int(getattr(self._index, "tombstone_count", 0))
            self._index.compact()
            after = int(getattr(self._index, "tombstone_count", 0))
            return max(before - after, 0)

    def stats(self, *, namespace: str | None = None) -> Stats:
        with self._lock:
            self._check_open()
            if namespace is None:
                counts = self._metrics.counters.aggregate()
                entries = self._store.count()
            else:
                counts = self._metrics.counters.get_namespace(namespace)
                entries = self._store.count(namespace)
            mem = memory_bytes_estimate(entries, self._embedder.dim, self._vector_dtype)
            idx_mem = getattr(self._index, "memory_bytes", None)
            idx_tomb = getattr(self._index, "tombstone_count", None)
            return Stats(
                namespace=namespace,
                entries=entries,
                hits_exact=counts.get("hits_exact", 0),
                hits_semantic=counts.get("hits_semantic", 0),
                misses=counts.get("misses", 0),
                evictions=counts.get("evictions", 0),
                expirations=counts.get("expirations", 0),
                embedder_fingerprint=self._embedder.fingerprint,
                vector_dtype=self._vector_dtype,
                memory_bytes_estimate=mem,
                index_memory_bytes=int(idx_mem) if idx_mem is not None else None,
                index_tombstone_count=int(idx_tomb) if idx_tomb is not None else None,
            )

    def health(self) -> Health:
        with self._lock:
            self._check_open()
            entries = self._store.count()
            namespaces = len(self._store.list_namespaces())
            try:
                integrity_ok = self._store.integrity_check()
            except Exception:
                integrity_ok = False
            stored_fp = self._store.read_meta("embedder_fingerprint") or ""
            schema_version_str = self._store.read_meta("schema_version") or "1"
            try:
                schema_version = int(schema_version_str)
            except ValueError:
                schema_version = 1
            oldest_age: int | None = None
            now = int(time.time())
            for entry in self._store.iter_all():
                age = now - entry.created_at
                if oldest_age is None or age > oldest_age:
                    oldest_age = age
            return Health(
                healthy=integrity_ok,
                schema_version=schema_version,
                integrity_ok=integrity_ok,
                embedder_fingerprint_match=(stored_fp == self._embedder.fingerprint),
                entries=entries,
                namespaces=namespaces,
                oldest_entry_age_seconds=oldest_age,
                index_backend=_backend_label(self._index, "Index"),
                store_backend=_backend_label(self._store, "Store"),
                vector_dtype=self._vector_dtype,
                multi_process_mode=self._multi_process_mode,
            )

    def list_namespaces(self) -> list[str]:
        with self._lock:
            self._check_open()
            return self._store.list_namespaces()

    def clear_namespace(self, namespace: str) -> int:
        with self._lock:
            self._check_open()
            # Collect ids before clearing so we can update the index.
            ns_count = self._store.count(namespace)
            ids_to_remove = list(self._store.iter_lru_ids(ns_count, namespace=namespace))
            count = self._store.clear_namespace(namespace)
            for id_ in ids_to_remove:
                self._index.remove(id_)
            self._metrics.counters.clear_namespace(namespace)
            return count

    def clear(self) -> int:
        """Wipe every entry across every namespace.

        Returns the total number of entries removed. Backend-agnostic:
        works against any ``Store`` implementation (Memory, SQLite,
        Redis, Postgres, DynamoDB, custom). Bumps the store's
        ``version_counter`` once per namespace cleared, so multi-process
        readers see the change.

        The in-memory index is rebuilt empty rather than tombstoned
        per-row — cheaper than ``O(n)`` ``remove()`` calls at scale.
        """
        with self._lock:
            self._check_open()
            total = 0
            for ns in self._store.list_namespaces():
                total += self._store.clear_namespace(ns)
                self._metrics.counters.clear_namespace(ns)
            # Index is now logically empty; rebuild from scratch is the
            # fastest way to reflect that.
            self._index.rebuild_from(())
            return total

    def requantize(self, dtype: VectorDtype) -> None:
        with self._lock:
            self._check_open()
            self._index.requantize(dtype)
            self._vector_dtype = dtype

    def set_similarity_threshold(self, value: float) -> None:
        """Adjust the cosine-similarity threshold for Layer-2 matches at runtime.

        Affects subsequent ``get`` calls only; entries already cached are
        not re-evaluated. ``value`` must be in ``[-1.0, 1.0]``. For
        L2-normalized embeddings the useful range is ``[0.0, 1.0]``;
        higher means stricter matching, lower means more permissive.
        """
        v = float(value)
        if not (-1.0 <= v <= 1.0):
            raise ValueError(
                f"similarity_threshold must be in [-1.0, 1.0]; got {value}. "
                "Remediation: cosine similarity over L2-normalized vectors "
                "is bounded to this range."
            )
        with self._lock:
            self._check_open()
            self._similarity_threshold = v

    @property
    def similarity_threshold(self) -> float:
        """Current Layer-2 similarity threshold."""
        return self._similarity_threshold

    # --- Checkpoint ---

    def dumps(self, dest: str | Path) -> None:
        """Write a checkpoint archive to ``dest`` (tar.gz)."""
        from ._checkpoint import dumps as _dumps

        with self._lock:
            self._check_open()
            _dumps(self, dest)

    @classmethod
    def loads(
        cls,
        source: str | Path,
        path: str | Path,
        embedder: Embedder,
        **kwargs: Any,
    ) -> SemanticCache:
        """Restore a checkpoint into a fresh ``SemanticCache`` at ``path``."""
        from ._checkpoint import loads as _loads

        return _loads(source, path, embedder, **kwargs)

    # --- Lifecycle ---

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._persist_counters()
            with suppress(Exception):
                self._store.close()
            self._closed = True

    def __enter__(self) -> SemanticCache:
        return self

    def __exit__(self, *exc_info: object) -> None:
        del exc_info
        self.close()


__all__ = ["SemanticCache"]
