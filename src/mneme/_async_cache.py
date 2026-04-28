"""``AsyncSemanticCache``: async wrapper around ``SemanticCache`` per PRD §13.

Design:

- All sync work runs inside the same RLock-protected ``SemanticCache`` core
  via ``asyncio.to_thread``. Each ``to_thread`` invocation acquires the lock,
  runs to completion on a worker thread, releases.
- The embedder is awaited *between* layered store operations, so the lock
  isn't held during embedder I/O — that's how async cache enables fan-out.
- Sync (non-IO) accessors (``stats``, ``health``, ``list_namespaces``,
  ``requantize``) pass through directly: they're cheap and lock-bounded.
- Cancellation during ``await self._embedder.embed(...)`` discards the
  partial work; cancellation during a ``to_thread`` lets the thread finish
  (Python doesn't interrupt threads). Cache state stays consistent because
  every store write is its own atomic transaction.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ._cache import SemanticCache
from ._exceptions import CacheClosedError, EmbedderDimensionError
from ._types import (
    AsyncEmbedder,
    ConfidenceFn,
    Health,
    Hit,
    IndexBackend,
    MetricsHook,
    MultiProcessMode,
    Stats,
    Store,
    Validator,
    VectorDtype,
)

logger = logging.getLogger("mneme.async_cache")


class _NoOpSyncEmbedder:
    """Provides ``dim`` and ``fingerprint`` from an AsyncEmbedder.

    The sync ``SemanticCache`` requires an Embedder (sync) at construction
    so it can validate fingerprint/dim against the store. The async layer
    never lets ``embed()`` be called inside the sync core (it always passes
    ``embedding=`` pre-computed). This stub guards against accidental
    misuse: any sync embed would mean a code path forgot to embed first.
    """

    def __init__(self, async_embedder: AsyncEmbedder) -> None:
        self._inner = async_embedder

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        del text
        raise RuntimeError(
            "AsyncSemanticCache: sync embedder stub called. This should never "
            "happen — async paths always pre-embed and pass the result. "
            "Remediation: report this as a bug."
        )


class AsyncSemanticCache:
    """Async layered semantic cache.

    Same constructor signature as ``SemanticCache`` except ``embedder`` is
    an ``AsyncEmbedder`` (its ``embed`` returns a coroutine).
    """

    def __init__(
        self,
        path: str | Path | None = None,
        embedder: AsyncEmbedder | None = None,
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
        max_response_bytes: int = 1_048_576,
        max_metadata_bytes: int = 65_536,
    ) -> None:
        if embedder is None:
            raise ValueError(
                "AsyncSemanticCache: `embedder` is required. Remediation: "
                "pass an object implementing the AsyncEmbedder Protocol."
            )
        self._async_embedder = embedder
        # Wrap the async embedder in a sync stub so SemanticCache can pull
        # its dim/fingerprint without ever calling embed().
        sync_stub = _NoOpSyncEmbedder(embedder)
        self._sync_core = SemanticCache(
            path=path,
            embedder=sync_stub,
            store=store,
            similarity_threshold=similarity_threshold,
            default_ttl=default_ttl,
            max_entries=max_entries,
            namespace_quotas=namespace_quotas,
            confidence_fn=confidence_fn,
            validator=validator,
            metrics_hook=metrics_hook,
            normalize=normalize,
            index_backend=index_backend,
            index_options=index_options,
            vector_dtype=vector_dtype,
            multi_process_mode=multi_process_mode,
            stale_check_interval=stale_check_interval,
            max_query_bytes=max_query_bytes,
            max_response_bytes=max_response_bytes,
            max_metadata_bytes=max_metadata_bytes,
        )

    # --- Async I/O methods ---

    async def get(
        self,
        query: str,
        *,
        embedding: npt.NDArray[Any] | None = None,
        namespace: str = "default",
        bypass: bool = False,
    ) -> Hit | None:
        # Layer 1: short-circuit before any embedder call.
        hit, need_embedder = await asyncio.to_thread(
            self._sync_core._async_layer1, query, namespace, bypass
        )
        if not need_embedder:
            return hit
        # Need embedder. Drop the lock during the await; sync core will
        # re-acquire for layer 2.
        if embedding is None:
            try:
                embedding = await self._async_embedder.embed(
                    self._sync_core._normalize_query_locked(query)
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Embedder failed during async get; treating as miss",
                    exc_info=True,
                )
                await asyncio.to_thread(self._sync_core._async_record_embedder_failure, namespace)
                return None
        return await asyncio.to_thread(self._sync_core._async_layer2, embedding, namespace)

    async def put(
        self,
        query: str,
        response: str,
        *,
        embedding: npt.NDArray[Any] | None = None,
        namespace: str = "default",
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> None:
        # Embedder failures during put propagate (§22).
        if embedding is None:
            embedding = await self._async_embedder.embed(
                self._sync_core._normalize_query_locked(query)
            )
        await asyncio.to_thread(
            self._sync_core._async_put,
            query,
            response,
            embedding,
            namespace,
            metadata,
            ttl,
        )

    async def delete(self, query: str, *, namespace: str = "default") -> bool:
        return await asyncio.to_thread(self._sync_core.delete, query, namespace=namespace)

    async def vacuum(self, *, namespace: str | None = None) -> int:
        return await asyncio.to_thread(self._sync_core.vacuum, namespace=namespace)

    async def dumps(self, dest: str | Path) -> None:
        await asyncio.to_thread(self._sync_core.dumps, dest)

    @classmethod
    async def loads(
        cls,
        source: str | Path,
        path: str | Path,
        embedder: AsyncEmbedder,
        **kwargs: Any,
    ) -> AsyncSemanticCache:
        del source, path, embedder, kwargs
        raise NotImplementedError(
            "AsyncSemanticCache.loads is implemented in Phase 10 (checkpoint)."
        )

    # --- Sync accessors (cheap, RLock-bounded inside the call) ---

    def stats(self, *, namespace: str | None = None) -> Stats:
        return self._sync_core.stats(namespace=namespace)

    def health(self) -> Health:
        return self._sync_core.health()

    def list_namespaces(self) -> list[str]:
        return self._sync_core.list_namespaces()

    def clear_namespace(self, namespace: str) -> int:
        # Pure store/index work; OK to dispatch to a thread but cheap enough
        # that a direct call is acceptable. The lock is acquired inside.
        return self._sync_core.clear_namespace(namespace)

    def requantize(self, dtype: VectorDtype) -> None:
        self._sync_core.requantize(dtype)

    # --- Lifecycle ---

    async def close(self) -> None:
        await asyncio.to_thread(self._sync_core.close)

    async def __aenter__(self) -> AsyncSemanticCache:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        del exc_info
        with suppress(CacheClosedError):
            await self.close()


# Helper bound names referenced by tests; re-export the dimension error here
# so AsyncSemanticCache users don't have to dig into the package internals.
__all__ = ["AsyncSemanticCache", "EmbedderDimensionError"]
