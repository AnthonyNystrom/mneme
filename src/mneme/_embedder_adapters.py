"""Sync <-> async embedder adapters per PRD §8.3.

- ``to_async_embedder(sync_embedder)``: wrap a sync embedder so its
  ``embed()`` is awaitable (runs the inner call in ``asyncio.to_thread``).
- ``to_sync_embedder(async_embedder)``: wrap an async embedder so its
  ``embed()`` is sync (uses ``asyncio.run`` when no loop is active; raises
  if called from inside a running loop, since blocking the loop would
  deadlock).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ._types import AsyncEmbedder, Embedder


class _SyncToAsyncEmbedder:
    """Sync embedder wrapped so callers can ``await``."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    async def embed(self, text: str) -> npt.NDArray[np.float32]:
        return await asyncio.to_thread(self._inner.embed, text)


class _AsyncToSyncEmbedder:
    """Async embedder wrapped so callers can call sync ``embed()``.

    Only safe outside a running event loop (because we'd deadlock blocking
    the loop on its own coroutine). Inside a running loop, raises
    ``RuntimeError``.
    """

    def __init__(self, inner: AsyncEmbedder) -> None:
        self._inner = inner

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._inner.embed(text))
        raise RuntimeError(
            "to_sync_embedder().embed() cannot be called from inside a running "
            "event loop. Remediation: pass the original AsyncEmbedder to "
            "AsyncSemanticCache, or call this from a sync context."
        )


def to_async_embedder(embedder: Embedder) -> AsyncEmbedder:
    """Wrap a sync ``Embedder`` so it satisfies ``AsyncEmbedder``."""
    return _SyncToAsyncEmbedder(embedder)


def to_sync_embedder(embedder: AsyncEmbedder) -> Embedder:
    """Wrap an ``AsyncEmbedder`` so it satisfies ``Embedder`` (sync)."""
    return _AsyncToSyncEmbedder(embedder)


__all__ = ["to_async_embedder", "to_sync_embedder"]
