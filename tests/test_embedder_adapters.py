"""Phase-8 sync<->async embedder adapter tests."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from mneme import to_async_embedder, to_sync_embedder

from .fakes import FakeAsyncEmbedder, FakeEmbedder

# --- to_async_embedder (sync -> async) ---


async def test_sync_to_async_preserves_dim_and_fingerprint():
    sync_e = FakeEmbedder(dim=16, fingerprint="orig:fp")
    async_e = to_async_embedder(sync_e)
    assert async_e.dim == 16
    assert async_e.fingerprint == "orig:fp"


async def test_sync_to_async_embed_is_awaitable():
    sync_e = FakeEmbedder(dim=8)
    async_e = to_async_embedder(sync_e)
    result = async_e.embed("hello")
    # The wrapped embed must return a coroutine that, when awaited, yields
    # the same vector as the sync embedder.
    assert asyncio.iscoroutine(result)
    v_async = await result
    v_sync = sync_e.embed("hello")
    np.testing.assert_array_equal(v_async, v_sync)


async def test_sync_to_async_round_trip_dim():
    sync_e = FakeEmbedder(dim=384)
    async_e = to_async_embedder(sync_e)
    v = await async_e.embed("anything")
    assert v.shape == (384,)
    assert v.dtype == np.float32


# --- to_sync_embedder (async -> sync) ---


def test_async_to_sync_preserves_dim_and_fingerprint():
    async_e = FakeAsyncEmbedder(dim=16, fingerprint="async:orig")
    sync_e = to_sync_embedder(async_e)
    assert sync_e.dim == 16
    assert sync_e.fingerprint == "async:orig"


def test_async_to_sync_embed_returns_array():
    async_e = FakeAsyncEmbedder(dim=8)
    sync_e = to_sync_embedder(async_e)
    v = sync_e.embed("hello")
    assert isinstance(v, np.ndarray)
    assert v.shape == (8,)


async def test_async_to_sync_inside_running_loop_raises():
    async_e = FakeAsyncEmbedder(dim=8)
    sync_e = to_sync_embedder(async_e)
    with pytest.raises(RuntimeError, match="running event loop"):
        sync_e.embed("hello")


# --- Round trip parity ---


def test_async_to_sync_then_to_async_preserves_behavior():
    """async -> sync -> async should still produce the original vectors."""
    async_e = FakeAsyncEmbedder(dim=8)
    sync_wrap = to_sync_embedder(async_e)
    async_wrap = to_async_embedder(sync_wrap)
    v1 = sync_wrap.embed("hello")
    v2 = asyncio.run(async_wrap.embed("hello"))
    np.testing.assert_array_equal(v1, v2)


def test_sync_to_async_works_with_semantic_cache_via_async_facade(tmp_path):
    """Functional smoke: a sync embedder wrapped via to_async_embedder powers
    AsyncSemanticCache end-to-end."""
    from mneme import AsyncSemanticCache

    sync_e = FakeEmbedder(dim=8)
    async_e = to_async_embedder(sync_e)

    async def run() -> None:
        async with AsyncSemanticCache(path=tmp_path / "c.db", embedder=async_e) as cache:
            await cache.put("hi", "there")
            hit = await cache.get("hi")
            assert hit is not None
            assert hit.response == "there"

    asyncio.run(run())
