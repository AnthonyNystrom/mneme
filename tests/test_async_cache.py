"""Phase-8 tests: AsyncSemanticCache parity, cancellation, concurrency."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from mneme import (
    AsyncSemanticCache,
    CacheClosedError,
    EmbedderDimensionError,
    Hit,
    MemoryStore,
)

from .fakes import FakeAsyncEmbedder, SlowAsyncEmbedder

# pytest-asyncio is in asyncio_mode=auto via pyproject; bare async functions
# are auto-marked as asyncio tests.


# --- Construction / parity with sync ---


async def test_async_construction_basic(tmp_path: Path):
    cache = AsyncSemanticCache(path=tmp_path / "c.db", embedder=FakeAsyncEmbedder(dim=8))
    try:
        assert cache.health().entries == 0
    finally:
        await cache.close()


async def test_async_embedder_required(tmp_path: Path):
    with pytest.raises(ValueError, match="embedder"):
        AsyncSemanticCache(path=tmp_path / "c.db")


async def test_async_put_then_get_exact(tmp_path: Path):
    async with AsyncSemanticCache(
        path=tmp_path / "c.db", embedder=FakeAsyncEmbedder(dim=8)
    ) as cache:
        await cache.put("hi", "there")
        hit = await cache.get("hi")
        assert hit is not None
        assert hit.layer == "exact"
        assert hit.response == "there"


async def test_async_get_miss_returns_none(tmp_path: Path):
    async with AsyncSemanticCache(
        path=tmp_path / "c.db", embedder=FakeAsyncEmbedder(dim=8)
    ) as cache:
        assert await cache.get("never put") is None


# --- Layer-1 short-circuit (no embedder call when exact match exists) ---


class _CountingAsyncEmbedder:
    """Increments ``calls`` per ``embed()`` invocation; awaits cleanly."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self.calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return "counting:async:v1"

    async def embed(self, text: str) -> np.ndarray:
        self.calls += 1
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(self._dim).astype(np.float32)
        v /= np.linalg.norm(v)
        return v


async def test_async_layer_1_skips_embedder():
    e = _CountingAsyncEmbedder()
    async with AsyncSemanticCache(store=MemoryStore(), embedder=e) as cache:
        await cache.put("hi", "there")  # 1 embed
        assert e.calls == 1
        await cache.get("hi")  # exact match — no embed
        assert e.calls == 1
        await cache.get("hi")  # exact again
        assert e.calls == 1


async def test_async_layer_2_invokes_embedder():
    e = _CountingAsyncEmbedder()
    async with AsyncSemanticCache(store=MemoryStore(), embedder=e) as cache:
        await cache.put("hi", "there")
        assert e.calls == 1
        # A different normalized query must go through layer 2 -> +1 embed.
        await cache.get("totally different query")
        assert e.calls == 2


# --- Caller-supplied embedding skips embedder ---


async def test_async_caller_supplied_embedding_short_circuits():
    e = _CountingAsyncEmbedder()
    async with AsyncSemanticCache(store=MemoryStore(), embedder=e) as cache:
        v = np.zeros(8, dtype=np.float32)
        v[0] = 1.0
        await cache.put("hi", "there", embedding=v)  # no embed call
        await cache.get("missing", embedding=v)  # no embed call
        assert e.calls == 0


# --- Dimension mismatch ---


async def test_async_put_dim_mismatch_raises():
    async with AsyncSemanticCache(store=MemoryStore(), embedder=FakeAsyncEmbedder(dim=8)) as cache:
        bad = np.zeros(16, dtype=np.float32)
        with pytest.raises(EmbedderDimensionError):
            await cache.put("q", "r", embedding=bad)


async def test_async_get_dim_mismatch_returns_miss():
    async with AsyncSemanticCache(store=MemoryStore(), embedder=FakeAsyncEmbedder(dim=8)) as cache:
        await cache.put("q", "r")
        bad = np.zeros(16, dtype=np.float32)
        assert await cache.get("missing", embedding=bad) is None


# --- TTL ---


async def test_async_ttl_expiration():
    async with AsyncSemanticCache(store=MemoryStore(), embedder=FakeAsyncEmbedder(dim=8)) as cache:
        await cache.put("hi", "there", ttl=1)
        await asyncio.sleep(1.1)
        assert await cache.get("hi") is None


# --- Delete / vacuum / clear ---


async def test_async_delete():
    async with AsyncSemanticCache(store=MemoryStore(), embedder=FakeAsyncEmbedder(dim=8)) as cache:
        await cache.put("hi", "there")
        assert await cache.delete("hi") is True
        assert await cache.get("hi") is None


async def test_async_vacuum():
    async with AsyncSemanticCache(store=MemoryStore(), embedder=FakeAsyncEmbedder(dim=8)) as cache:
        await cache.put("alive", "r")
        await cache.put("dead", "r", ttl=1)
        await asyncio.sleep(1.1)
        removed = await cache.vacuum()
        assert removed == 1
        assert await cache.get("alive") is not None


async def test_async_clear_namespace_is_sync_facade():
    async with AsyncSemanticCache(store=MemoryStore(), embedder=FakeAsyncEmbedder(dim=8)) as cache:
        await cache.put("a", "r", namespace="t1")
        await cache.put("b", "r", namespace="t2")
        cleared = cache.clear_namespace("t1")
        assert cleared == 1
        assert cache.stats(namespace="t1").entries == 0


# --- Stats / health pass through ---


async def test_async_stats_pass_through():
    async with AsyncSemanticCache(store=MemoryStore(), embedder=FakeAsyncEmbedder(dim=8)) as cache:
        await cache.put("a", "r")
        await cache.get("a")
        s = cache.stats()
        assert s.entries == 1
        assert s.hits_exact == 1


async def test_async_health_pass_through(tmp_path: Path):
    async with AsyncSemanticCache(
        path=tmp_path / "c.db", embedder=FakeAsyncEmbedder(dim=8)
    ) as cache:
        await cache.put("hi", "there")
        h = cache.health()
        assert h.healthy is True
        assert h.entries == 1


# --- Lifecycle + cancellation ---


async def test_async_context_manager_closes_on_exit(tmp_path: Path):
    cache = AsyncSemanticCache(path=tmp_path / "c.db", embedder=FakeAsyncEmbedder(dim=8))
    async with cache:
        await cache.put("hi", "there")
    with pytest.raises(CacheClosedError):
        await cache.get("hi")


async def test_async_close_is_idempotent(tmp_path: Path):
    cache = AsyncSemanticCache(path=tmp_path / "c.db", embedder=FakeAsyncEmbedder(dim=8))
    await cache.close()
    await cache.close()  # no-op


async def test_async_cancellation_during_embed_safe():
    """Cancelling the embedder coroutine must not corrupt cache state."""
    e = SlowAsyncEmbedder(dim=8, delay_seconds=0.5)
    async with AsyncSemanticCache(store=MemoryStore(), embedder=e) as cache:
        # First put completes normally so the cache has an entry.
        fast_embedder = FakeAsyncEmbedder(dim=8)
        v = await fast_embedder.embed("anchor")
        await cache.put("anchor", "stored", embedding=v)

        # Now start a put with the slow embedder and cancel it mid-await.
        task = asyncio.create_task(cache.put("slow", "won't land"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # State remains consistent: anchor is still there; slow entry is not.
        assert (await cache.get("anchor")) is not None
        assert (await cache.get("slow")) is None


# --- Concurrency: parallel gets ---


async def test_async_concurrent_gets():
    """Multiple awaiters can run in parallel; cache state stays consistent."""
    async with AsyncSemanticCache(
        store=MemoryStore(),
        embedder=SlowAsyncEmbedder(dim=8, delay_seconds=0.02),
    ) as cache:
        # Pre-populate.
        for i in range(20):
            await cache.put(f"q{i}", f"r{i}")
        # Fire 50 concurrent gets across the populated keys.
        results = await asyncio.gather(*(cache.get(f"q{i % 20}") for i in range(50)))
        assert all(isinstance(r, Hit) for r in results)
        # All hit layer 1 (same exact queries).
        assert all(r.layer == "exact" for r in results if r is not None)


# --- Embedder failure ---


class _FailingAsyncEmbedder:
    @property
    def dim(self) -> int:
        return 8

    @property
    def fingerprint(self) -> str:
        return "fail:async:v1"

    async def embed(self, text: str) -> np.ndarray:
        raise RuntimeError("forced async failure")


async def test_async_embedder_failure_during_get_returns_miss():
    async with AsyncSemanticCache(store=MemoryStore(), embedder=_FailingAsyncEmbedder()) as cache:
        # Cold cache + failing embedder -> get returns None (no raise).
        assert await cache.get("anything") is None


async def test_async_embedder_failure_during_put_raises():
    async with AsyncSemanticCache(store=MemoryStore(), embedder=_FailingAsyncEmbedder()) as cache:
        with pytest.raises(RuntimeError, match="forced async failure"):
            await cache.put("q", "r")


# --- Counter persistence parity with sync ---


async def test_async_counters_persist_across_open_close(tmp_path: Path):
    db = tmp_path / "c.db"
    e = FakeAsyncEmbedder(dim=8)
    async with AsyncSemanticCache(path=db, embedder=e) as cache:
        await cache.put("hi", "there")
        await cache.get("hi")
        await cache.get("hi")
    async with AsyncSemanticCache(path=db, embedder=e) as cache:
        s = cache.stats()
        assert s.hits_exact == 2  # carried over


# --- Bypass ---


async def test_async_bypass_returns_none_and_increments_miss(tmp_path: Path):
    async with AsyncSemanticCache(
        path=tmp_path / "c.db", embedder=FakeAsyncEmbedder(dim=8)
    ) as cache:
        await cache.put("hi", "there")
        before = cache.stats().misses
        assert await cache.get("hi", bypass=True) is None
        assert cache.stats().misses == before + 1


# --- Loads stub ---


async def test_async_loads_raises_not_implemented(tmp_path: Path):
    with pytest.raises(NotImplementedError, match="Phase 10"):
        await AsyncSemanticCache.loads(
            tmp_path / "snap.tar.gz",
            tmp_path / "dst.db",
            FakeAsyncEmbedder(dim=8),
        )
