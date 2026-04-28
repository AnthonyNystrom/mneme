"""Phase-9 stale-tolerant multi-process tests.

Two-process tests use ``multiprocessing.spawn`` so workers don't inherit
state from the parent. Both connect to the same SQLiteStore file. Writes
in one process become visible in the other after the second process'
``version_counter`` poll fires.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import pytest

from mneme import MemoryStore, SemanticCache
from mneme._multiproc import StaleTolerantCoordinator

from .fakes import FakeEmbedder

# --- Direct coordinator unit tests ---


def _make_cache_for_coordinator(store: MemoryStore) -> SemanticCache:
    """Construct a SemanticCache that shares ``store``."""
    return SemanticCache(store=store, embedder=FakeEmbedder(dim=8))


def test_coordinator_no_refresh_when_version_unchanged():
    store = MemoryStore()
    cache = _make_cache_for_coordinator(store)
    try:
        coord = StaleTolerantCoordinator(store, cache._index)
        # Counter is at 0; no refresh should happen.
        assert coord.refresh() == 0
    finally:
        cache.close()


def test_coordinator_incremental_sync_when_below_threshold():
    """A handful of new entries in the store get pulled in via iter_since."""
    store = MemoryStore()
    cache_a = _make_cache_for_coordinator(store)
    try:
        # coord_b polls the SAME store as cache_a. It starts at the current
        # version (0); cache_a's puts will bump the counter and coord_b will
        # stream them in on the next refresh.
        coord_b = StaleTolerantCoordinator(store, cache_a._index)
        cache_a.put("q1", "r1")
        cache_a.put("q2", "r2")
        added = coord_b.refresh()
        # incremental_sync streams 2 entries (idempotent against the shared index).
        assert added == 2
        assert coord_b.local_version >= 2
    finally:
        cache_a.close()


def test_coordinator_full_rebuild_above_threshold():
    """When delta >= REBUILD_THRESHOLD, do a full rebuild instead of streaming."""
    store = MemoryStore()
    cache = _make_cache_for_coordinator(store)
    try:
        # Build a fresh coordinator BEFORE any puts.
        coord = StaleTolerantCoordinator(store, cache._index)
        # Now do many puts to push the version counter past REBUILD_THRESHOLD.
        n = StaleTolerantCoordinator.REBUILD_THRESHOLD + 5
        for i in range(n):
            cache.put(f"q{i:03d}", "r")
        added = coord.refresh()
        # Full rebuild path: the index already reflected those puts (cache_a
        # appended directly to the same index instance), so refresh sees no
        # NEW state. Returned count is from rebuild_from(rows): it's the
        # current count in the store.
        assert added == n
        assert coord.local_version > 0
    finally:
        cache.close()


def test_coordinator_throttle_via_stale_check_interval():
    """If less than ``stale_check_interval`` has elapsed, refresh is a no-op."""
    store = MemoryStore()
    cache = _make_cache_for_coordinator(store)
    try:
        coord = StaleTolerantCoordinator(store, cache._index, stale_check_interval=10.0)
        # First refresh checks; subsequent within 10s should skip even if
        # the version bumps.
        coord.refresh()
        cache.put("q", "r")
        assert coord.refresh() == 0  # throttled
    finally:
        cache.close()


# --- Cache-level: stale-tolerant mode wires the coordinator ---


def test_stale_tolerant_mode_attaches_coordinator(tmp_path: Path):
    cache = SemanticCache(
        path=tmp_path / "c.db",
        embedder=FakeEmbedder(dim=8),
        multi_process_mode="stale-tolerant",
    )
    try:
        assert cache._coordinator is not None
        # Type-check via attribute presence
        assert hasattr(cache._coordinator, "refresh")
    finally:
        cache.close()


def test_single_mode_has_no_coordinator(tmp_path: Path):
    cache = SemanticCache(
        path=tmp_path / "c.db",
        embedder=FakeEmbedder(dim=8),
        multi_process_mode="single",
    )
    try:
        assert cache._coordinator is None
    finally:
        cache.close()


# --- Multi-process integration via SQLiteStore ---


def _writer_worker(db_path: str, count: int, ready_evt, done_evt) -> None:  # type: ignore[no-untyped-def]
    """Open a cache, wait for the reader to attach, then write ``count``
    entries and close."""
    cache = SemanticCache(
        path=db_path,
        embedder=FakeEmbedder(dim=8),
        multi_process_mode="stale-tolerant",
    )
    try:
        ready_evt.set()
        for i in range(count):
            cache.put(f"writer{i:04d}", f"response{i}")
            time.sleep(0.001)
        done_evt.set()
    finally:
        cache.close()


def _reader_worker(  # type: ignore[no-untyped-def]
    db_path: str, ready_evt, done_evt, result_q
) -> None:
    """Open a cache, wait for writer, then poll until it sees writer's entries."""
    cache = SemanticCache(
        path=db_path,
        embedder=FakeEmbedder(dim=8),
        multi_process_mode="stale-tolerant",
    )
    try:
        ready_evt.wait(timeout=5.0)
        done_evt.wait(timeout=5.0)
        # After writer is done, every put should be visible.
        seen = 0
        for _ in range(50):  # poll up to 5 seconds
            cache._refresh_coordinator()  # force sync
            count = (cache.list_namespaces() and cache.stats().entries) or 0
            if count >= 5:
                seen = count
                break
            time.sleep(0.1)
        result_q.put(seen)
    finally:
        cache.close()


def test_two_processes_share_state_via_sqlite(tmp_path: Path):
    """Writer + reader processes share a SQLiteStore.

    The reader sees the writer's entries after refresh.
    """
    db_path = str(tmp_path / "shared.db")

    ctx = mp.get_context("spawn")
    ready_evt = ctx.Event()
    done_evt = ctx.Event()
    result_q: mp.Queue[int] = ctx.Queue()

    writer = ctx.Process(target=_writer_worker, args=(db_path, 5, ready_evt, done_evt))
    reader = ctx.Process(target=_reader_worker, args=(db_path, ready_evt, done_evt, result_q))
    writer.start()
    reader.start()
    writer.join(timeout=15)
    reader.join(timeout=15)

    if writer.is_alive():
        writer.terminate()
        pytest.fail("writer process hung")
    if reader.is_alive():
        reader.terminate()
        pytest.fail("reader process hung")

    assert writer.exitcode == 0, f"writer exit {writer.exitcode}"
    assert reader.exitcode == 0, f"reader exit {reader.exitcode}"

    seen = result_q.get(timeout=2.0)
    assert seen >= 5, f"reader only saw {seen} entries from writer"
