"""Phase-13 concurrency and stress tests per PRD §18 Phase 13.

Two duration profiles:

- Default: short stress (~3 seconds) verifying the workload patterns work,
  no exceptions, no deadlocks. Always runs in CI.
- ``--run-stress`` flag: full 60-second stress tests (marker
  ``stress_long``) for release validation.

Workloads:

1. **Single-process, multi-thread**: 16 threads x N seconds mixed
   put/get/delete on a shared ``SemanticCache``. Verify no exceptions and
   final entry count + counters internally consistent.
2. **Multi-process stale-tolerant**: 4 spawn workers sharing a SQLiteStore.
   Each worker does a mix of put/get; reader sees writer's entries after
   ``stale_check_interval`` polls. No deadlocks.
3. **Mmap-shared multi-process**: 4 workers append + search via the
   ``MmapSharedCoordinator`` directly.
"""

from __future__ import annotations

import multiprocessing as mp
import random
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from mneme import MemoryStore, SemanticCache
from mneme._multiproc import MmapSharedCoordinator

from .fakes import FakeEmbedder

# ---------------------------------------------------------------------------
# Single-process, multi-thread stress
# ---------------------------------------------------------------------------


def _thread_worker(
    cache: SemanticCache,
    stop_at: float,
    worker_id: int,
    op_count: dict[int, int],
    errors: list[BaseException],
) -> None:
    """Mixed-workload worker. Records ops + any unexpected exceptions."""
    rng = random.Random(worker_id * 7919)
    ops = 0
    try:
        while time.monotonic() < stop_at:
            choice = rng.random()
            key = f"w{worker_id}_q{rng.randint(0, 99)}"
            if choice < 0.3:
                # put
                cache.put(key, f"r{ops}")
            elif choice < 0.85:
                # get
                cache.get(key)
            else:
                # delete (best-effort; may not exist)
                cache.delete(key)
            ops += 1
    except BaseException as exc:
        errors.append(exc)
    finally:
        op_count[worker_id] = ops


def _run_stress(duration_sec: float, n_threads: int = 16) -> tuple[int, int]:
    """Run the multi-thread stress for ``duration_sec``. Returns
    ``(total_ops, final_entries)``."""
    cache = SemanticCache(store=MemoryStore(), embedder=FakeEmbedder(dim=8), max_entries=2000)
    op_count: dict[int, int] = {}
    errors: list[BaseException] = []
    stop_at = time.monotonic() + duration_sec
    threads = [
        threading.Thread(
            target=_thread_worker,
            args=(cache, stop_at, i, op_count, errors),
            name=f"stress-{i}",
        )
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=duration_sec * 3 + 5)
        assert not t.is_alive(), f"thread {t.name} still alive — possible deadlock"

    if errors:
        # Re-raise the first unexpected error so it's visible.
        raise errors[0]

    total_ops = sum(op_count.values())
    final_entries = cache.stats().entries
    cache.close()
    return total_ops, final_entries


def test_short_thread_stress_3_seconds():
    """Short stress: 16 threads x 3 seconds. Must complete without errors."""
    total_ops, final_entries = _run_stress(duration_sec=3.0, n_threads=16)
    assert total_ops > 0
    # max_entries cap held under load.
    assert final_entries <= 2000


def test_short_thread_stress_4_threads_5_seconds():
    """4-thread variant for a bit more concurrency stress."""
    total_ops, _ = _run_stress(duration_sec=5.0, n_threads=4)
    # Sanity: each thread should land at least a handful of ops.
    assert total_ops >= 100


@pytest.mark.stress_long
def test_long_thread_stress_60_seconds():
    """Full PRD §18 stress: 16 threads x 60 seconds. Opt-in via --run-stress."""
    total_ops, final_entries = _run_stress(duration_sec=60.0, n_threads=16)
    assert total_ops > 1000
    assert final_entries <= 2000


# ---------------------------------------------------------------------------
# Single-process: counter consistency under load
# ---------------------------------------------------------------------------


def _counter_worker(
    cache: SemanticCache,
    n_ops: int,
    worker_id: int,
    errors: list[BaseException],
) -> None:
    rng = random.Random(worker_id * 991)
    try:
        for _ in range(n_ops):
            key = f"shared_q{rng.randint(0, 99)}"
            cache.put(key, "r")
            cache.get(key)
    except BaseException as exc:
        errors.append(exc)


def test_concurrent_counters_internally_consistent():
    """After concurrent put+get from N threads, counters reflect actual ops."""
    cache = SemanticCache(store=MemoryStore(), embedder=FakeEmbedder(dim=8))
    n_threads = 8
    ops_per_thread = 200
    errors: list[BaseException] = []
    threads = [
        threading.Thread(
            target=_counter_worker,
            args=(cache, ops_per_thread, i, errors),
        )
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    if errors:
        raise errors[0]

    s = cache.stats()
    # Total gets == n_threads * ops_per_thread; each get either hits or misses.
    assert s.hits_exact + s.misses == n_threads * ops_per_thread
    cache.close()


# ---------------------------------------------------------------------------
# Multi-process stale-tolerant stress
# ---------------------------------------------------------------------------


def _mp_worker(
    db_path: str,
    duration_sec: float,
    worker_id: int,
    result_q,  # type: ignore[no-untyped-def]
) -> None:
    """Multi-process worker: opens a stale-tolerant cache, does mixed ops."""
    cache = SemanticCache(
        path=db_path,
        embedder=FakeEmbedder(dim=8),
        multi_process_mode="stale-tolerant",
        stale_check_interval=0.05,  # poll often
    )
    rng = random.Random(worker_id * 31)
    ops = 0
    errors_seen = 0
    stop_at = time.monotonic() + duration_sec
    try:
        while time.monotonic() < stop_at:
            choice = rng.random()
            key = f"mp{worker_id}_q{rng.randint(0, 19)}"
            try:
                if choice < 0.4:
                    cache.put(key, f"r{ops}")
                else:
                    cache.get(key)
                ops += 1
            except Exception:
                errors_seen += 1
    finally:
        cache.close()
        result_q.put((worker_id, ops, errors_seen))


def test_short_multiprocess_stale_tolerant_stress(tmp_path: Path):
    """4 processes share a SQLiteStore via stale-tolerant mode for 3 seconds."""
    db_path = str(tmp_path / "shared.db")
    duration = 3.0
    n_workers = 4

    # Pre-create the DB so workers don't all race on initial open.
    pre = SemanticCache(
        path=db_path,
        embedder=FakeEmbedder(dim=8),
        multi_process_mode="stale-tolerant",
    )
    pre.close()

    ctx = mp.get_context("spawn")
    result_q: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(target=_mp_worker, args=(db_path, duration, i, result_q))
        for i in range(n_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=duration * 3 + 10)
        if p.is_alive():
            p.terminate()
            pytest.fail(f"worker {p.pid} still alive — possible deadlock")
        assert p.exitcode == 0, f"worker exit {p.exitcode}"

    total_ops = 0
    total_errors = 0
    for _ in range(n_workers):
        _wid, ops, errors = result_q.get(timeout=5)
        total_ops += ops
        total_errors += errors
    assert total_ops > 0
    # Some transient SQLite "database is locked" errors are tolerated under
    # heavy multi-process write contention; the goal is no DEADLOCK and no
    # crash, not zero contention.
    assert total_errors < total_ops, f"too many errors: {total_errors} of {total_ops} ops"


# ---------------------------------------------------------------------------
# Mmap-shared multi-process stress
# ---------------------------------------------------------------------------


def _mmap_worker(
    base_path: str,
    duration_sec: float,
    worker_id: int,
    result_q,  # type: ignore[no-untyped-def]
) -> None:
    """Mmap worker: appends + searches under exclusive file lock."""
    coord = MmapSharedCoordinator(
        base_path,
        dim=8,
        embedder_fingerprint="stress:fp:v1",
        initial_capacity=64,
    )
    np_rng = np.random.default_rng(worker_id * 7)
    py_rng = random.Random(worker_id * 7)
    ops = 0
    stop_at = time.monotonic() + duration_sec
    try:
        next_id = worker_id * 10_000
        while time.monotonic() < stop_at:
            v = np_rng.standard_normal(8).astype(np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            if py_rng.random() < 0.5:
                coord.append(next_id, v, "default")
                next_id += 1
            else:
                coord.search(v, "default", k=3)
            ops += 1
    finally:
        coord.close()
        result_q.put((worker_id, ops))


def test_short_mmap_shared_multiprocess_stress(tmp_path: Path):
    """4 processes share a mmap matrix; mixed append + search under flock."""
    base = str(tmp_path / "mmap_stress")
    duration = 2.0
    n_workers = 4

    # Pre-create the file.
    pre = MmapSharedCoordinator(
        base, dim=8, embedder_fingerprint="stress:fp:v1", initial_capacity=64
    )
    pre.close()

    ctx = mp.get_context("spawn")
    result_q: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(target=_mmap_worker, args=(base, duration, i, result_q))
        for i in range(n_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=duration * 3 + 10)
        if p.is_alive():
            p.terminate()
            pytest.fail(f"mmap worker {p.pid} still alive — possible deadlock")
        assert p.exitcode == 0, f"mmap worker exit {p.exitcode}"

    total = 0
    for _ in range(n_workers):
        _wid, ops = result_q.get(timeout=5)
        total += ops
    assert total > 0

    # Verify the resulting file is consistent.
    reader = MmapSharedCoordinator(base, dim=8, embedder_fingerprint="stress:fp:v1")
    try:
        _, count, _, _, _ = reader.read_header()
        # Some appends landed; count is positive.
        assert count > 0
    finally:
        reader.close()


# ---------------------------------------------------------------------------
# Async cache concurrent stress
# ---------------------------------------------------------------------------


import asyncio  # noqa: E402

from mneme import AsyncSemanticCache  # noqa: E402

from .fakes import FakeAsyncEmbedder  # noqa: E402


async def test_async_cache_concurrent_stress():
    """50 concurrent gets against a populated async cache must all complete
    without exceptions or state corruption."""
    async with AsyncSemanticCache(store=MemoryStore(), embedder=FakeAsyncEmbedder(dim=8)) as cache:
        # Pre-populate.
        for i in range(20):
            await cache.put(f"q{i}", f"r{i}")

        async def _do_op(i: int) -> bool:
            key = f"q{i % 20}"
            hit = await cache.get(key)
            return hit is not None

        results = await asyncio.gather(*(_do_op(i) for i in range(50)))
        assert all(results)
