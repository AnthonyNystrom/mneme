"""Phase-6 eviction tests: namespace-quota-first then global LRU."""

from __future__ import annotations

import numpy as np

from mneme._eviction import evict_for_namespace, evict_global, maybe_evict
from mneme._store_memory import MemoryStore
from mneme._types import StoredEntry


def _entry(h: str, ns: str = "default", lat: int = 0) -> StoredEntry:
    return StoredEntry(
        id=0,
        namespace=ns,
        query_hash=h,
        query="q",
        response="r",
        embedding=np.zeros(4, dtype=np.float32).tobytes(),
        metadata={},
        created_at=lat,
        last_accessed_at=lat,
        ttl=None,
        access_count=0,
    )


def _make_store() -> MemoryStore:
    s = MemoryStore()
    s.open("fp:v1", 4)
    return s


# --- evict_for_namespace ---


def test_evict_for_namespace_under_quota_is_noop():
    s = _make_store()
    for i in range(3):
        s.insert(_entry(f"{i:064d}", "tenant", i))
    assert evict_for_namespace(s, "tenant", 10) == 0
    assert s.count("tenant") == 3
    s.close()


def test_evict_for_namespace_over_quota_evicts_lru_first():
    s = _make_store()
    # Insert 12 entries with monotonic last_accessed_at; quota = 10.
    for i in range(12):
        s.insert(_entry(f"{i:064d}", "tenant", lat=i))
    evicted = evict_for_namespace(s, "tenant", 10)
    # Batch = max(excess=2, pct=1, MIN=1) = 2 evictions.
    assert evicted == 2
    # The two oldest (i=0 and i=1) should be gone.
    assert s.get_by_hash("tenant", f"{0:064d}") is None
    assert s.get_by_hash("tenant", f"{1:064d}") is None
    # The newest survives.
    assert s.get_by_hash("tenant", f"{11:064d}") is not None
    s.close()


def test_evict_for_namespace_does_not_touch_other_namespaces():
    s = _make_store()
    for i in range(12):
        s.insert(_entry(f"{i:064d}", "a", lat=i))
    for i in range(5):
        s.insert(_entry(f"{i:064d}", "b", lat=i))
    evict_for_namespace(s, "a", 10)
    assert s.count("b") == 5
    s.close()


def test_evict_for_namespace_batch_is_at_least_one():
    """Even tiny quotas evict at least one entry over the cap."""
    s = _make_store()
    for i in range(3):
        s.insert(_entry(f"{i:064d}", "tenant", lat=i))
    evicted = evict_for_namespace(s, "tenant", 2)
    assert evicted >= 1
    assert s.count("tenant") <= 2
    s.close()


def test_evict_for_namespace_with_large_quota_evicts_10_percent():
    s = _make_store()
    for i in range(120):
        s.insert(_entry(f"{i:064d}", "tenant", lat=i))
    # 10% of 100 = 10.
    evicted = evict_for_namespace(s, "tenant", 100)
    assert evicted == 20  # capped at excess=20
    assert s.count("tenant") == 100
    s.close()


# --- evict_global ---


def test_evict_global_under_max_is_noop():
    s = _make_store()
    for i in range(5):
        s.insert(_entry(f"{i:064d}", "default", i))
    assert evict_global(s, 10) == 0
    s.close()


def test_evict_global_over_max_evicts_lru_first():
    s = _make_store()
    # Insert 5 in "a" with older timestamps + 7 in "b" with newer.
    for i in range(5):
        s.insert(_entry(f"a{i:063d}", "a", lat=i))
    for i in range(7):
        s.insert(_entry(f"b{i:063d}", "b", lat=100 + i))
    # max = 10, 12 total → excess = 2.
    evicted = evict_global(s, 10)
    assert evicted == 2
    # The two oldest globally are in "a".
    assert s.count("a") == 3
    assert s.count("b") == 7
    s.close()


# --- maybe_evict (PRD §11.6 policy) ---


def test_maybe_evict_namespace_quota_takes_precedence():
    s = _make_store()
    for i in range(12):
        s.insert(_entry(f"{i:064d}", "tenant", lat=i))
    # Both quotas configured; namespace quota should fire, not global.
    out = maybe_evict(s, "tenant", namespace_quotas={"tenant": 10}, max_entries=1000)
    assert "tenant" in out
    assert out["tenant"] >= 1
    assert s.count("tenant") <= 10
    s.close()


def test_maybe_evict_falls_back_to_global_when_no_ns_quota():
    s = _make_store()
    for i in range(12):
        s.insert(_entry(f"{i:064d}", "default", lat=i))
    out = maybe_evict(s, "default", max_entries=10)
    assert out.get("default", 0) >= 1
    assert s.count() <= 10
    s.close()


def test_maybe_evict_returns_empty_when_no_caps_configured():
    s = _make_store()
    for i in range(20):
        s.insert(_entry(f"{i:064d}", "default", lat=i))
    out = maybe_evict(s, "default")
    assert out == {}
    assert s.count() == 20
    s.close()


def test_maybe_evict_empty_when_under_caps():
    s = _make_store()
    s.insert(_entry("x" * 64, "default", lat=0))
    out = maybe_evict(s, "default", namespace_quotas={"default": 100}, max_entries=1000)
    assert out == {}
    s.close()


def test_maybe_evict_unknown_namespace_uses_global():
    """If the put goes to a namespace without its own quota, global cap applies."""
    s = _make_store()
    for i in range(15):
        s.insert(_entry(f"{i:064d}", "other", lat=i))
    out = maybe_evict(s, "other", namespace_quotas={"tenant_a": 10}, max_entries=10)
    # Global cap fires.
    assert out.get("other", 0) >= 1
    assert s.count() <= 10
    s.close()
