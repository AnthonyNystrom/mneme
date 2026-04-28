"""Phase-7 namespace tests: isolation across get/put/delete/stats/quotas."""

from __future__ import annotations

import time
from pathlib import Path

from mneme import MemoryStore, NamespaceQuotaExceededError, SemanticCache  # noqa: F401

from .fakes import FakeEmbedder


def _cache(tmp_path: Path | None = None, **kwargs: object) -> SemanticCache:
    e = FakeEmbedder(dim=8)
    if tmp_path is None:
        return SemanticCache(store=MemoryStore(), embedder=e, **kwargs)  # type: ignore[arg-type]
    return SemanticCache(path=tmp_path / "c.db", embedder=e, **kwargs)  # type: ignore[arg-type]


# --- Read-side isolation ---


def test_get_does_not_cross_namespaces():
    cache = _cache()
    try:
        cache.put("q", "tenant_a_response", namespace="tenant_a")
        # tenant_b doesn't have this entry.
        assert cache.get("q", namespace="tenant_b") is None
        # tenant_a does.
        hit = cache.get("q", namespace="tenant_a")
        assert hit is not None
        assert hit.response == "tenant_a_response"
        assert hit.namespace == "tenant_a"
    finally:
        cache.close()


def test_same_query_different_responses_per_namespace():
    cache = _cache()
    try:
        cache.put("how to reset password", "answer A", namespace="tenant_a")
        cache.put("how to reset password", "answer B", namespace="tenant_b")
        a = cache.get("how to reset password", namespace="tenant_a")
        b = cache.get("how to reset password", namespace="tenant_b")
        assert a is not None
        assert a.response == "answer A"
        assert b is not None
        assert b.response == "answer B"
    finally:
        cache.close()


# --- Write-side isolation ---


def test_put_in_one_namespace_does_not_appear_in_another():
    cache = _cache()
    try:
        cache.put("q", "r", namespace="t1")
        assert cache.stats(namespace="t1").entries == 1
        assert cache.stats(namespace="t2").entries == 0
    finally:
        cache.close()


def test_delete_only_affects_target_namespace():
    cache = _cache()
    try:
        cache.put("q", "r", namespace="t1")
        cache.put("q", "r", namespace="t2")
        assert cache.delete("q", namespace="t1") is True
        # t2 still holds the entry.
        assert cache.get("q", namespace="t2") is not None
        assert cache.get("q", namespace="t1") is None
    finally:
        cache.close()


def test_clear_namespace_isolated():
    cache = _cache()
    try:
        cache.put("a", "r", namespace="t1")
        cache.put("b", "r", namespace="t2")
        cache.clear_namespace("t1")
        assert cache.stats(namespace="t1").entries == 0
        assert cache.stats(namespace="t2").entries == 1
    finally:
        cache.close()


# --- Quotas ---


def test_namespace_quota_evicts_within_namespace_only():
    cache = _cache(namespace_quotas={"tenant_a": 3})
    try:
        for i in range(6):
            cache.put(f"a{i}", "r", namespace="tenant_a")
            time.sleep(0.001)
        for i in range(5):
            cache.put(f"b{i}", "r", namespace="tenant_b")
        assert cache.stats(namespace="tenant_a").entries <= 3
        assert cache.stats(namespace="tenant_b").entries == 5
    finally:
        cache.close()


def test_namespace_quota_takes_precedence_over_global():
    cache = _cache(namespace_quotas={"tenant_a": 2}, max_entries=100)
    try:
        for i in range(5):
            cache.put(f"q{i}", "r", namespace="tenant_a")
            time.sleep(0.001)
        # Global cap=100 is high, but ns quota=2 is the binding constraint.
        assert cache.stats(namespace="tenant_a").entries <= 2
    finally:
        cache.close()


def test_global_max_entries_when_no_ns_quota_for_target():
    cache = _cache(namespace_quotas={"other": 1000}, max_entries=3)
    try:
        for i in range(5):
            cache.put(f"q{i}", "r", namespace="default")
            time.sleep(0.001)
        assert cache.stats().entries <= 3
    finally:
        cache.close()


# --- Stats ---


def test_stats_namespace_filter_isolated_counters():
    cache = _cache()
    try:
        cache.put("a", "r", namespace="t1")
        cache.put("b", "r", namespace="t2")
        cache.get("a", namespace="t1")  # exact hit on t1
        cache.get("missing", namespace="t1")  # miss on t1
        cache.get("missing", namespace="t2")  # miss on t2
        s1 = cache.stats(namespace="t1")
        s2 = cache.stats(namespace="t2")
        assert s1.hits_exact == 1
        assert s1.misses == 1
        assert s2.hits_exact == 0
        assert s2.misses == 1
    finally:
        cache.close()


# --- list_namespaces ---


def test_list_namespaces_grows_with_puts():
    cache = _cache()
    try:
        assert cache.list_namespaces() == []
        cache.put("q", "r", namespace="alpha")
        cache.put("q", "r", namespace="beta")
        assert cache.list_namespaces() == ["alpha", "beta"]
    finally:
        cache.close()


def test_default_namespace_is_default():
    cache = _cache()
    try:
        cache.put("q", "r")  # no namespace kwarg
        assert cache.list_namespaces() == ["default"]
    finally:
        cache.close()
