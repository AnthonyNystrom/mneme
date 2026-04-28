"""Phase-6 metrics tests: counter bookkeeping + hook dispatch."""

from __future__ import annotations

import logging
from typing import Any

from mneme._metrics import Counters, MetricsDispatcher

# --- Counters ---


def test_counter_starts_at_zero():
    c = Counters()
    assert c.get("default", "hits_exact") == 0


def test_counter_increment_then_get():
    c = Counters()
    c.increment("default", "hits_exact")
    c.increment("default", "hits_exact")
    assert c.get("default", "hits_exact") == 2


def test_counter_increment_with_delta():
    c = Counters()
    c.increment("default", "evictions", delta=5)
    assert c.get("default", "evictions") == 5


def test_counter_increment_zero_or_negative_is_noop():
    c = Counters()
    c.increment("default", "evictions", delta=0)
    c.increment("default", "evictions", delta=-3)
    assert c.get("default", "evictions") == 0


def test_counter_namespace_isolation():
    c = Counters()
    c.increment("tenant_a", "hits_exact", 5)
    c.increment("tenant_b", "hits_exact", 3)
    assert c.get("tenant_a", "hits_exact") == 5
    assert c.get("tenant_b", "hits_exact") == 3


def test_counter_get_namespace_returns_zero_filled():
    c = Counters()
    c.increment("default", "hits_exact", 2)
    ns = c.get_namespace("default")
    assert ns["hits_exact"] == 2
    # Other documented counters present and zero
    assert ns["hits_semantic"] == 0
    assert ns["misses"] == 0
    assert ns["evictions"] == 0
    assert ns["expirations"] == 0


def test_counter_aggregate_sums_across_namespaces():
    c = Counters()
    c.increment("a", "hits_exact", 3)
    c.increment("b", "hits_exact", 2)
    c.increment("a", "misses", 1)
    agg = c.aggregate()
    assert agg["hits_exact"] == 5
    assert agg["misses"] == 1


def test_counter_clear_namespace_drops_only_that_ns():
    c = Counters()
    c.increment("a", "hits_exact", 5)
    c.increment("b", "hits_exact", 3)
    c.clear_namespace("a")
    assert c.get("a", "hits_exact") == 0
    assert c.get("b", "hits_exact") == 3


def test_counter_serialize_round_trip():
    c1 = Counters()
    c1.increment("a", "hits_exact", 5)
    c1.increment("b", "misses", 2)
    serialized = c1.serialize()
    c2 = Counters()
    c2.restore(serialized)
    assert c2.get("a", "hits_exact") == 5
    assert c2.get("b", "misses") == 2


def test_counter_restore_clears_existing_state():
    c = Counters()
    c.increment("a", "hits_exact", 100)
    c.restore({"b:misses": 7})
    assert c.get("a", "hits_exact") == 0
    assert c.get("b", "misses") == 7


# --- MetricsDispatcher ---


def test_dispatcher_no_hook_only_increments_counters():
    d = MetricsDispatcher()
    d.emit_hit("default", "exact", similarity=1.0, confidence=1.0, age_seconds=0)
    assert d.counters.get("default", "hits_exact") == 1


def test_dispatcher_emit_hit_exact_vs_semantic():
    d = MetricsDispatcher()
    d.emit_hit("default", "exact", 1.0, 1.0, 0)
    d.emit_hit("default", "semantic", 0.9, 0.85, 100)
    assert d.counters.get("default", "hits_exact") == 1
    assert d.counters.get("default", "hits_semantic") == 1


def test_dispatcher_emit_miss_increments_misses():
    d = MetricsDispatcher()
    d.emit_miss("default", reason="not_found")
    assert d.counters.get("default", "misses") == 1


def test_dispatcher_emit_eviction_increments_with_count():
    d = MetricsDispatcher()
    d.emit_eviction("default", count=5)
    assert d.counters.get("default", "evictions") == 5


def test_dispatcher_emit_zero_eviction_is_noop():
    d = MetricsDispatcher()
    received: list[tuple[str, dict[str, Any]]] = []
    d.set_hook(lambda e, a: received.append((e, a)))
    d.emit_eviction("default", count=0)
    d.emit_expired("default", count=0)
    assert received == []
    assert d.counters.get("default", "evictions") == 0


def test_dispatcher_hook_receives_documented_attrs():
    received: list[tuple[str, dict[str, Any]]] = []
    d = MetricsDispatcher(hook=lambda e, a: received.append((e, a)))
    d.emit_hit("default", "semantic", similarity=0.9, confidence=0.85, age_seconds=42)
    d.emit_miss("default", reason="below_threshold")
    d.emit_eviction("default", count=3)
    d.emit_expired("default", count=7)

    events = {name: attrs for name, attrs in received}
    assert events["hit"] == {
        "namespace": "default",
        "layer": "semantic",
        "similarity": 0.9,
        "confidence": 0.85,
        "age_seconds": 42,
    }
    assert events["miss"] == {"namespace": "default", "reason": "below_threshold"}
    assert events["eviction"] == {"namespace": "default", "count": 3}
    assert events["expired"] == {"namespace": "default", "count": 7}


def test_dispatcher_hook_exception_caught_and_logged(caplog):
    def hook(event: str, attrs: dict[str, Any]) -> None:
        raise RuntimeError("hook explode")

    d = MetricsDispatcher(hook=hook)
    with caplog.at_level(logging.WARNING, logger="mneme.metrics"):
        d.emit_miss("default", reason="x")  # must not raise
    # Counter still incremented despite hook failure.
    assert d.counters.get("default", "misses") == 1
    # WARNING was logged.
    assert any("hook raised" in record.message for record in caplog.records)


def test_dispatcher_set_hook_swaps_at_runtime():
    received: list[str] = []
    d = MetricsDispatcher()
    d.emit_miss("default", reason="x")
    d.set_hook(lambda e, a: received.append(e))
    d.emit_miss("default", reason="y")
    assert received == ["miss"]


def test_dispatcher_hook_property_reflects_current_hook():
    d = MetricsDispatcher()
    assert d.hook is None
    h = lambda e, a: None  # noqa: E731
    d.set_hook(h)
    assert d.hook is h
