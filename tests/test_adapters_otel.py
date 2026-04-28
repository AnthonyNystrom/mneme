"""Phase-12 OpenTelemetry adapter tests.

Uses ``InMemoryMetricReader`` from ``opentelemetry.sdk.metrics.export`` so
we can collect metric points without a backend.
"""

from __future__ import annotations

from typing import Any

import pytest

# Skip the whole module if otel isn't installed.
pytest.importorskip("opentelemetry")
pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from mneme import MemoryStore, SemanticCache
from mneme._types import Stats
from mneme.adapters.opentelemetry import OTelMetricsHook

from .fakes import FakeEmbedder


def _make_meter() -> tuple[Any, InMemoryMetricReader, MeterProvider]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("mneme_test")
    return meter, reader, provider


def _collect(reader: InMemoryMetricReader) -> dict[str, list[Any]]:
    """Collect metrics into a {metric_name: [data_points...]} mapping."""
    data = reader.get_metrics_data()
    out: dict[str, list[Any]] = {}
    if data is None:
        return out
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                pts = list(metric.data.data_points)
                out.setdefault(metric.name, []).extend(pts)
    return out


def _attr_value(point: Any, key: str) -> str | None:
    return dict(point.attributes).get(key)


# --- Construction ---


def test_construct_with_explicit_meter():
    meter, _reader, provider = _make_meter()
    hook = OTelMetricsHook(meter=meter, namespace="test_ns")
    assert hook.meter is meter
    provider.shutdown()


def test_construct_with_default_meter_provider():
    """Without an explicit meter, the global MeterProvider is used."""
    hook = OTelMetricsHook(namespace="default_meter_ns")
    assert hook.meter is not None


# --- Counter events ---


def test_hit_emits_counter_and_histograms():
    meter, reader, provider = _make_meter()
    hook = OTelMetricsHook(meter=meter)
    hook(
        "hit",
        {
            "layer": "exact",
            "similarity": 0.95,
            "confidence": 0.9,
            "age_seconds": 30,
            "namespace": "default",
        },
    )
    metrics = _collect(reader)
    assert "mneme_hits_total" in metrics
    hit_pts = metrics["mneme_hits_total"]
    assert any(
        p.value == 1
        and _attr_value(p, "layer") == "exact"
        and _attr_value(p, "namespace") == "default"
        for p in hit_pts
    )
    assert "mneme_hit_similarity" in metrics
    assert "mneme_hit_age_seconds" in metrics
    provider.shutdown()


def test_hit_distinguishes_layer_label():
    meter, reader, provider = _make_meter()
    hook = OTelMetricsHook(meter=meter)
    hook("hit", {"layer": "exact", "similarity": 1.0, "age_seconds": 0, "namespace": "ns1"})
    hook("hit", {"layer": "semantic", "similarity": 0.9, "age_seconds": 60, "namespace": "ns1"})
    metrics = _collect(reader)
    layers = {_attr_value(p, "layer") for p in metrics["mneme_hits_total"]}
    assert {"exact", "semantic"} <= layers
    provider.shutdown()


def test_miss_emits_counter_with_reason():
    meter, reader, provider = _make_meter()
    hook = OTelMetricsHook(meter=meter)
    hook("miss", {"reason": "no_match", "namespace": "default"})
    hook("miss", {"reason": "below_threshold", "namespace": "default"})
    metrics = _collect(reader)
    reasons = {_attr_value(p, "reason") for p in metrics["mneme_misses_total"]}
    assert {"no_match", "below_threshold"} <= reasons
    provider.shutdown()


def test_eviction_increments_by_count():
    meter, reader, provider = _make_meter()
    hook = OTelMetricsHook(meter=meter)
    hook("eviction", {"count": 5, "namespace": "tenant_a"})
    metrics = _collect(reader)
    point = next(
        p for p in metrics["mneme_evictions_total"] if _attr_value(p, "namespace") == "tenant_a"
    )
    assert point.value == 5
    provider.shutdown()


def test_eviction_zero_count_is_noop():
    meter, reader, provider = _make_meter()
    hook = OTelMetricsHook(meter=meter)
    hook("eviction", {"count": 0, "namespace": "default"})
    metrics = _collect(reader)
    # The metric exists but has no data points (or value 0).
    points = metrics.get("mneme_evictions_total", [])
    assert all(p.value == 0 for p in points)
    provider.shutdown()


def test_expired_increments_by_count():
    meter, reader, provider = _make_meter()
    hook = OTelMetricsHook(meter=meter)
    hook("expired", {"count": 3, "namespace": "default"})
    metrics = _collect(reader)
    point = next(p for p in metrics["mneme_expirations_total"])
    assert point.value == 3
    provider.shutdown()


def test_unknown_event_silently_ignored():
    meter, _reader, provider = _make_meter()
    hook = OTelMetricsHook(meter=meter)
    hook("totally_unknown", {"namespace": "default"})  # must not raise
    provider.shutdown()


# --- Observable gauges via update_gauges ---


def test_update_gauges_observable_callback_returns_state():
    meter, reader, provider = _make_meter()
    hook = OTelMetricsHook(meter=meter)
    stats = Stats(
        namespace="tenant_a",
        entries=42,
        hits_exact=0,
        hits_semantic=0,
        misses=0,
        evictions=0,
        expirations=0,
        embedder_fingerprint="fp",
        vector_dtype="float32",
        memory_bytes_estimate=12_345,
    )
    hook.update_gauges(stats)
    metrics = _collect(reader)
    entry_points = metrics.get("mneme_entries", [])
    assert any(_attr_value(p, "namespace") == "tenant_a" and p.value == 42 for p in entry_points)
    mem_points = metrics.get("mneme_memory_bytes", [])
    assert any(_attr_value(p, "namespace") == "tenant_a" and p.value == 12_345 for p in mem_points)
    provider.shutdown()


# --- End-to-end with cache ---


def test_end_to_end_cache_emits_to_otel():
    meter, reader, provider = _make_meter()
    hook = OTelMetricsHook(meter=meter)
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e, metrics_hook=hook) as cache:
        cache.put("hi", "there")
        cache.get("hi")  # exact hit
        cache.get("hi")  # exact hit
        cache.get("missing")  # miss
    metrics = _collect(reader)
    # Two exact hits.
    exact_hits = [
        p
        for p in metrics.get("mneme_hits_total", [])
        if _attr_value(p, "layer") == "exact" and _attr_value(p, "namespace") == "default"
    ]
    assert len(exact_hits) >= 1
    assert sum(p.value for p in exact_hits) == 2
    # At least one miss point.
    miss_points = metrics.get("mneme_misses_total", [])
    assert any(_attr_value(p, "namespace") == "default" for p in miss_points)
    provider.shutdown()
