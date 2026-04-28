"""Phase-12 Prometheus adapter tests."""

from __future__ import annotations

import pytest

# Skip the whole module if the optional extra is not installed.
prom = pytest.importorskip("prometheus_client")

from mneme import MemoryStore, SemanticCache  # noqa: E402
from mneme._types import Stats  # noqa: E402
from mneme.adapters.prometheus import PrometheusMetricsHook  # noqa: E402

from .fakes import FakeEmbedder  # noqa: E402


def _registry() -> prom.CollectorRegistry:
    return prom.CollectorRegistry()


def _value(reg: prom.CollectorRegistry, name: str, **labels: str) -> float:
    """Read a metric sample by name + labels from the registry."""
    return float(reg.get_sample_value(name, labels) or 0.0)


# --- Construction ---


def test_construct_with_explicit_registry():
    reg = _registry()
    hook = PrometheusMetricsHook(registry=reg, namespace="test_ns")
    assert hook.registry is reg


def test_construct_with_default_namespace():
    reg = _registry()
    hook = PrometheusMetricsHook(registry=reg)
    # Metric exists under the documented namespace.
    hook("miss", {"reason": "x", "namespace": "default"})
    assert _value(reg, "mneme_misses_total", reason="x", namespace="default") == 1.0


# --- Counter events ---


def test_hit_increments_counter_and_observes_histograms():
    reg = _registry()
    hook = PrometheusMetricsHook(registry=reg)
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
    assert _value(reg, "mneme_hits_total", layer="exact", namespace="default") == 1.0
    # Histograms expose _count / _sum samples.
    assert _value(reg, "mneme_hit_similarity_count", namespace="default") == 1.0
    assert _value(reg, "mneme_hit_age_seconds_count", namespace="default") == 1.0


def test_hit_distinguishes_layer_label():
    reg = _registry()
    hook = PrometheusMetricsHook(registry=reg)
    hook("hit", {"layer": "exact", "similarity": 1.0, "age_seconds": 0, "namespace": "ns1"})
    hook("hit", {"layer": "semantic", "similarity": 0.9, "age_seconds": 60, "namespace": "ns1"})
    assert _value(reg, "mneme_hits_total", layer="exact", namespace="ns1") == 1.0
    assert _value(reg, "mneme_hits_total", layer="semantic", namespace="ns1") == 1.0


def test_miss_increments_counter_with_reason():
    reg = _registry()
    hook = PrometheusMetricsHook(registry=reg)
    hook("miss", {"reason": "no_match", "namespace": "default"})
    hook("miss", {"reason": "below_threshold", "namespace": "default"})
    assert _value(reg, "mneme_misses_total", reason="no_match", namespace="default") == 1.0
    assert _value(reg, "mneme_misses_total", reason="below_threshold", namespace="default") == 1.0


def test_eviction_increments_by_count():
    reg = _registry()
    hook = PrometheusMetricsHook(registry=reg)
    hook("eviction", {"count": 5, "namespace": "tenant_a"})
    assert _value(reg, "mneme_evictions_total", namespace="tenant_a") == 5.0


def test_eviction_zero_count_is_noop():
    reg = _registry()
    hook = PrometheusMetricsHook(registry=reg)
    hook("eviction", {"count": 0, "namespace": "default"})
    assert _value(reg, "mneme_evictions_total", namespace="default") == 0.0


def test_expired_increments_by_count():
    reg = _registry()
    hook = PrometheusMetricsHook(registry=reg)
    hook("expired", {"count": 3, "namespace": "default"})
    assert _value(reg, "mneme_expirations_total", namespace="default") == 3.0


def test_unknown_event_silently_ignored():
    reg = _registry()
    hook = PrometheusMetricsHook(registry=reg)
    hook("totally_unknown", {"namespace": "default"})  # must not raise


# --- Histograms ---


def test_similarity_buckets_match_prd_spec():
    """PRD §15.3: buckets = 0.5, 0.7, 0.85, 0.9, 0.95, 0.99, 1.0."""
    reg = _registry()
    hook = PrometheusMetricsHook(registry=reg)
    hook(
        "hit",
        {"layer": "semantic", "similarity": 0.92, "age_seconds": 0, "namespace": "default"},
    )
    # The bucket for 0.92 should be at the 0.95 boundary.
    # prometheus_client labels histograms with `le` (less-or-equal) buckets.
    samples = list(reg.collect())
    sim_metric = next(m for m in samples if m.name == "mneme_hit_similarity")
    le_values = sorted(
        float(s.labels["le"])
        for s in sim_metric.samples
        if s.name.endswith("_bucket") and s.labels.get("le", "+Inf") != "+Inf"
    )
    expected = [0.5, 0.7, 0.85, 0.9, 0.95, 0.99, 1.0]
    assert le_values == expected


def test_age_buckets_match_prd_spec():
    """PRD §15.3: age buckets = 60, 600, 3600, 86400, 604800."""
    reg = _registry()
    hook = PrometheusMetricsHook(registry=reg)
    hook(
        "hit",
        {"layer": "exact", "similarity": 1.0, "age_seconds": 100, "namespace": "default"},
    )
    samples = list(reg.collect())
    age_metric = next(m for m in samples if m.name == "mneme_hit_age_seconds")
    le_values = sorted(
        float(s.labels["le"])
        for s in age_metric.samples
        if s.name.endswith("_bucket") and s.labels.get("le", "+Inf") != "+Inf"
    )
    expected = [60.0, 600.0, 3600.0, 86400.0, 604800.0]
    assert le_values == expected


# --- Gauges via update_gauges ---


def test_update_gauges_sets_entries_and_memory():
    reg = _registry()
    hook = PrometheusMetricsHook(registry=reg)
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
    assert _value(reg, "mneme_entries", namespace="tenant_a") == 42.0
    assert _value(reg, "mneme_memory_bytes", namespace="tenant_a") == 12_345.0


def test_update_gauges_aggregate_uses_total_label():
    reg = _registry()
    hook = PrometheusMetricsHook(registry=reg)
    stats = Stats(
        namespace=None,  # aggregate
        entries=10,
        hits_exact=0,
        hits_semantic=0,
        misses=0,
        evictions=0,
        expirations=0,
        embedder_fingerprint="fp",
        vector_dtype="float32",
        memory_bytes_estimate=0,
    )
    hook.update_gauges(stats)
    assert _value(reg, "mneme_entries", namespace="_total") == 10.0


# --- End-to-end with cache ---


def test_end_to_end_cache_emits_to_prometheus(tmp_path):
    reg = _registry()
    hook = PrometheusMetricsHook(registry=reg)
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e, metrics_hook=hook) as cache:
        cache.put("hi", "there")
        cache.get("hi")  # exact hit
        cache.get("hi")  # exact hit
        cache.get("missing")  # miss
    assert _value(reg, "mneme_hits_total", layer="exact", namespace="default") == 2.0
    misses = sum(
        v
        for label_set, v in [
            ((s.labels.get("reason"), s.labels.get("namespace")), s.value)
            for m in reg.collect()
            for s in m.samples
            if s.name == "mneme_misses_total"
        ]
        if label_set[1] == "default"
    )
    assert misses >= 1.0


# --- Custom namespace prefix ---


def test_namespace_argument_renames_metric_prefix():
    reg = _registry()
    hook = PrometheusMetricsHook(registry=reg, namespace="myapp")
    hook("miss", {"reason": "x", "namespace": "default"})
    # Metric name uses the constructor's namespace as the prefix.
    assert (
        reg.get_sample_value("myapp_misses_total", {"reason": "x", "namespace": "default"}) == 1.0
    )
