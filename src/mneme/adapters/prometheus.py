"""Prometheus metrics adapter per PRD §15.3.

Exposes a ``PrometheusMetricsHook`` callable that satisfies the
``MetricsHook`` Protocol. Pass it as ``metrics_hook=`` to ``SemanticCache``
or ``AsyncSemanticCache`` and the cache's events become Prometheus
counters and histograms with the documented metric names.

Counters/histograms update on every event. Gauges (``mneme_entries``,
``mneme_memory_bytes``) cannot be derived from event attributes alone —
call ``hook.update_gauges(stats)`` periodically with a fresh ``cache.stats()``
to refresh them (e.g. from a readiness-probe handler).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._exceptions import MnemeError

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

    from .._types import Stats


def _import_prometheus() -> Any:
    try:
        import prometheus_client
    except ImportError as exc:  # pragma: no cover
        raise MnemeError(
            "PrometheusMetricsHook requires the optional 'prometheus' extra. "
            "Remediation: pip install mneme[prometheus]"
        ) from exc
    return prometheus_client


# PRD §15.3 bucket boundaries.
_SIMILARITY_BUCKETS = (0.5, 0.7, 0.85, 0.9, 0.95, 0.99, 1.0)
_AGE_BUCKETS = (60.0, 600.0, 3600.0, 86400.0, 604800.0)


class PrometheusMetricsHook:
    """Callable hook that emits the documented Prometheus metric series.

    Construct with an explicit ``CollectorRegistry`` to keep mneme's metrics
    isolated from the global default registry (recommended for tests and
    multi-tenant exposition).
    """

    def __init__(
        self,
        registry: CollectorRegistry | None = None,
        namespace: str = "mneme",
    ) -> None:
        prom = _import_prometheus()
        self._namespace = namespace
        self._registry = registry

        # Build all metrics up-front.
        common_kwargs: dict[str, Any] = {}
        if registry is not None:
            common_kwargs["registry"] = registry

        self._hits = prom.Counter(
            f"{namespace}_hits_total",
            "Total cache hits.",
            labelnames=("layer", "namespace"),
            **common_kwargs,
        )
        self._misses = prom.Counter(
            f"{namespace}_misses_total",
            "Total cache misses.",
            labelnames=("reason", "namespace"),
            **common_kwargs,
        )
        self._evictions = prom.Counter(
            f"{namespace}_evictions_total",
            "Total entries evicted by LRU.",
            labelnames=("namespace",),
            **common_kwargs,
        )
        self._expirations = prom.Counter(
            f"{namespace}_expirations_total",
            "Total entries expired by TTL.",
            labelnames=("namespace",),
            **common_kwargs,
        )
        self._similarity = prom.Histogram(
            f"{namespace}_hit_similarity",
            "Cosine similarity of cache hits.",
            labelnames=("namespace",),
            buckets=_SIMILARITY_BUCKETS,
            **common_kwargs,
        )
        self._age = prom.Histogram(
            f"{namespace}_hit_age_seconds",
            "Age of cache hits in seconds.",
            labelnames=("namespace",),
            buckets=_AGE_BUCKETS,
            **common_kwargs,
        )
        self._entries = prom.Gauge(
            f"{namespace}_entries",
            "Number of entries currently in the cache.",
            labelnames=("namespace",),
            **common_kwargs,
        )
        self._memory_bytes = prom.Gauge(
            f"{namespace}_memory_bytes",
            "Approximate index memory footprint in bytes.",
            labelnames=("namespace",),
            **common_kwargs,
        )

    @property
    def registry(self) -> CollectorRegistry | None:
        return self._registry

    def __call__(self, event: str, attrs: dict[str, Any]) -> None:
        ns = str(attrs.get("namespace", "default"))
        if event == "hit":
            self._hits.labels(layer=str(attrs.get("layer", "unknown")), namespace=ns).inc()
            sim = attrs.get("similarity")
            if sim is not None:
                self._similarity.labels(namespace=ns).observe(float(sim))
            age = attrs.get("age_seconds")
            if age is not None:
                self._age.labels(namespace=ns).observe(float(age))
        elif event == "miss":
            self._misses.labels(reason=str(attrs.get("reason", "unknown")), namespace=ns).inc()
        elif event == "eviction":
            count = int(attrs.get("count", 0))
            if count > 0:
                self._evictions.labels(namespace=ns).inc(count)
        elif event == "expired":
            count = int(attrs.get("count", 0))
            if count > 0:
                self._expirations.labels(namespace=ns).inc(count)
        # Unknown events are silently ignored — same posture as the dispatcher
        # itself, which catches hook exceptions and logs WARNING.

    def update_gauges(self, stats: Stats) -> None:
        """Refresh the entry-count and memory gauges from a ``cache.stats()``
        snapshot. Call from a periodic task or readiness handler."""
        ns = stats.namespace if stats.namespace is not None else "_total"
        self._entries.labels(namespace=ns).set(float(stats.entries))
        self._memory_bytes.labels(namespace=ns).set(float(stats.memory_bytes_estimate))


__all__ = ["PrometheusMetricsHook"]
