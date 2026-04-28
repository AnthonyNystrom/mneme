"""OpenTelemetry metrics adapter per PRD §15.3.

Exposes ``OTelMetricsHook`` with the same metric names and labels as the
Prometheus adapter. Pass it as ``metrics_hook=`` to the cache and events
become OTel counter / histogram / gauge instruments under your application's
``MeterProvider``.

For histogram bucket boundaries on a specific instrument, configure a
``View`` on your MeterProvider with ``ExplicitBucketHistogramAggregation``;
the OTel API doesn't support per-instrument buckets at construction time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._exceptions import MnemeError

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter

    from .._types import Stats


def _import_metrics() -> Any:
    try:
        from opentelemetry import metrics
    except ImportError as exc:  # pragma: no cover
        raise MnemeError(
            "OTelMetricsHook requires the optional 'otel' extra. "
            "Remediation: pip install mneme[otel]"
        ) from exc
    return metrics


class OTelMetricsHook:
    """Callable hook that records OpenTelemetry metrics.

    Pass an explicit ``Meter`` (e.g. from ``MeterProvider.get_meter(...)``)
    to keep mneme's metrics under a known scope. If omitted, a meter is
    obtained from the global ``MeterProvider``.
    """

    def __init__(
        self,
        meter: Meter | None = None,
        namespace: str = "mneme",
    ) -> None:
        if meter is None:
            metrics_mod = _import_metrics()
            meter = metrics_mod.get_meter("mneme")
        self._meter = meter
        self._namespace = namespace

        self._hits = meter.create_counter(
            name=f"{namespace}_hits_total",
            description="Total cache hits.",
            unit="1",
        )
        self._misses = meter.create_counter(
            name=f"{namespace}_misses_total",
            description="Total cache misses.",
            unit="1",
        )
        self._evictions = meter.create_counter(
            name=f"{namespace}_evictions_total",
            description="Total entries evicted by LRU.",
            unit="1",
        )
        self._expirations = meter.create_counter(
            name=f"{namespace}_expirations_total",
            description="Total entries expired by TTL.",
            unit="1",
        )
        self._similarity = meter.create_histogram(
            name=f"{namespace}_hit_similarity",
            description="Cosine similarity of cache hits.",
            unit="1",
        )
        self._age = meter.create_histogram(
            name=f"{namespace}_hit_age_seconds",
            description="Age of cache hits in seconds.",
            unit="s",
        )
        # Most recently observed values (set by update_gauges); the
        # ObservableGauge callbacks read these in.
        self._gauge_state: dict[str, dict[str, float]] = {
            "entries": {},
            "memory_bytes": {},
        }
        self._entries_gauge = meter.create_observable_gauge(
            name=f"{namespace}_entries",
            description="Number of entries currently in the cache.",
            unit="1",
            callbacks=[self._make_gauge_callback("entries")],
        )
        self._memory_gauge = meter.create_observable_gauge(
            name=f"{namespace}_memory_bytes",
            description="Approximate index memory footprint in bytes.",
            unit="By",
            callbacks=[self._make_gauge_callback("memory_bytes")],
        )

    def _make_gauge_callback(self, key: str) -> Any:
        # Lazy import to avoid pulling otel.metrics symbols when not needed.
        from opentelemetry.metrics import (
            CallbackOptions,
            Observation,
        )

        state = self._gauge_state[key]

        def _cb(_options: CallbackOptions) -> list[Observation]:
            return [Observation(value=v, attributes={"namespace": ns}) for ns, v in state.items()]

        return _cb

    @property
    def meter(self) -> Meter:
        return self._meter

    def __call__(self, event: str, attrs: dict[str, Any]) -> None:
        ns = str(attrs.get("namespace", "default"))
        if event == "hit":
            self._hits.add(
                1,
                attributes={
                    "layer": str(attrs.get("layer", "unknown")),
                    "namespace": ns,
                },
            )
            sim = attrs.get("similarity")
            if sim is not None:
                self._similarity.record(float(sim), attributes={"namespace": ns})
            age = attrs.get("age_seconds")
            if age is not None:
                self._age.record(float(age), attributes={"namespace": ns})
        elif event == "miss":
            self._misses.add(
                1,
                attributes={
                    "reason": str(attrs.get("reason", "unknown")),
                    "namespace": ns,
                },
            )
        elif event == "eviction":
            count = int(attrs.get("count", 0))
            if count > 0:
                self._evictions.add(count, attributes={"namespace": ns})
        elif event == "expired":
            count = int(attrs.get("count", 0))
            if count > 0:
                self._expirations.add(count, attributes={"namespace": ns})
        # Unknown events are silently ignored.

    def update_gauges(self, stats: Stats) -> None:
        """Refresh entry-count + memory gauges from a ``cache.stats()`` snapshot."""
        ns = stats.namespace if stats.namespace is not None else "_total"
        self._gauge_state["entries"][ns] = float(stats.entries)
        self._gauge_state["memory_bytes"][ns] = float(stats.memory_bytes_estimate)


__all__ = ["OTelMetricsHook"]
