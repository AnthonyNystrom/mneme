# Metrics

`mneme` doesn't ship with a hard-coded metrics backend. It fires events through a single `MetricsHook` callable; you wire those to whatever you operate. Two adapters are bundled - Prometheus and OpenTelemetry - but a custom hook is just a function.

## The hook signature

```python
MetricsHook = Callable[[str, dict[str, Any]], None]
```

Two arguments:

- **`event: str`** - one of: `hit_exact`, `hit_semantic`, `miss`, `put`, `put_rejected`, `evict`, `expire`, `embedder_failure`.
- **`fields: dict`** - event-specific payload. Always includes `namespace`. Hits also include `similarity` (Layer 2 only) and `confidence`.

The hook fires inline on the operation's thread. Treat it as cheap: any blocking work in the hook serializes against the cache lock.

## Custom hook

```python
from mneme import SemanticCache

def my_hook(event: str, fields: dict) -> None:
    print(f"{event} {fields}")

cache = SemanticCache(..., metrics_hook=my_hook)
```

That's the whole interface. Send events to whatever you want - logs, statsd, internal counters, a queue for async fan-out.

## Failure handling

Hook exceptions are **caught and downgraded to WARNING**. The cache never crashes for an observability problem. If your hook raises, the corresponding `get` / `put` still succeeds - only the metric is dropped.

This is intentional: a flaky StatsD client should never break user-facing requests.

## Prometheus

`PrometheusMetricsHook` registers the standard `mneme_*` counters and histograms with the default Prometheus registry:

```python
from mneme.adapters.prometheus import PrometheusMetricsHook

cache = SemanticCache(..., metrics_hook=PrometheusMetricsHook())
```

Metrics exposed:

| Metric | Type | Labels |
| --- | --- | --- |
| `mneme_hits_total` | Counter | `namespace`, `layer` (exact / semantic) |
| `mneme_misses_total` | Counter | `namespace`, `reason` (no_match / below_threshold / embedder_failure) |
| `mneme_puts_total` | Counter | `namespace`, `result` (ok / rejected) |
| `mneme_evictions_total` | Counter | `namespace`, `reason` |
| `mneme_expirations_total` | Counter | `namespace` |
| `mneme_similarity_score` | Histogram | `namespace` |
| `mneme_get_latency_seconds` | Histogram | `namespace`, `layer` |
| `mneme_cache_entries` | Gauge | `namespace` |

Install with `mneme[prometheus]`. Wire to your scrape endpoint via the standard `prometheus_client.start_http_server(...)` or a Flask/FastAPI exposition endpoint.

## OpenTelemetry

`OTelMetricsHook` emits the same metrics through the OTel meter:

```python
from mneme.adapters.opentelemetry import OTelMetricsHook
from opentelemetry import metrics

provider = metrics.get_meter_provider()
cache = SemanticCache(..., metrics_hook=OTelMetricsHook(meter=provider.get_meter("mneme")))
```

Names match Prometheus where convention permits; labels become OTel attributes. Install with `mneme[otel]`. Configure your OTel exporter (OTLP, Console, etc.) the usual way.

## Common patterns

### Stack hooks

`metrics_hook=` takes one callable. To run two (e.g. logs + Prometheus), compose:

```python
def composite(event: str, fields: dict) -> None:
    log_hook(event, fields)
    prom_hook(event, fields)

cache = SemanticCache(..., metrics_hook=composite)
```

### Add static labels

If you want all events tagged with a `service=` or `region=` label:

```python
def tagged_hook(event: str, fields: dict) -> None:
    fields = {**fields, "service": "intent-classifier", "region": "us-east-1"}
    inner_hook(event, fields)
```

### Async-safe fan-out

For very high event rates, push to an `asyncio.Queue` and drain in a background task:

```python
queue = asyncio.Queue(maxsize=10_000)

def queue_hook(event: str, fields: dict) -> None:
    try:
        queue.put_nowait((event, fields))
    except asyncio.QueueFull:
        pass        # drop on overflow rather than block the cache

async def drain():
    while True:
        event, fields = await queue.get()
        await fan_out(event, fields)
```

The hook itself stays cheap (one queue put); heavy fan-out runs off the request path.

## Counters from `stats()`

`cache.stats()` returns a snapshot of the same counters the hooks fire on:

```python
s = cache.stats()
print(f"hit_rate: {(s.hits_exact + s.hits_semantic) / (s.hits_exact + s.hits_semantic + s.misses):.2%}")
```

This is a polling alternative to hooks - useful for occasional inspection but not for sustained observability.

## What to watch in production

| Signal | Why |
| --- | --- |
| **Hit rate** by namespace | Drops correlate with embedder regressions or threshold drift |
| **`embedder_failure` rate** | Network or quota issues with the embedder; cache silently falling through to LLM |
| **`similarity_score` p50/p99** | Distribution shifts mean your corpus is changing |
| **`get_latency_seconds`** | p99 spikes mean the index is too big for the dtype/backend combo |
| **Cache entry count** | Growing without bound? Eviction misconfigured |

The [showcase dashboard](../showcase.md) exposes most of these live as a reference UI.

## Where to go next

- **[API reference: adapters](../reference/adapters.md)** - `PrometheusMetricsHook` and `OTelMetricsHook` signatures.
- **[Confidence and validators](../concepts/confidence-and-validators.md)** - `put_rejected` events come from the validator.
- **[Performance tuning](performance-tuning.md)** - interpret hit-rate / latency dashboards.
