# Adapters

Drop-in `MetricsHook` implementations for popular observability stacks. Each is a no-op until installed via the corresponding optional extra.

## Prometheus

Install with `mneme[prometheus]`.

::: mneme.adapters.prometheus.PrometheusMetricsHook

## OpenTelemetry

Install with `mneme[otel]`.

::: mneme.adapters.opentelemetry.OTelMetricsHook
