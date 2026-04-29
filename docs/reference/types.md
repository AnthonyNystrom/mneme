# Types

Public dataclasses, Protocols, and type aliases. Everything here is re-exported by `mneme.__init__`.

## Dataclasses

::: mneme._types.Hit

::: mneme._types.Stats

::: mneme._types.Health

::: mneme._types.StoredEntry

## Protocols

::: mneme._types.Embedder

::: mneme._types.AsyncEmbedder

::: mneme._types.Index

::: mneme._types.Store

## Sync ↔ async embedder adapters

::: mneme._embedder_adapters.to_async_embedder

::: mneme._embedder_adapters.to_sync_embedder

## Type aliases

`mneme` exports four `Literal` aliases for configuration keys:

- **`VectorDtype`** - `"float32"` | `"float16"` | `"int8"`. The in-memory matrix dtype.
- **`HitLayer`** - `"exact"` | `"semantic"`. Which layer answered a `get`.
- **`IndexBackend`** - `"numpy"` | `"hnsw"` | `"auto"`. The index implementation.
- **`MultiProcessMode`** - `"single"` | `"stale-tolerant"` | `"mmap-shared"`. Coordination across processes.

Plus three callable aliases:

- **`ConfidenceFn`** - `Callable[[float, int, dict[str, Any]], float]` - your confidence scorer.
- **`Validator`** - `Callable[[str], bool]` - your response validator.
- **`MetricsHook`** - `Callable[[str, dict[str, Any]], None]` - your metrics fan-out.
