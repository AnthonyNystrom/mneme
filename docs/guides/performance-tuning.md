# Performance tuning

A practical guide to making `mneme` faster — or knowing when you've hit the ceiling for a given backend / dtype combo. The [Performance baseline](../performance.md) page has the measured numbers; this page is the *what to do about them*.

## The latency budget

For a typical `get`:

```
total = layer1_lookup + (if Layer 1 misses)
        embed + matvec + score
```

For a typical `put`:

```
total = embed + store_insert + index_append (+ eviction if cap is breached)
```

Each step has its own bottleneck. Tune the ones that dominate *your* workload.

## Layer 1 (exact match)

Dominated by the store's read path — usually a primary-key lookup.

| Backend | p99 cost @ 100k entries | Bottleneck |
| --- | --- | --- |
| MemoryStore | <50 µs | Python dict |
| SQLiteStore | ~2 ms | The `UPDATE last_accessed_at` per get (LRU bookkeeping) |
| RedisStore | ~0.5–1 ms | Network RTT + HGETALL + ZADD |
| PostgresStore | ~1–3 ms | Network RTT + UPDATE last_accessed_at |
| DynamoDBStore | ~2–5 ms | Network RTT + UpdateItem |

**Tunes**:

- **Drop `update_access` if you don't need cross-process LRU.** Subclass the store and make it a no-op; `get` drops to <100 µs on SQLite. You lose accurate LRU eviction in stale-tolerant mode.
- **Use a connection pool for network stores.** Per-call connection setup is the bulk of the latency on cold paths.
- **Run the cache process near the store.** Same AZ for Redis/Postgres/DynamoDB; same host (Unix socket) for Postgres/Redis when possible.

## Layer 2 (semantic match)

Dominated by the matvec across the in-memory index.

```
matvec_cost = n × d × bytes_per_element  # memory bandwidth bound
```

For 100k × 1536:

- `float32`: 614 MB read → ~6 ms at 100 GB/s memory bandwidth
- `float16`: 307 MB read + cast → ~10 ms (cast adds work)
- `int8`: 154 MB read + 614 MB cast write → ~50 ms (the cast dominates, not the matmul)

**Tunes**:

- **Pick the right `vector_dtype`.** float32 is fastest at matvec time; int8 is smallest in memory but slowest on pure NumPy. See [Quantization](../concepts/quantization.md).
- **Switch to hnsw past 500k entries.** `index_backend="hnsw"` does approximate-NN in O(log n) instead of full scan. Latency stays under 1 ms at 1M+ entries.

  ```python
  SemanticCache(..., index_backend="hnsw", index_options={
      "M": 16,                 # graph degree; 16 is a good default
      "ef_construction": 200,  # build-time accuracy
      "ef": 64,                # query-time accuracy (higher = slower + more accurate)
  })
  ```

  The hnsw backend builds incrementally on each `put`; expect put latency to rise slightly. Recall is ~99% at default settings.

- **Use `index_backend="auto"`** to let the cache pick: NumPy below 500k, hnsw above. With a fallback to NumPy + WARNING if hnswlib isn't installed.

## Embedder

For each Layer-2 `get` and every `put`:

| Embedder | Latency per call |
| --- | --- |
| sentence-transformers MiniLM (CPU) | 3–10 ms |
| sentence-transformers (GPU) | <1 ms |
| OpenAI `text-embedding-3-small` | 50–150 ms (network) |
| Local Ollama `nomic-embed-text` | 5–20 ms (localhost) |
| AWS Bedrock Titan | 50–150 ms (cross-AZ network) |

**Tunes**:

- **Pre-embed in batches when you know the queries.** Pass `embedding=v` to `get`/`put` to skip the embedder. Useful when you have a corpus and want to warm the cache offline.
- **Run the embedder co-located.** Local model > network call. If you must use a hosted embedder, pin to the same region as your app.
- **Skip the embedder on Layer-1 hits** — the cache already does this. Make sure your monitoring distinguishes Layer 1 from Layer 2 latency so you don't optimize the wrong path.

## Open time

Cold-start cost for the cache: read every entry from the store, push them into the in-memory index.

| Backend | Open time @ 100k/d=768 |
| --- | --- |
| MemoryStore | n/a (state lives in process) |
| SQLiteStore | ~300 ms |
| RedisStore | ~500 ms (network) |
| PostgresStore | ~400 ms |
| DynamoDBStore | ~3–10 s (Scan is slow) |

**Tunes**:

- **Don't reopen often.** The cache is meant to live for the lifetime of the process.
- **For SQLite**: the `iter_index_rows()` fast path skips JSON metadata parsing during rebuild — already enabled. Pushing under 100 ms needs a binary blob format or memory-mapped store; not in v1.
- **For DynamoDB at scale**: open time can be unbounded. Consider a per-tenant cache file with a smaller working set.

## Eviction

Triggered when a `put` would breach `max_entries` or a `namespace_quotas` cap. Batches 10% of the cap (min 1) at a time.

For cap=10k, a triggering put deletes 1000 entries. On SQLite at p99 ~40 ms.

**Tunes**:

- **Raise `max_entries` if you have RAM + disk for it.** Eviction overhead drops; hit rate rises.
- **Set per-namespace `namespace_quotas`** instead of one global cap. Per-tenant eviction keeps small tenants from being squeezed by big ones.
- **Run `cache.vacuum()` periodically** to clear TTL'd entries before they hit eviction. Cheaper than reactive eviction at scale.

## Stale-tolerant polling

When `multi_process_mode="stale-tolerant"`, the cache polls `version_counter` every `stale_check_interval` seconds.

**Tunes**:

- **Smaller interval = fresher reads, more counter checks.** 0.1 s for tight consistency, 1 s for relaxed.
- **Bigger interval = more pending changes per check.** Above a threshold, the cache full-rebuilds instead of applying deltas. Tune the interval to keep deltas small relative to your insert rate.
- **For mostly-readonly fleets**: 5–10 s is fine. The cache lags reality but recovers eventually.

## hnsw tuning

If you've moved to `index_backend="hnsw"`:

- **`M`** (graph degree): higher = better recall, more memory. 16 default; 32 for large indices.
- **`ef_construction`**: build-time accuracy. 200 default; 400 for higher-recall indices.
- **`ef`** (query-time): exposed via `index_options={"ef": 64}` and adjustable per-query through the index. Higher = more accurate, slower. Sweep this against your calibration corpus.

## Profiling

For workload-specific tuning:

```python
import cProfile, pstats

with cProfile.Profile() as pr:
    for _ in range(1000):
        cache.get(query)

pstats.Stats(pr).sort_stats("cumulative").print_stats(20)
```

Look for `embed`, `matvec`, store calls in the top frames. The hot path tells you where to optimize.

## Where to go next

- **[Performance baseline](../performance.md)** — measured numbers across stores and dtypes.
- **[Quantization](../concepts/quantization.md)** — picking the right dtype.
- **[Multi-process](../concepts/multi-process.md)** — when stale-tolerant polling cost matters.
