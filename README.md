# mneme

A layered semantic cache for LLM applications.

`mneme` (Greek: μνήμη, "memory"; pronounced *NEE-mee*) is an embeddable, in-process Python library that caches LLM completions across paraphrased queries. It pairs an exact-match layer (normalized query hash) with a semantic-match layer (cosine similarity over L2-normalized embeddings) and persists durably to a single SQLite file by default.

```python
from mneme import SemanticCache
from my_embedders import OpenAIEmbedder

embedder = OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536)

with SemanticCache(path="cache.db", embedder=embedder) as cache:
    hit = cache.get("How do I reset my password?")
    if hit is None:
        response = call_my_llm("How do I reset my password?")
        cache.put("How do I reset my password?", response)
    else:
        response = hit.response
```

## Why mneme

- **Cache before you call.** A semantic cache turns redundant LLM calls into a microsecond `dict` lookup or a millisecond NumPy matvec. For chatbots, agent loops, and batch-style scoring jobs, this is the difference between a viable product and one that burns tokens on every paraphrase.
- **One required dependency.** NumPy. Optional extras for hnswlib, redis, psycopg, boto3, prometheus, and opentelemetry. Bring your own embedder, your own LLM client, your own server.
- **In-process, no daemon.** `mneme` is a library you `import`, not a service you operate. Persists to a single SQLite file by default; swap in Redis / Postgres / DynamoDB when you need shared state across hosts.
- **Strict typing, zero magic.** Public surface is a small set of frozen `@dataclass`es and `Protocol`s in [src/mneme/__init__.py](src/mneme/__init__.py). `py.typed` shipped.

## Features

- **Layered cache.** O(1) exact match (normalized query hash → SHA-256), then cosine similarity over an in-memory matrix.
- **Sync and async** APIs with parity (`SemanticCache` and `AsyncSemanticCache`).
- **Five `Store` backends:** Memory, SQLite (default), Redis, Postgres, DynamoDB. `Store` Protocol for custom backends.
- **Two `Index` backends:** NumPy (default, fast to ~500k entries) and hnswlib (opt-in, scales past 1M).
- **Three vector dtypes:** `float32` (default), `float16`, `int8` for memory-constrained deployments.
- **Three multi-process modes:** `single`, `stale-tolerant`, `mmap-shared`.
- **Multi-tenant** via namespaces with per-namespace LRU quotas.
- **Calibration tooling** (Python API + `python -m mneme.tools.calibrate` CLI) for tuning similarity thresholds against your own paraphrase / distractor corpora.
- **Checkpoint export/import** (`tar.gz` round-trip) for backup and environment promotion.
- **Re-embed migration tool** when the embedder changes dimension or model.
- **Prometheus and OpenTelemetry** metrics adapters shipped (lazy-imported).

## Install

```bash
pip install mneme                       # core (NumPy only)
pip install "mneme[hnsw]"               # add hnswlib for >500k entries
pip install "mneme[redis]"              # RedisStore
pip install "mneme[postgres]"           # PostgresStore
pip install "mneme[dynamodb]"           # DynamoDBStore
pip install "mneme[prometheus,otel]"    # metrics adapters
pip install "mneme[all]"                # everything
```

Python 3.10 or newer.

## Quickstart (sync)

See [examples/quickstart.py](examples/quickstart.py).

```python
from mneme import SemanticCache, MemoryStore

embedder = MyEmbedder()  # any callable returning float32 numpy arrays

with SemanticCache(store=MemoryStore(), embedder=embedder) as cache:
    cache.put("How do I reset my password?", "Click 'Forgot password' on login.")
    hit = cache.get("Where do I reset my password?")  # paraphrase
    assert hit is not None
    print(hit.layer, hit.similarity, hit.response)
```

## Quickstart (async)

See [examples/async_quickstart.py](examples/async_quickstart.py).

```python
import asyncio
from mneme import AsyncSemanticCache

async def main():
    async with AsyncSemanticCache(path="cache.db", embedder=async_embedder) as cache:
        hit = await cache.get("how can I cancel my subscription")
        if hit is None:
            response = await call_llm_async(...)
            await cache.put("how can I cancel my subscription", response)

asyncio.run(main())
```

`AsyncSemanticCache` wraps the same core, awaits the embedder directly, and runs blocking store work via `asyncio.to_thread`. Sync and async caches are interchangeable from a behavior standpoint; they share the public Protocol surface.

## Public API at a glance

Re-exported by [`mneme.__init__`](src/mneme/__init__.py):

| Symbol | Purpose |
| --- | --- |
| `SemanticCache`, `AsyncSemanticCache` | The cache classes you instantiate. |
| `Hit`, `Stats`, `Health`, `StoredEntry` | Frozen dataclasses returned by query / introspection methods. |
| `Embedder`, `AsyncEmbedder` | Protocols you implement (or adapt with `to_async_embedder` / `to_sync_embedder`). |
| `Index`, `Store` | Protocols if you want to bring your own backend. |
| `MemoryStore`, `SQLiteStore` | Two of the five shipped Store impls (always-imported). |
| `VectorDtype`, `IndexBackend`, `MultiProcessMode`, `HitLayer` | Literal type aliases. |
| `MnemeError` and the full exception hierarchy | All listed in [src/mneme/_exceptions.py](src/mneme/_exceptions.py). |

Optional-dep stores import directly from their module so `import mneme` stays lightweight:

```python
from mneme._store_redis import RedisStore
from mneme._store_postgres import PostgresStore
from mneme._store_dynamodb import DynamoDBStore
```

## Stores

All five backends satisfy the `Store` Protocol and are exercised by the same conformance battery in [tests/test_store_protocol_compliance.py](tests/test_store_protocol_compliance.py).

| Backend | When to use | Notes |
| --- | --- | --- |
| `MemoryStore` | Tests, scratch, ephemeral caches | Lost on process exit. |
| `SQLiteStore` (default) | Single-process or stale-tolerant multi-process on one host | One file, WAL mode, atomic writes. Snapshot via SQLite's `.backup` API. |
| `RedisStore` | Shared cache across hosts, high write rate | Atomic `MULTI`/`EXEC`. Requires `[redis]` extra. |
| `PostgresStore` | Shared cache when you already run Postgres | `version_counter` bumped in same txn as the data write. Requires `[postgres]` extra. |
| `DynamoDBStore` | Serverless / multi-region | Atomic `TransactWriteItems`; auto-create-on-open opt-in. Requires `[dynamodb]` extra. See [examples/dynamodb_quickstart.py](examples/dynamodb_quickstart.py). |

### Custom stores

Implement the `Store` Protocol from [src/mneme/_types.py](src/mneme/_types.py). The conformance battery is the contract — your impl is "done" when it passes that battery. See [examples/custom_store.py](examples/custom_store.py).

## Index backends

```python
SemanticCache(..., index_backend="auto")    # numpy below 500k, hnsw above (default)
SemanticCache(..., index_backend="numpy")
SemanticCache(..., index_backend="hnsw", index_options={"M": 16, "ef": 64})
```

NumPy is the default and is fast enough for most workloads up to ~500k entries. hnswlib is opt-in via the `[hnsw]` extra and shines past 1M entries; if you ask for `"hnsw"` without the extra installed, `mneme` issues a WARNING and falls back to NumPy with a clear remediation.

## Quantization

Vectors are persisted as `float32` (source of truth). The in-memory matrix can be `float32`, `float16`, or `int8` per `vector_dtype=`:

| dtype | Memory at 100k × d=1536 | Latency on pure NumPy | Use when |
| --- | --- | --- | --- |
| `float32` (default) | 614 MB | Fastest | RAM is plentiful |
| `float16` | 307 MB | Slightly slower (cast each search) | Memory-constrained, latency-sensitive |
| `int8` | 154 MB | Slower at high d (no fused int8 GEMM in NumPy) | Memory is *very* constrained; pair with hnsw if you also need low latency |

See [examples/high_dim_quantized.py](examples/high_dim_quantized.py). Always **calibrate** the similarity threshold against the same `vector_dtype` you'll run in production — int8 can shift cosine scores 1–3% on typical embeddings.

## Multi-process modes

```python
SemanticCache(path="cache.db", embedder=..., multi_process_mode="single")          # default
SemanticCache(path="cache.db", embedder=..., multi_process_mode="stale-tolerant",
              stale_check_interval=0.5)
SemanticCache(path="cache.db", embedder=..., multi_process_mode="mmap-shared")     # advanced
```

| Mode | Coordination | Trade-off |
| --- | --- | --- |
| `single` | None | Fastest. Fine when one process owns the cache file. |
| `stale-tolerant` | Periodic poll of the store's `version_counter`; `iter_since` deltas, full rebuild over a threshold | Eventually-consistent reads across processes. Most users want this. |
| `mmap-shared` | One mmap matrix shared across processes under `fcntl.flock` (POSIX) / `msvcrt.locking` (Windows) | Strongest consistency for read-heavy workloads on the same host. Compaction at 25% tombstone density. POSIX is first-class; Windows is best-effort. |

`mneme` never spawns background threads, never uses `multiprocessing.shared_memory`, and never holds an `asyncio.Lock` over the cache `RLock`.

## Multi-tenant (namespaces & quotas)

```python
cache = SemanticCache(
    path="cache.db",
    embedder=embedder,
    namespace_quotas={"tenant_a": 1000, "tenant_b": 5000},  # per-namespace LRU
)
cache.put("...", "...", namespace="tenant_a")
cache.put("...", "...", namespace="tenant_b")
hit = cache.get("...", namespace="tenant_a")
```

Eviction is namespace-quota-first then global LRU; batches are 10% of cap (min 1) per [src/mneme/_eviction.py](src/mneme/_eviction.py). See [examples/multi_tenant.py](examples/multi_tenant.py).

## Calibration

Tune your similarity threshold against your own paraphrases and distractors:

```python
from mneme.tools.calibrate import find_threshold

result = find_threshold(
    paraphrase_pairs=[("How do I cancel?", "Cancel my subscription"), ...],
    distractor_pairs=[("How do I cancel?", "What is the weather?"), ...],
    embedder=my_embedder,
    target_metric="f1",            # or "precision" / "recall"
    min_precision=0.95,            # optional constraint
    vector_dtype="int8",           # calibrate against the same dtype prod uses
)
print(result.threshold, result.precision, result.recall, result.f1)
```

Or via CLI (`python -m mneme.tools.calibrate --help`).

See [examples/calibration.py](examples/calibration.py).

## Checkpoints

`dumps()` and `loads()` round-trip the full cache state (entries + counters + namespaces + dtype + manifest) as a single `tar.gz`:

```python
cache.dumps("backup.tar.gz")

# Later, possibly on a different host:
restored = SemanticCache.loads("backup.tar.gz", embedder=embedder)
```

Fingerprint and dimension are validated on `loads`; mismatches raise `EmbedderMismatchError` / `EmbedderDimensionError`. SQLiteStore is the only backend that implements `snapshot_to` / `restore_from` directly; for Redis / Postgres / DynamoDB use the platform-native backup tool (`BGSAVE`, `pg_dump`, on-demand backup / PITR).

## Re-embed migration

Switching embedder model or dimension? `mneme.tools.migrate.reembed` walks every entry through the new embedder, writes to a new cache, and leaves the source untouched:

```python
from mneme.tools.migrate import reembed

reembed(source=old_cache, dest=new_cache, batch_size=128)
```

`areembed` is the async equivalent. The fingerprint changes; the entries do not.

## Metrics

Bring-your-own metrics hook:

```python
def my_hook(event: str, fields: dict[str, Any]) -> None:
    print(event, fields)

cache = SemanticCache(..., metrics_hook=my_hook)
```

Or use the shipped adapters:

```python
from mneme.adapters.prometheus import PrometheusMetricsHook
from mneme.adapters.opentelemetry import OTelMetricsHook

cache = SemanticCache(..., metrics_hook=PrometheusMetricsHook())
```

Hook failures are caught and downgraded to `WARNING` — never crash the cache for an observability problem.

## Reference embedders

`mneme` does **not** bundle an embedder; you provide one that returns a 1-D `float32` `numpy.ndarray` of length `dim` with a stable `fingerprint` string. See [examples/reference_embedders/](examples/reference_embedders/) for documentation-only snippets covering:

- OpenAI (`text-embedding-3-small` / `-large`)
- sentence-transformers (any local model)
- AWS Bedrock (Titan / Cohere via `boto3`)
- Ollama (local self-hosted)

These files are not imported by the package — they're starting points you copy into your own code.

## Performance baseline

Measured on an Apple M4 Max, macOS 26.3, Python 3.12.13, NumPy 2.4.4 with Apple Accelerate BLAS. Run `pytest tests/test_perf.py --run-perf -s` to record your own. `MNEME_PERF_HEAVY=1` enables the 1M-entry hnsw benchmarks.

| Workload | Aspirational target | Observed (p99) |
| --- | --- | --- |
| Exact-match `get` @ 100k | < 500 µs | ~2.3 ms |
| Semantic `get` fp32 @ 100k/d=768 | < 5 ms | ~2.7 ms |
| Semantic `get` fp32 @ 100k/d=1536 | < 8 ms | ~4.0 ms |
| Semantic `get` int8 @ 100k/d=1536 | < 6 ms | ~50–60 ms |
| `put` @ 100k (no eviction) | < 2 ms | ~0.9 ms |
| `put` with 10% eviction (cap 10k) | < 20 ms | ~40–45 ms |
| Open + rebuild fp32 @ 100k/d=768 | < 100 ms | ~300 ms |
| Open + rebuild int8 @ 100k/d=768 | < 200 ms | ~400–450 ms |
| Single-thread throughput | > 5000 ops/sec | ~5700 ops/sec |
| Async throughput (100 concurrent) | > 2000 ops/sec | ~5100 ops/sec |
| Direct `NumpyIndex.search` p99 @ 100k/d=768 | < 5 ms | ~2.8 ms |

Tests use **regression bars** above the observed baseline (1.5–2× headroom) so the suite stays green on a typical contributor laptop while still flagging gross regressions.

### Notes on the gaps

- **`exact_get`** is dominated by the per-`get` SQLite `UPDATE` of `last_used_unix`. Removing the UPDATE drops it under 100 µs but degrades cross-process LRU accuracy.
- **`int8` semantic search at d=1536** is bandwidth-bound: pure NumPy has no fused int8 GEMM, so the int8 → fp32 cast (~750 MB of memory traffic per search) is the floor. The win for int8 is **memory footprint** (4× smaller in-memory matrix), not latency. Use the hnsw backend if you need both.
- **`put` with eviction** does 1000 small WAL writes per batch; future work could batch the deletes into one `DELETE … WHERE id IN (…)`.
- **Open time** is dominated by physical IO + Python row iteration. The fast path `SQLiteStore.iter_index_rows()` already skips JSON metadata parsing during rebuild, halving open time. Pushing below the target needs a binary blob format or memory-mapped store.

## Comparison

| | mneme | GPTCache |
| --- | --- | --- |
| Required runtime deps | NumPy | many (sqlite, faiss, ...) |
| In-process / embeddable | yes | yes |
| Bundled embedder | **no** (BYOE) | yes (langchain, etc.) |
| Bundled LLM client | no | yes |
| Sync + async parity | yes | partial |
| Strict typing (`py.typed`) | yes | no |
| Multi-process modes | 3 (single / stale-tolerant / mmap) | n/a |
| Multi-tenant quotas | per-namespace LRU | n/a |
| Calibration tooling | yes (CLI + Python API) | no |

## Status

Pre-release. The §8 public surface in [src/mneme/__init__.py](src/mneme/__init__.py) is locked for v1.0.

## License

Apache 2.0. See [LICENSE](LICENSE).
