# Capacity planning & RAM management

This page is the honest answer to two questions:

1. **How big can a `mneme` cache get before something breaks?**
2. **How does RAM behave over time, and how do I keep it from drifting?**

The short version:

- The **store** scales independently of the index (SQLite handles billions of rows; Redis/Postgres/DynamoDB are effectively unbounded).
- The **in-memory index** is the real ceiling on a single host. Past ~500k entries at d=768 fp32, switch to `index_backend="hnsw"`. Past 1M, you usually also want `vector_dtype="int8"`.
- The index is a NumPy matrix (or hnswlib graph) that grows as you `put` and **does not shrink on `delete`/`vacuum` until you call `compact()`** — long-running caches with churn must compact periodically.

The rest of the page makes those statements concrete.

---

## Per-store scaling limits

| Store | Storage limit | Latency at 1M entries (p99) | Where it stops scaling |
| --- | --- | --- | --- |
| `MemoryStore` | RAM | ~µs | Process restart loses everything; bound by host RAM |
| `SQLiteStore` *(default)* | Disk; SQLite handles billions of rows | ~3 ms | Index rebuild on `open()` (seconds at 1M) |
| `RedisStore` | Redis server RAM | ~0.5–2 ms (same DC) | Redis instance memory |
| `PostgresStore` | Disk; effectively unbounded | ~2–10 ms | Network round-trip cost dominates |
| `DynamoDBStore` | AWS-managed; effectively unbounded | ~10–30 ms | Per-item cost; eventually consistency knobs |

Latency numbers are rough — your network and hardware vary, but the relative ordering holds. See [Performance](../performance.md) for measured baselines.

## The hidden ceiling: the in-memory index

Regardless of which store you pick, every embedding is also held in an in-memory matrix (or hnswlib graph) so Layer-2 cosine similarity is fast. **That is the true single-host scaling wall**:

| Records | NumpyIndex RAM (d=768, fp32) | NumpyIndex p99 search | Comment |
| --- | --- | --- | --- |
| 100k | ~300 MB | ~3 ms | Sweet spot; default config |
| 500k | ~1.5 GB | ~12 ms | Auto-switches to hnsw at this point |
| 1M | ~3 GB | ~25 ms (NumPy) / ~0.5 ms (hnsw) | hnsw becomes mandatory |
| 10M | ~30 GB | NumPy infeasible / ~1 ms (hnsw) | hnsw + tuned `M`/`ef` |

Two knobs change the math 4×:

- `vector_dtype="int8"` — cuts index RAM by 4× (1M × 768 × 1 byte ≈ 750 MB). Some accuracy loss (~1–3% similarity drift); calibrate against your corpus.
- `index_backend="hnsw"` (or `"auto"` past 500k) — makes search O(log N) at sub-ms even at 10M.

## TTL — time-based expiry

Set a max age per entry. Two ways:

```python
SemanticCache(
    path="cache.db",
    embedder=my_embedder,
    default_ttl=86400,             # every entry expires 24h after creation
)

# or per-put override:
cache.put(query, response, ttl=3600)  # this entry expires in 1h
```

**How TTL actually expires:**

1. **Lazy on read** (always on): every `cache.get()` checks the candidate's `created_at + ttl`; if stale, the entry is deleted on the spot, the `expired` metric increments, and the get returns `None`. Entries you query naturally fall off when stale.
2. **Sweep on demand**: `cache.vacuum()` scans everything and deletes any expired entries up front. Not automatic — you call it from a scheduler.

```python
# good place: a daily cron, or once on app boot
n_purged = cache.vacuum()
log.info(f"vacuumed {n_purged} expired entries")
```

There is **no background thread** doing this automatically — by design. `mneme` is a library, not a daemon. Schedule `cache.vacuum()` from your app's existing scheduler (APScheduler, Celery beat, cron, systemd timer).

!!! warning "Re-`put` resets the TTL"
    A second `put()` for the same query **replaces** the existing entry with a fresh `created_at` and the new TTL (or `default_ttl` if not specified). So calling `cache.put(q, fresh_response)` to refresh stale data extends the entry's life by `default_ttl` from now — not by remaining TTL. If you want a "refresh-doesn't-extend" pattern, pass `ttl=remaining_seconds` explicitly.

## LRU eviction — size-based fall-off

Set a max number of entries; the cache automatically drops the least-recently-accessed entries when you go over.

```python
SemanticCache(
    path="cache.db",
    embedder=my_embedder,
    max_entries=1_000_000,         # global cap
)
```

**How LRU actually evicts:**

- Runs **automatically on every `put()`**. No cron, no manual call.
- "Oldest" = lowest `last_accessed_at`. Every `get()` updates that timestamp, so frequently-queried entries stay; quiet ones fall off first.
- Eviction is batched: when you go over, it drops `max(excess, 10% of cap, 1)` entries at once, so it amortizes the cost.

**Per-namespace quotas** for multi-tenant fairness:

```python
SemanticCache(
    ...,
    namespace_quotas={
        "tenant_premium":   500_000,
        "tenant_free":       50_000,
    },
)
```

Per-namespace quotas take precedence over the global cap; one chatty namespace can't push out another's data. See [Multi-tenant](../concepts/multi-tenant.md) for the full picture.

## How RAM actually behaves — and the tombstone problem

This is the part most users miss until they hit it in production.

### What happens on `put`

The in-memory NumPy matrix has a starting capacity (256 rows). When you exceed it, the matrix is reallocated at 2× capacity and the old rows are copied. Capacity grows geometrically:

```
256 → 512 → 1024 → 2048 → 4096 → ...
```

### What happens on `delete`, TTL expiry, or LRU eviction

The corresponding row in the matrix is marked as a **tombstone** (`_tombstones: set[int]`) but its bytes are **not** reclaimed. The matrix capacity stays at whatever it was.

Concrete consequence:

```
1. put 1,000,000 entries          → matrix is 1M × 768 × 4B ≈ 3 GB
2. TTL expires 990,000 entries    → matrix is still ~3 GB; 10k rows are "live"
3. cache.stats().entries          → 10,000 (correct; queried from store)
4. cache.stats().memory_bytes_estimate → ~30 MB (live × dim × dtype) — does NOT include tombstones
5. cache.stats().index_memory_bytes    → ~3 GB (the actual matrix nbytes)
```

The two new `Stats` fields (`index_memory_bytes` and `index_tombstone_count`) make this drift visible. See [Types reference](../reference/types.md) for full field definitions. If they're out of proportion to `entries`, your in-memory index needs to be compacted.

### How to actually reclaim the memory

Call `cache.compact()`:

```python
# Reclaims tombstone memory in the in-memory index. The store is not touched —
# entries already deleted from the store remain deleted.
n_reclaimed = cache.compact()
```

Or let `vacuum()` do it for you (the default since v1.0):

```python
# vacuum() purges TTL-expired entries AND auto-compacts the index.
n_purged = cache.vacuum()                       # auto-compact=True (default)
n_purged = cache.vacuum(compact=False)          # split if you want compact on a different cadence
```

`compact()` is cheap when there are no tombstones (early-return). It does **not** touch the store; entries already deleted from the store remain deleted.

For `HnswIndex`, compact rebuilds each per-namespace index from the live set — same effect as for NumpyIndex but the implementation is a graph rebuild rather than an in-place matrix shrink.

!!! note "Compact cost at scale"
    Compact is O(N) over the live entry count: NumpyIndex allocates a fresh matrix and copies the live rows; HnswIndex re-`add_items` each live row into a fresh hnswlib index. At 1M entries that's ~tens to hundreds of milliseconds and holds the cache lock for the duration. Schedule it from a non-critical-latency context (overnight cron, low-traffic window) for big caches. The async cache routes compact through `asyncio.to_thread`, so it doesn't block the event loop's I/O — but the cache lock is held the whole time, so other concurrent gets/puts wait.

## Operational recipe — production deployment

For a 1M-entry single-host cache that doesn't drift:

```python
from mneme import SemanticCache

cache = SemanticCache(
    path="cache.db",
    embedder=my_embedder,
    index_backend="hnsw",           # mandatory at this scale
    vector_dtype="int8",            # 4× memory cut
    max_entries=1_000_000,          # LRU cap so RAM is bounded
    default_ttl=30 * 24 * 3600,     # 30-day expiry
)

# Once a day, somewhere (cron, APScheduler, your existing scheduler):
def daily_maintenance() -> None:
    n = cache.vacuum()              # purge TTL-expired + auto-compact
    log.info(f"daily vacuum: removed {n} expired")

    s = cache.stats()
    log.info(
        f"entries={s.entries} "
        f"index_memory_bytes={s.index_memory_bytes} "
        f"tombstones={s.index_tombstone_count}"
    )
```

### Multi-host (5M entries, shared cache)

```python
from mneme import SemanticCache, RedisStore

cache = SemanticCache(
    store=RedisStore(url="redis://cache.internal:6379/0"),
    embedder=my_embedder,
    index_backend="hnsw",
    vector_dtype="int8",
    max_entries=5_000_000,
    default_ttl=14 * 24 * 3600,
)
```

Each replica rebuilds its in-memory `HnswIndex` from Redis on `open()` (one-time cost; ~tens of seconds at 5M). Within a process, the same RAM-management rules apply: schedule `cache.vacuum()` to keep the index compact.

## Monitoring — what to watch

Track these from `cache.stats()` and alert when they drift:

| Metric | Healthy | Warning | What to do |
| --- | --- | --- | --- |
| `entries` | Below `max_entries` | At cap, sustained | LRU is doing its job — fine. Concern only if cap is wrong for your workload. |
| `index_tombstone_count` | < 10% of `entries` | > 25% of `entries` | Call `cache.compact()` (or just `cache.vacuum()`). |
| `index_memory_bytes` | Within 2× of `memory_bytes_estimate` | > 4× | Same fix: compact. |
| `expirations` | Steady non-zero (TTL purges) | Zero with `default_ttl` set | TTL not firing. Are you calling `vacuum()`? |
| `evictions` | Non-zero with `max_entries` set | Many evictions per second | Cap is too low for your put rate; raise `max_entries` or shorten `default_ttl`. |

The Prometheus and OpenTelemetry adapters surface all of these via [Metrics](metrics.md).

## What `mneme` deliberately doesn't do

- **No background expiration thread.** TTL purges happen lazily on read or when you call `vacuum()`. No daemon, no extra threads.
- **No automatic cap.** `max_entries=None` is the default — the cache grows until you fill the disk. **Always set `max_entries` in production.**
- **No "drop random N if too big" panic mode.** If you blow past `max_entries` between puts (e.g., bulk insert), nothing kicks in until the next `put()`. Use `cache.vacuum()` or call `Store.delete_expired()` directly to force a sweep.
- **No automatic compact threshold.** A future version may add an opt-in auto-compact when `tombstone_count / entries > 0.25`, but v1 keeps it explicit so you control when the (~O(N)) cost lands.

## Quick "what to use when" rules of thumb

- **"I'm caching LLM responses for one Flask app on one server."** → `SQLiteStore` with `max_entries` + `default_ttl` + scheduled `vacuum()`.
- **"I'm running 5 worker processes on one box."** → `SQLiteStore` + `multi_process_mode="stale-tolerant"`.
- **"I have 3 app servers behind a load balancer."** → `RedisStore`.
- **"I'm caching at 1M+ entries."** → `index_backend="hnsw"` + `vector_dtype="int8"`.
- **"I'm writing tests."** → `MemoryStore`.
- **"I'm on AWS Lambda."** → `DynamoDBStore`.
- **"I have Postgres but no Redis."** → `PostgresStore`.
