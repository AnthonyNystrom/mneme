# Multi-process

`mneme` ships three modes for running the cache across processes. Pick with `multi_process_mode=` when you instantiate the cache:

```python
SemanticCache(path="cache.db", embedder=..., multi_process_mode="single")          # default
SemanticCache(path="cache.db", embedder=..., multi_process_mode="stale-tolerant",
              stale_check_interval=0.5)
SemanticCache(path="cache.db", embedder=..., multi_process_mode="mmap-shared")
```

## At a glance

| Mode | Coordination cost | Read consistency | Best for |
| --- | --- | --- | --- |
| `single` (default) | none | own writes only | One process owning the cache file |
| `stale-tolerant` | periodic poll of `version_counter` + `iter_since` deltas | eventually consistent (configurable lag) | Multiple processes on one host, mostly-readonly workloads |
| `mmap-shared` | `fcntl.flock` (POSIX) / `msvcrt.locking` (Windows) on a shared mmap matrix | strong on the same host | Read-heavy, latency-sensitive, willing to operate on a single host |

Cross-host coordination is a different problem — use a shared store backend (Redis, Postgres, DynamoDB) instead of a multi-process mode.

## `single` (default)

Assumes one process owns the cache file. Writes go to the store; reads serve from the in-memory index. No coordination logic at all. Fastest path; correct only when you actually have one writer.

If two processes both run `single` against the same SQLite file, SQLite's WAL mode prevents corruption — but the in-memory indices drift. One process's `put()` is invisible to the other until it reopens the cache.

## `stale-tolerant`

The cache periodically polls `store.version_counter`. When the counter has advanced, it asks the store for entries inserted since the last seen `last_id` (`store.iter_since`) and applies the delta to its in-memory index.

```python
cache = SemanticCache(
    path="cache.db",
    embedder=embedder,
    multi_process_mode="stale-tolerant",
    stale_check_interval=0.5,    # poll every 500 ms
)
```

Trade-offs:

- **Lag is bounded by `stale_check_interval`.** Smaller is fresher, larger is cheaper. 0.1–1 s is a typical range.
- **Writes are immediately visible to the writer**; readers see them after the next poll.
- **Tombstones from `delete()` propagate the same way** — readers don't see deleted rows on the very next `get`, but will after the next stale check.
- **Above a threshold of pending changes, the cache full-rebuilds** instead of applying deltas. The threshold is tuned to keep stale checks cheap; the rebuild is amortized.

This mode is the workhorse for **multiple processes sharing a SQLite file on one host** — typical for Gunicorn/Uvicorn workers, multi-process job runners, or simple production setups.

## `mmap-shared`

Advanced. The in-memory index is replaced with a single mmap-backed matrix shared across processes. Writes acquire an exclusive file lock; reads acquire a shared lock.

```python
SemanticCache(path="cache.db", embedder=..., multi_process_mode="mmap-shared")
```

Pros:

- Strong read consistency on the same host. After a write commits, all readers see the new entry on their next `get`.
- No periodic polling overhead.
- One physical copy of the matrix in RAM regardless of process count.

Cons:

- POSIX (`fcntl.flock`) is the first-class implementation. Windows (`msvcrt.locking`) is best-effort and not exhaustively tested.
- Compaction at 25% tombstone density rewrites the file via atomic-rename; readers transparently re-mmap.
- Careful integration: `MmapSharedCoordinator` is exposed as a separately-instantiable primitive, used directly by advanced users (it's how the showcase's stress tests verify the coordination semantics).
- Single-host only. For cross-host shared cache, use a network store backend.

## Cross-host: use a network store

For genuinely distributed deployments, the multi-process modes don't help — they coordinate processes on one machine. Across hosts, use a network-backed `Store`:

| Backend | When |
| --- | --- |
| [`RedisStore`](../stores/redis.md) | Lowest latency, ephemeral OK, you already run Redis |
| [`PostgresStore`](../stores/postgres.md) | Durable, transactional, you already run Postgres |
| [`DynamoDBStore`](../stores/dynamodb.md) | Serverless, multi-region, AWS shop |

Each one bumps `version_counter` in the same transaction as every data write, so even cross-host the `stale-tolerant` polling pattern works. Set `multi_process_mode="stale-tolerant"` plus a network store and your processes converge on the shared state with bounded staleness.

## What `mneme` doesn't do

- **No background threads.** The cache never spawns its own thread. The `stale-tolerant` polling happens inline on each `get` if the interval has elapsed; it's a piggyback, not a worker.
- **No `multiprocessing.shared_memory`.** That API has known cleanup issues across Python versions; `mmap-shared` uses plain `mmap` + `fcntl.flock` instead.
- **No `asyncio.Lock` over the cache `RLock`.** The async cache reuses the sync core's `RLock`; there's no second async lock layered on top.

## Concurrency invariants

Whichever mode you pick, the cache's locking semantics are the same:

- One `RLock` per `SemanticCache` instance, held for the full duration of every public method.
- Writes (`put`, `delete`, `clear_namespace`, `clear`) bump `version_counter` in the same transaction as the data write at the store level.
- The lock is released across embedder awaits in `AsyncSemanticCache` so concurrent `get` calls overlap their I/O.

## Where to go next

- **[Stores: SQLite](../stores/sqlite.md)** — the default `single` / `stale-tolerant` backing.
- **[Stores: Redis / Postgres / DynamoDB](../stores/redis.md)** — cross-host options.
- **[Performance tuning](../guides/performance-tuning.md)** — picking `stale_check_interval` for your workload.
