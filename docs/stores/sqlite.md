# SQLiteStore

The default backing for `mneme`. One file, WAL mode, atomic writes, no daemon. Always available.

```python
from mneme import SemanticCache

# Convenience: pass a path, the cache constructs SQLiteStore for you.
with SemanticCache(path="cache.db", embedder=embedder) as cache:
    cache.put("hello", "world")

# Equivalent explicit form:
from mneme import SQLiteStore
with SemanticCache(store=SQLiteStore("cache.db"), embedder=embedder) as cache:
    ...
```

## When to pick it

- **Single-host applications.** A web app, a CLI tool, a desktop app — anything that runs as one process or a small worker pool on one machine.
- **Multiple processes on one host with `multi_process_mode="stale-tolerant"`.** SQLite's WAL mode lets readers and writers coexist; the stale-tolerant coordinator handles eventual consistency.
- **Durable cache that needs to survive restarts.** The cache picks up where it left off after a crash or upgrade.

For shared state across hosts, pick [Redis](redis.md), [Postgres](postgres.md), or [DynamoDB](dynamodb.md).

## File semantics

- **Mode 0o600 on creation.** Owner-only readable. The cache may contain LLM responses you don't want a co-tenant on a shared box to grep.
- **WAL (Write-Ahead Log) mode.** Reads don't block writes; one writer at a time, but readers see a consistent snapshot.
- **Three sidecar files.** `cache.db-wal` (the WAL) and `cache.db-shm` (shared memory index). Both are checkpointed back into `cache.db` on `.close()` or via SQLite's auto-checkpoint. **Always include the sidecars when you copy the file out-of-band**; copying just `cache.db` while the app is running leaves uncheckpointed entries on the floor.
- **Foreign keys disabled.** None of `mneme`'s schema needs them, and disabling avoids a class of subtle migration headaches.

## Schema

`mneme` ships a single migration that creates these tables:

| Table | Purpose |
| --- | --- |
| `entries` | One row per cached entry — id, namespace, query_hash, query, response, embedding (BLOB float32), metadata (JSON), created_at, last_accessed_at, ttl, access_count |
| `cache_counters` | Per-namespace counters (hits_exact, hits_semantic, misses, etc.) — mirrored from the in-memory metrics on commit |
| `namespace_quotas` | Per-namespace `max_entries` |
| `multi_process_state` | `version_counter` for stale-tolerant polling |
| `schema_meta` | `embedder_fingerprint`, `embedder_dim`, `schema_version` |

Indexes on `(namespace, query_hash)` (uniqueness) and `(namespace, last_accessed_at)` (LRU iteration). All SQL is parameterized; no f-string SQL anywhere.

## Atomic writes

Every mutation runs in a single transaction that includes a `version_counter` increment:

```sql
BEGIN IMMEDIATE;
INSERT INTO entries (...) VALUES (?, ?, ?, ...);
UPDATE multi_process_state SET value = (CAST(value AS INTEGER) + 1) WHERE key = 'version_counter';
COMMIT;
```

That single-transaction guarantee is what makes the [stale-tolerant multi-process mode](../concepts/multi-process.md#stale-tolerant) safe — once another process sees the counter advance, the data is durable.

## Snapshot / restore

`SQLiteStore` is the **only** shipped store with a real `snapshot_to` / `restore_from` implementation. It uses SQLite's online backup API (`.backup()`):

```python
store.snapshot_to("backup.db")    # streams a consistent snapshot, even while writes continue

# Restore is just opening a new SQLiteStore against the snapshot file:
restored = SQLiteStore.restore_from("backup.db", "live.db")
restored.open(embedder.fingerprint, embedder.dim)
```

The cache-level `dumps()` / `loads()` use this internally and also bundle the manifest, schema version, and (optionally) the in-memory matrix dtype.

## Tuning

The defaults are tuned for the typical 100k–500k entry case. For larger caches:

- **`PRAGMA journal_mode=WAL`** is set on every open; you don't change this.
- **`PRAGMA synchronous=NORMAL`** is set on every open. Trade-off: slight write speed gain over `FULL`, very small durability window after kernel crash.
- **`PRAGMA cache_size`** is left at SQLite's default (~2 MB). For caches over 1M entries, raise it to 64–256 MB via your own connection if you need faster cold-start scans. The cache's `iter_index_rows()` fast path bypasses most of the cost regardless.

## Common operations

```python
# Stats
cache.stats(namespace="support")
# -> Stats(entries=12345, hits_exact=900, hits_semantic=400, misses=200, ...)

# Wipe one tenant
cache.clear_namespace("tenant_a")

# Wipe everything
cache.clear()

# Vacuum (TTL'd entries only)
cache.vacuum()

# Snapshot
cache.dumps("backup.tar.gz")
```

## Performance baseline

On Apple M4 Max with NVMe SSD:

| Operation | Latency |
| --- | --- |
| Exact-match `get` (warm) @ 100k | ~2.3 ms p99 |
| Semantic `get` fp32 @ 100k/d=768 | ~2.7 ms p99 |
| `put` (no eviction) @ 100k | ~0.9 ms p99 |
| `put` with 10% eviction (cap=10k) | ~40–45 ms p99 |
| Open + rebuild fp32 @ 100k/d=768 | ~300 ms |

Full breakdown on the [Performance baseline](../performance.md) page. The exact-match `get` latency is dominated by the `UPDATE last_accessed_at` per get; if you don't need cross-process LRU accuracy, you can subclass and skip it for sub-100µs gets.

## Where to go next

- **[Multi-process](../concepts/multi-process.md)** — running multiple workers against one SQLite file.
- **[Checkpoints](../guides/checkpoints.md)** — full backup/restore workflow.
- **[Performance tuning](../guides/performance-tuning.md)** — open-time optimization, eviction batching.
