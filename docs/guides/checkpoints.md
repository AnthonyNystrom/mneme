# Checkpoints

`SemanticCache.dumps()` and `SemanticCache.loads()` round-trip a full cache state - entries, namespaces, counters, dtype, embedder fingerprint - as a single `tar.gz` archive.

```python
cache.dumps("backup.tar.gz")

# Later, possibly on a different host:
restored = SemanticCache.loads("backup.tar.gz", path="restored.db", embedder=embedder)
```

Useful for:

- **Backups before destructive operations** - model swap, schema change, big eviction sweep.
- **Promoting a cache between environments** - warm staging from a prod snapshot, ship a pre-populated cache to a new region.
- **Reproducing a state** - bug reports, regression tests.

## What's in the archive

The `.tar.gz` contains:

```
manifest.json       # schema_version, embedder_fingerprint, embedder_dim,
                    # index_backend, store_backend, vector_dtype,
                    # created_at, namespaces[]
store/              # delegated to store.snapshot_to(); typically a SQLite .db file
vectors/            # optional float32 source-of-truth vectors (when present)
```

The `manifest.json` lets `loads()` validate compatibility before unpacking. Mismatched fingerprint or dim raises [`EmbedderMismatchError`](../reference/exceptions.md) / [`EmbedderDimensionError`](../reference/exceptions.md) before any data is touched.

## Round-trip example

```python
from mneme import SemanticCache

# Populate a cache
with SemanticCache(path="prod.db", embedder=embedder) as cache:
    for query, response in seed_corpus:
        cache.put(query, response)
    cache.dumps("prod-backup.tar.gz")

# Restore on a fresh host
restored = SemanticCache.loads(
    "prod-backup.tar.gz",
    path="prod-restored.db",
    embedder=embedder,
)
try:
    assert restored.stats().entries == len(seed_corpus)
    hit = restored.get("how do I cancel?")     # cached state intact
    assert hit is not None
finally:
    restored.close()
```

The cache that `loads` returns is **already opened**; you can `get`/`put` against it immediately. `close()` it when you're done as usual.

## Per-store snapshot semantics

| Store | `snapshot_to` / `restore_from` | Recommended backup path |
| --- | --- | --- |
| `MemoryStore` | raises `CheckpointError` | `cache.dumps()` works; the archive contains the in-memory state serialized |
| `SQLiteStore` | implemented (uses SQLite's online backup API) | `cache.dumps()` end-to-end |
| `RedisStore` | raises `CheckpointError` | Use `redis-cli BGSAVE` externally |
| `PostgresStore` | raises `CheckpointError` | Use `pg_dump` externally |
| `DynamoDBStore` | raises `CheckpointError` | Use AWS on-demand backup or PITR |

When a backend doesn't implement `snapshot_to`, the cache-level `dumps()` raises `CheckpointError` too - there's no path through the library, but operators have backend-native tools that are usually better anyway.

## What `dumps()` does not preserve

- **Live in-memory hnsw index state.** When the restored cache opens, it rebuilds the index from the store's vectors. If you tuned `index_options` (`M`, `ef_construction`, `ef`), the new index uses the tuning passed to `loads()`, not what was in the archive.
- **Multi-process coordinator state.** Polling thresholds, mmap-shared lock files - those are runtime-only. The restored cache picks a fresh `multi_process_mode`.
- **Metrics hooks.** Hooks live on the running process; the archive records counter values but not the hook itself. Re-attach your hook on `loads()`.

## Promotion workflow

A common production pattern: keep a "golden" cache in staging, promote it to production after validation.

```python
# Nightly job in staging:
def nightly_snapshot():
    with SemanticCache(path="staging.db", embedder=embedder) as cache:
        cache.vacuum()                  # drop expired entries first
        cache.dumps("/snapshots/staging-{:%Y-%m-%d}.tar.gz".format(datetime.utcnow()))

# Promotion job in production (manual or scheduled):
def promote(snapshot_path):
    cache = SemanticCache.loads(
        snapshot_path,
        path="prod.db",
        embedder=embedder,
        # production-specific settings:
        max_entries=500_000,
        vector_dtype="int8",
    )
    cache.close()
```

The `loads()` call validates the embedder fingerprint up front, so a staging snapshot built with the wrong model fails fast in production rather than silently mixing vectors.

## Compression

The archive uses `tar.gz` compression. For caches over a few GB on disk, the gzip step dominates `dumps()` latency. If you need faster snapshots, write your own snapshot logic against `store.iter_all()` and your preferred compression.

## Where to go next

- **[Re-embed migration](reembed-migration.md)** - when the embedder changes, dumps don't help; you need migration.
- **[SQLiteStore: snapshot/restore](../stores/sqlite.md#snapshot-restore)** - the store-level mechanics.
- **[API reference: cache](../reference/cache.md)** - `dumps()` / `loads()` signatures.
