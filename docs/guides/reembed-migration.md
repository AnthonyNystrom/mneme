# Re-embed migration

When you change the embedder model or its dimension, every cached vector in the existing cache is now incompatible. The cache validates this on `open` and refuses to mix vectors:

```python
SemanticCache(path="cache.db", embedder=NewEmbedder())
# raises EmbedderMismatchError(
#   "Stored fingerprint 'old:v1' does not match supplied 'new:v1'.
#    Remediation: open with the original embedder, or use
#    mneme.tools.migrate.reembed() to migrate."
# )
```

The supported migration path is `mneme.tools.migrate.reembed`: walk every entry through the new embedder, write to a new cache, leave the source untouched.

## The basic call

```python
from mneme import SemanticCache, MemoryStore
from mneme.tools.migrate import reembed

# Source cache - opened with the old embedder.
source = SemanticCache(path="old.db", embedder=old_embedder)

# Destination cache - opened with the new embedder.
dest = SemanticCache(path="new.db", embedder=new_embedder)

# Walks source.iter_all() through new_embedder.embed() and inserts into dest.
n = reembed(source=source, dest=dest, batch_size=128)

print(f"migrated {n} entries")
source.close()
dest.close()
```

`reembed` returns the count of entries successfully migrated.

## What's preserved, what's not

Preserved across migration:

- **Query, response, metadata** - same strings, same dict.
- **Namespace.** Each entry lands in its original namespace.
- **TTL.** If the source had `ttl=`, the dest gets the same TTL (counted from the original `created_at`, not from migration time).
- **Counters.** Per-namespace metrics counters copy over.

Not preserved:

- **Embedding bytes.** Re-computed from scratch - that's the whole point.
- **Embedder fingerprint.** Now matches the new embedder.
- **`id`s.** The dest assigns its own ids; expect them to differ from the source.
- **`last_accessed_at`.** Reset to `created_at` (migration is not an access).
- **Index backend state.** The dest builds its index fresh from the new vectors.

## Async variant

For async embedders, use `areembed`:

```python
from mneme.tools.migrate import areembed

async def migrate():
    source = SemanticCache(path="old.db", embedder=to_sync_embedder(old_async))
    dest = AsyncSemanticCache(path="new.db", embedder=new_async_embedder)
    try:
        n = await areembed(source=source, dest=dest, batch_size=128)
    finally:
        source.close()
        await dest.close()
```

The source can be either sync or async (use `to_sync_embedder` to adapt); the destination's embedder shape determines which migration variant you call.

## Cost

Re-embedding is expensive - every cached entry runs through the new embedder. For an OpenAI embedder, that's one API call per entry, billed at embedding rates. For a local model, it's one matvec per entry but on CPU/GPU.

A 100k-entry cache through `text-embedding-3-small`:

- ~100k API calls at OpenAI's pricing
- Or batched with `dimensions=` parameter and `input=` as a list - `reembed`'s `batch_size=` controls the batch size

For very large caches, consider:

1. **Migrate during a maintenance window.** The cache returns misses while migration runs (you're using the *old* cache against new traffic).
2. **Migrate top-k LRU instead of everything.** Drop the cold tail; only re-embed the hot entries.

   ```python
   def hot_iter():
       for id_ in source._store.iter_lru_ids(n=10_000):  # private API; see Custom stores
           yield source._store.get_by_id(id_)
   ```

3. **Run migration in parallel against multiple destinations.** Each destination shard handles a fraction of the corpus; the script splits by id-modulo or namespace.

## When *not* to migrate

If you're just tuning `similarity_threshold`, `vector_dtype`, or `index_backend` - **don't re-embed**. Those are runtime settings, not embedder properties:

| Change | Re-embed? |
| --- | --- |
| `similarity_threshold` | no - runtime knob |
| `vector_dtype` (fp32 ↔ fp16 ↔ int8) | no - `cache.requantize()` |
| `index_backend` (numpy ↔ hnsw) | no - `cache.requantize()` rebuilds the index from the store vectors |
| `multi_process_mode` | no - runtime knob |
| `max_entries` / `namespace_quotas` | no - runtime knob |
| Embedder model | **yes** - different vector space |
| Embedder dim (e.g. OpenAI `dimensions=`) | **yes** - different vector space |
| Tokenizer change in the same model | depends - if vectors drift, yes |

Migration is for vector-space-incompatible changes only.

## Failure handling

`reembed` doesn't checkpoint mid-stream. If the new embedder is flaky, the dest cache ends up with whatever entries succeeded before the failure. Re-running the migration is idempotent - re-puts use the same `(namespace, query_hash)` key, so the previous attempt's entries get overwritten.

For very expensive migrations, wrap your own checkpoint logic:

```python
seen_ids = set()
try:
    n = reembed(source=source, dest=dest)
except Exception:
    # Recover: dest's iter_all tells you what was migrated; rerun on the rest.
    seen_ids = {e.query_hash for e in dest._store.iter_all()}
    raise
```

Then on retry, filter `source.iter_all()` against `seen_ids` before passing to a custom migration loop.

## Where to go next

- **[Embedders concept](../concepts/embedders.md)** - why fingerprints matter.
- **[Checkpoints](checkpoints.md)** - for backup-without-migration.
- **[API reference: tools](../reference/tools.md)** - `reembed` / `areembed` signatures.
