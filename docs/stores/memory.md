# MemoryStore

A dict-backed `Store` for tests, scratch work, and ephemeral caches. Always available - part of the core install, no extra needed.

```python
from mneme import MemoryStore, SemanticCache

with SemanticCache(store=MemoryStore(), embedder=embedder) as cache:
    cache.put("hello", "world")
```

## When to pick it

- **Tests.** All of `mneme`'s own tests and the conformance battery use `MemoryStore` for the fast-path scenarios. It eliminates filesystem and network from the loop.
- **Short-lived processes.** A batch job that warms up, runs through a fixed corpus, and exits doesn't need durable storage.
- **Embedded one-shot workloads.** Notebook-style exploration where a fresh cache per session is the right behavior.

For everything else - anything that needs to survive a process restart - pick [`SQLiteStore`](sqlite.md) or a network-backed store.

## Semantics

- **Backed by a Python `dict`.** Lookup is O(1). Deletion is O(1). Iteration over `iter_all` is O(n).
- **Thread-safe.** Internal `RLock`; safe across threads in one process. The cache layer holds its own `RLock` on top, so concurrency through `SemanticCache` is fully serialized.
- **No persistence.** When the process exits, every entry is gone. No file is written.
- **No multi-process visibility.** Two processes each get their own dict, even if both code paths instantiate `MemoryStore()`.

## What's inside

```python
class MemoryStore:
    def __init__(self) -> None: ...
    # Implements the full Store Protocol. See API reference.
```

The constructor takes nothing. Open with `cache.open(embedder.fingerprint, embedder.dim)` (the cache class does this automatically); the store stamps the fingerprint on first open and validates on subsequent opens.

## snapshot / restore

`MemoryStore.snapshot_to(dest)` raises `CheckpointError` - there's no on-disk format to copy. If you want a serialization path for an in-memory cache, use the cache-level `dumps()` / `loads()` round-trip:

```python
cache.dumps("backup.tar.gz")
# ... later ...
restored = SemanticCache.loads("backup.tar.gz", path="restored.db", embedder=embedder)
```

`loads` writes to a SQLiteStore by default; you can't round-trip back into a MemoryStore via the checkpoint path (use [`MemoryStore` re-create + `iter_all` insert] in your own code if you really need that).

## Conformance

`MemoryStore` passes the same `tests/test_store_protocol_compliance.py` battery as every other store: `get_by_hash`, `get_by_id`, `iter_all`, `iter_lru_ids`, version_counter, namespace quotas, integrity_check, et al.

That's the contract. If you write a custom dict-like store, the conformance battery is your yardstick.

## Where to go next

- **[SQLiteStore](sqlite.md)** - the durable default.
- **[Custom stores](../guides/custom-stores.md)** - implement your own backend.
- **[API reference: stores](../reference/stores.md)** - full method list.
