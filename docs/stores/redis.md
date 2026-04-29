# RedisStore

Network-backed `Store` for sharing a cache across hosts. Optional install via `mneme[redis]`.

```bash
pip install "mneme-cache[redis]"
```

```python
from mneme import SemanticCache
from mneme._store_redis import RedisStore         # imported from the private module

store = RedisStore(url="redis://localhost:6379/0", key_prefix="mneme")
with SemanticCache(store=store, embedder=embedder) as cache:
    cache.put("hello", "world")
```

`RedisStore` lives at `mneme._store_redis`, not `mneme`, so `import mneme` doesn't pull `redis` when the extra isn't installed. Same convention applies to Postgres and DynamoDB.

## When to pick it

- **Multiple hosts share a cache.** Web frontends, worker fleets, or geographically separated services that should agree on which queries have been answered.
- **You already run Redis.** Avoid adding a second piece of infrastructure for caching.
- **Latency matters more than durability.** Redis is in-memory by default; a hard crash without RDB or AOF loses recent writes. For a cache, that's usually fine.

For durable cross-host caches, prefer [Postgres](postgres.md) or [DynamoDB](dynamodb.md).

## Constructor

```python
RedisStore(
    url: str,                      # redis://[user:pass@]host:port/db, or rediss:// for TLS
    *,
    key_prefix: str = "mneme",     # all keys live under this prefix
    use_native_ttl: bool = False,  # if True, set EXPIREAT on entries with ttl
    decode_responses: bool = False,
)
```

`use_native_ttl=True` lets Redis evict expired entries on its own schedule - useful when you want the cache size to drift back down without calling `vacuum()`. The cache's own TTL logic still runs; native TTL is the belt-and-braces. For caches without TTLs the flag is a no-op.

## Key layout

Under your prefix (default `"mneme"`):

| Key | Type | Purpose |
| --- | --- | --- |
| `mneme:meta` | hash | embedder_fingerprint, embedder_dim, schema_version |
| `mneme:version` | string (int) | version_counter - incremented on every write |
| `mneme:entry:{id}` | hash | one cached entry's fields |
| `mneme:hash:{ns}:{query_hash}` | string (id) | namespace + query_hash → id reverse index |
| `mneme:lru:{ns}` | sorted set | (id, last_accessed_at) for LRU iteration |
| `mneme:by_id` | sorted set | global (id, id) for iter_all / iter_since |
| `mneme:by_ttl` | sorted set | (id, expire_at) for vacuum |
| `mneme:counters:{ns}` | hash | per-namespace metrics counters |
| `mneme:quotas` | hash | per-namespace max_entries |

Multiple `mneme` caches against the same Redis instance must use **different `key_prefix`** values. The conformance battery uses one per test to keep them isolated.

## Atomic writes

Every mutation pipelines a `MULTI` / `EXEC` transaction including the `INCR mneme:version`:

```python
pipe = client.pipeline(transaction=True)
pipe.hset(f"mneme:entry:{id}", mapping=fields)
pipe.set(f"mneme:hash:{ns}:{qh}", id)
pipe.zadd(f"mneme:lru:{ns}", {id: ts})
pipe.zadd("mneme:by_id", {id: id})
pipe.incr("mneme:version")
pipe.execute()
```

That's the same same-txn-as-version-counter invariant that makes [stale-tolerant multi-process mode](../concepts/multi-process.md#stale-tolerant) work - but now across hosts.

## Snapshot / restore

`RedisStore.snapshot_to` raises `CheckpointError`. The library does not subprocess to `redis-cli BGSAVE` for you; use it externally:

```bash
redis-cli BGSAVE                       # async background save
# wait for completion, then copy:
cp /var/lib/redis/dump.rdb backup.rdb
```

Restore by stopping Redis, replacing `dump.rdb`, restarting. The cache picks up automatically on next open.

For the application-level `dumps()` / `loads()` round-trip with a Redis-backed cache, the cache will raise `CheckpointError` because the underlying store doesn't support it. Use a SQLite cache for round-trippable backups, or use Redis-native tooling.

## Auth and TLS

Standard `redis://` URL semantics:

```
redis://username:password@host:port/db_index
rediss://username:password@host:port/db_index    # TLS
unix:///var/run/redis.sock?db=0                   # Unix socket
```

For AWS ElastiCache, use the `rediss://` URL with the cluster endpoint. For Azure Cache for Redis, the same. The library doesn't ship special handling for either - `redis-py` does the right thing.

## Multi-tenant pitfalls

When two applications share a Redis instance:

- **Distinct `key_prefix`.** A second `RedisStore(url=..., key_prefix="other_app")` shares nothing with the first.
- **Different ACL users** (Redis 6+) so app A can't `KEYS mneme:*` against app B's keys. Per-prefix ACLs are out of scope; use Redis-native auth.
- **Watch for `KEYS` / `FLUSHDB` from neighbors.** `mneme` never calls them, but operators sometimes do. Document the prefix contract in your runbook.

## Failure modes

| Failure | Behavior |
| --- | --- |
| Redis is unreachable on `cache.open()` | `StoreBackendError` with the URL in the message |
| Redis dies mid-call | `redis-py` raises; the cache wraps and re-raises as `StoreBackendError` |
| TLS cert validation fails | `StoreBackendError` with the underlying SSL error chained |
| `redis` package not installed | `StoreBackendError("requires the optional 'redis' extra")` |

## Where to go next

- **[Multi-process](../concepts/multi-process.md)** - pairing RedisStore with stale-tolerant polling.
- **[Performance tuning](../guides/performance-tuning.md)** - pipelining, pool sizing.
- **[Custom stores](../guides/custom-stores.md)** - using RedisStore as a starting point for, e.g., Memcached.
