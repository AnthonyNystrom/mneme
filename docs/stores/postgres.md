# PostgresStore

Durable, transactional, network-backed `Store`. Optional install via `mneme[postgres]`.

```bash
pip install "mneme-cache[postgres]"
```

```python
from mneme import SemanticCache
from mneme._store_postgres import PostgresStore

store = PostgresStore(
    dsn="postgresql://user:pass@host:5432/dbname",
    schema="mneme",
)
with SemanticCache(store=store, embedder=embedder) as cache:
    cache.put("hello", "world")
```

`PostgresStore` lives at `mneme._store_postgres` (private module) so `import mneme` doesn't pull `psycopg` when the extra isn't installed.

## When to pick it

- **Multi-host deployments where you already run Postgres.** Re-use the same primary; `mneme`'s schema lives in its own namespace.
- **Durability matters.** Postgres is ACID; a hard crash without `synchronous_commit=off` doesn't lose committed entries.
- **You want SQL-level introspection.** Operators can query the cache schema directly - `SELECT count(*) FROM mneme.entries WHERE namespace = 'tenant_a'` answers ad-hoc questions without going through `mneme`.

For ephemeral caches with low durability needs, [Redis](redis.md) is faster and lighter.

## Constructor

Provide exactly one connection source:

```python
PostgresStore(
    dsn: str | None = None,                       # one of these three
    *,
    connection: psycopg.Connection | None = None,
    pool: Any = None,                              # any psycopg-style pool with getconn() / putconn()
    schema: str = "mneme",
)
```

- **`dsn`** is the simple case. The store opens its own connection.
- **`connection`** lets you pass an existing one - useful if you've already configured TLS, application_name, etc.
- **`pool`** integrates with `psycopg-pool`'s `ConnectionPool` (or any compatible pool). The cache acquires a connection per call.

`schema` is whitelisted at construction (alphanumeric + underscore only) so the DDL formatting is safe. Same schema can be reused across services - but they share the same cache state, so usually you want one schema per service.

## Schema

The store creates these tables under your `schema`:

| Table | Purpose |
| --- | --- |
| `entries` | One row per cached entry - `id BIGSERIAL`, namespace, query_hash, query, response, embedding `BYTEA`, metadata `JSONB`, created_at, last_accessed_at, ttl, access_count |
| `cache_counters` | Per-namespace counters - primary key `(namespace, name)` |
| `namespace_quotas` | Per-namespace `max_entries` |
| `multi_process_state` | `version_counter`, schema_version |
| `schema_meta` | `embedder_fingerprint`, `embedder_dim`, schema_version |

Indexes:

- `idx_entries_ns_hash` on `(namespace, query_hash)` - Layer-1 lookup
- `idx_entries_ns_lru` on `(namespace, last_accessed_at)` - eviction
- `idx_entries_created_at` - TTL vacuum

`UNIQUE(namespace, query_hash)` enforces at-most-one cached entry per (tenant, query) pair.

## Atomic writes

Every mutation runs inside `BEGIN`/`COMMIT` with autocommit forced on the connection so each `with conn.transaction():` block is a top-level transaction. The `version_counter` UPDATE is in the same transaction as the data write:

```sql
BEGIN;
INSERT INTO mneme.entries (...) VALUES (..., ...) RETURNING id;
UPDATE mneme.multi_process_state
   SET value = (CAST(value AS BIGINT) + 1)::TEXT
 WHERE key = 'version_counter';
COMMIT;
```

`stale-tolerant` polling against a Postgres store sees the counter advance the same way it would against SQLite, just over a network hop.

## Snapshot / restore

`PostgresStore.snapshot_to` raises `CheckpointError`. The library does not subprocess to `pg_dump` for you. Use Postgres-native tooling:

```bash
pg_dump --schema=mneme mydb > backup.sql
psql mydb_restore < backup.sql
```

Or use Postgres's logical replication / point-in-time recovery - out of scope for `mneme` but standard ops territory.

## Auth and TLS

Standard `psycopg` DSN parsing. For AWS RDS or Azure Database for PostgreSQL, the connection string with `sslmode=require` works:

```
postgresql://user:pass@my-rds.us-east-1.rds.amazonaws.com:5432/mydb?sslmode=require
```

For IAM-based auth (RDS), pass a `connection=` you constructed yourself with the IAM token.

## Connection pooling

Two patterns:

=== "DSN - open per-store"

    ```python
    store = PostgresStore(dsn="postgresql://...", schema="mneme")
    ```

    One physical connection per store. Fine for single-process apps and notebooks.

=== "psycopg-pool - share across stores"

    ```python
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool("postgresql://...", min_size=1, max_size=20)
    store = PostgresStore(pool=pool, schema="mneme")
    ```

    The store calls `pool.getconn()` per operation and `pool.putconn()` on `close()`. For multi-process workers (Gunicorn, etc.), one pool per process, sized to your concurrency.

## Multi-tenant pitfalls

- **One schema per service.** Don't share `mneme` schemas between unrelated services - they'll fight over the `version_counter` and inflate each other's stale-check work.
- **Permissions.** The role used by `mneme` needs `CREATE` on first open (it provisions the schema + tables) and `INSERT/UPDATE/DELETE/SELECT` thereafter. For pre-provisioned schemas, the role can drop to data-only privileges.
- **Long-running transactions hold locks.** `mneme`'s own transactions are short, but if you wrap `cache.put` calls inside a larger application transaction, you can serialize against your own writers.

## Failure modes

| Failure | Behavior |
| --- | --- |
| DSN refuses connection on open | `StoreBackendError` with the DSN in the message |
| Schema role lacks `CREATE` privileges | `StoreBackendError` from the DDL step on first open |
| `psycopg` not installed | `StoreBackendError("requires the optional 'postgres' extra")` |
| Server disconnects mid-call | `psycopg` raises; the cache wraps and re-raises as `StoreBackendError` with the underlying error chained |

## Where to go next

- **[Multi-process](../concepts/multi-process.md)** - Postgres + stale-tolerant for multi-worker apps.
- **[Performance tuning](../guides/performance-tuning.md)** - pool sizing, connection-per-call cost.
- **[DynamoDB](dynamodb.md)** - serverless alternative when you don't want a Postgres to operate.
