# Stores

Every shipped `Store` backend. All five satisfy the same `Store` Protocol — see [Custom stores](../guides/custom-stores.md) for the contract.

## Always available

::: mneme._store_memory.MemoryStore

::: mneme._store_sqlite.SQLiteStore

## Optional extras

These are imported from their submodules to keep `import mneme` lightweight when their dependencies aren't installed.

::: mneme._store_redis.RedisStore

::: mneme._store_postgres.PostgresStore

::: mneme._store_dynamodb.DynamoDBStore
