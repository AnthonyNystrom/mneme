# Custom stores

The `Store` Protocol from `mneme._types` is the contract every backend must satisfy. Implement it and the cache works against your code with no other changes.

## When you'd write one

- **A backend `mneme` doesn't ship.** Memcached, FoundationDB, Cassandra, Cockroach, S3-with-an-index - all reasonable.
- **Custom durability semantics.** A read replica with eventual consistency, a multi-region active/active setup, a tiered hot/cold split.
- **Compliance constraints.** A backend that encrypts at rest with a customer-managed key, or one that audits every read.
- **Testing fakes.** `tests/stores/inmemory_store.py` is the reference custom impl used to verify the conformance battery is portable.

## The Protocol

```python
class Store(Protocol):
    # --- Lifecycle ---
    def open(self, embedder_fingerprint: str, embedder_dim: int) -> None: ...
    def close(self) -> None: ...

    # --- Read ---
    def get_by_hash(self, namespace: str, query_hash: str) -> StoredEntry | None: ...
    def get_by_id(self, id: int) -> StoredEntry | None: ...
    def count(self, namespace: str | None = None) -> int: ...
    def list_namespaces(self) -> list[str]: ...
    def iter_lru_ids(self, n: int, namespace: str | None = None) -> Iterable[int]: ...
    def iter_all(self) -> Iterable[StoredEntry]: ...
    def iter_since(self, last_id: int) -> Iterable[StoredEntry]: ...

    # --- Write (must be transactional) ---
    def insert(self, entry: StoredEntry) -> int: ...
    def update_access(self, id: int, now: int) -> None: ...
    def delete_by_id(self, id: int) -> bool: ...
    def delete_expired(self, now: int, namespace: str | None = None) -> int: ...
    def clear_namespace(self, namespace: str) -> int: ...

    # --- Quotas ---
    def set_namespace_quota(self, namespace: str, max_entries: int) -> None: ...
    def get_namespace_quota(self, namespace: str) -> int | None: ...

    # --- Coordination ---
    def read_version_counter(self) -> int: ...
    def read_meta(self, key: str) -> str | None: ...
    def write_meta(self, key: str, value: str) -> None: ...

    # --- Health ---
    def integrity_check(self) -> bool: ...

    # --- Backup ---
    def snapshot_to(self, dest_path: str | Path) -> None: ...

    @classmethod
    def restore_from(cls, source_path: str | Path, dest_path: str | Path) -> Store: ...
```

24 methods. Most are 1–10 lines. The library code in `_cache.py` only depends on this Protocol; it never imports a concrete store.

## The contract

A few invariants the cache relies on:

1. **`insert` returns the assigned id.** Stable across reopens. The id-to-entry mapping must survive a `close()` / open cycle.
2. **`version_counter` increments in the same transaction as the data write.** Multi-process readers poll this; if they see the counter advance, the data must be durable.
3. **`get_by_hash` is exact-match only.** No fuzzy logic. The cache layer normalizes + hashes the query before calling.
4. **`iter_*` returns items ordered by `id`.** `iter_since(last_id)` returns entries with `id > last_id`, in ascending id order.
5. **`update_access` is idempotent and side-effect-light.** Called on every Layer-1 hit. Make it cheap.
6. **`integrity_check` returns False rather than raising on corruption.** The cache uses this on `health()` calls.

The library doesn't enforce these; the conformance battery does.

## The conformance battery

Every shipped store passes `tests/test_store_protocol_compliance.py`. It exercises 50+ behaviors against your store. Pass that battery and your store is "done" by `mneme`'s definition.

Wire your store into the parametrized fixture:

```python
# In your test file:
import pytest
from tests.test_store_protocol_compliance import _STORE_NAMES

# Add your store's identifier:
_STORE_NAMES.append("MyStore")

@pytest.fixture
def my_store(request):
    if request.param == "MyStore":
        return MyStore(...)
```

Or copy the test file into your own project and adapt the fixture parametrization to instantiate your store. The conformance assertions are independent of the test harness.

## Reference impl

[examples/custom_store.py](https://github.com/anthonynystrom/mneme/blob/main/examples/custom_store.py) is a full working `DictStore` - a Python-dict-backed `Store` that satisfies the Protocol. ~150 lines. Use it as scaffolding when you start a new backend.

The structure to follow:

1. **Constructor.** Accept connection params, pool, schema/prefix, etc.
2. **`open()`.** Connect, provision schema if missing, validate fingerprint+dim, initialize counters.
3. **`close()`.** Release connection / pool.
4. **Reads.** Direct backend lookups. Map your row format to `StoredEntry`.
5. **Writes.** Transactional. Bump `version_counter` in the same transaction.
6. **`snapshot_to` / `restore_from`.** Implement if your backend has a native backup primitive; otherwise raise `CheckpointError` (most server-backed stores do this).

## Common gotchas

- **Forgetting to bump `version_counter`.** Multi-process readers won't see your writes. The conformance test `test_version_counter_increments_on_insert` catches this.
- **Returning a different `id` on duplicate `(namespace, query_hash)`.** The cache treats `insert` of a duplicate as an upsert; the id must match the existing row, not get newly allocated. `test_insert_replaces_on_hash_collision` catches this.
- **Auto-incrementing without a transaction.** If your backend allocates ids out-of-band (e.g. UUIDs at app level, or a sequence outside the txn), the data write and id allocation can drift. Use the backend's primitives for atomic allocation.
- **`iter_*` not sorted by id.** The cache assumes ascending id. Sort if your backend doesn't natively.
- **Storing embeddings as strings or JSON.** Wasteful. Use a binary type - `BLOB` (SQLite), `BYTEA` (Postgres), Binary type (Redis/DynamoDB).

## Wiring it in

```python
from mneme import SemanticCache
from my_module import MyStore

store = MyStore(connection_string="...")
with SemanticCache(store=store, embedder=embedder) as cache:
    cache.put("hello", "world")
```

That's it. No registration, no plugin system. The Protocol is structurally typed so any object with the right methods works.

## Where to go next

- **[examples/custom_store.py](https://github.com/anthonynystrom/mneme/blob/main/examples/custom_store.py)** - a runnable reference.
- **[API reference: types](../reference/types.md#mneme._types.Store)** - the Protocol with full type hints.
- **[Stores: Memory / SQLite / Redis / Postgres / DynamoDB](../stores/memory.md)** - five implementations to learn from.
