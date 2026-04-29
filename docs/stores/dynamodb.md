# DynamoDBStore

Serverless, multi-region NoSQL `Store` backed by AWS DynamoDB. Optional install via `mneme[dynamodb]`.

```bash
pip install "mneme[dynamodb]"
```

```python
from mneme import SemanticCache
from mneme._store_dynamodb import DynamoDBStore

store = DynamoDBStore(
    table_name="my_app_cache",
    region_name="us-east-1",
    create_table=True,                      # opt-in; default False
    billing_mode="PAY_PER_REQUEST",
)
with SemanticCache(store=store, embedder=embedder) as cache:
    cache.put("hello", "world")
```

## When to pick it

- **Multi-region or AWS-native deployments.** DynamoDB Global Tables span regions natively.
- **Bursty workloads with PAY_PER_REQUEST.** No capacity to provision; you pay for actual reads/writes.
- **Serverless functions (Lambda).** No connection pool to manage; boto3 reuses HTTP connections per container.
- **You already operate AWS.** Re-use IAM, CloudWatch, KMS, etc.

For high-volume sustained read traffic, DynamoDB's per-request cost adds up. [Redis](redis.md) is cheaper at sustained-millisecond-latency scale; DynamoDB wins on operational simplicity and durability.

## Constructor

```python
DynamoDBStore(
    table_name: str,
    *,
    region_name: str | None = None,
    endpoint_url: str | None = None,             # for moto / DynamoDB Local
    aws_profile: str | None = None,
    create_table: bool = False,                  # auto-create-on-open
    billing_mode: Literal["PAY_PER_REQUEST", "PROVISIONED"] = "PAY_PER_REQUEST",
    provisioned_capacity: tuple[int, int] | None = None,  # (rcu, wcu); required if PROVISIONED
)
```

`endpoint_url` is for testing against [moto](https://github.com/getmoto/moto) (in-process AWS mock) or [DynamoDB Local](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/DynamoDBLocal.html) (Amazon's emulator container). Production: leave it unset.

`aws_profile` is forwarded to `boto3.session.Session(profile_name=...)`. Default credential chain (env vars, shared config, IAM role) is the usual path; only set this when you need a specific profile.

## Auto-create vs pre-provisioned

```python
# Dev: create the table on first open if it doesn't exist
DynamoDBStore(table_name="mycache", create_table=True)

# Prod: assume the table is provisioned by CDK / Terraform / console
DynamoDBStore(table_name="mycache", create_table=False)
```

When `create_table=False` and the table is missing, open raises `StoreBackendError` with a remediation hint.

## Schema

A single table with two GSIs:

```
Table: <table_name>
  PK: id (Number)

  Reserved item id=0 - the counter:
    next_id            (N)   next id to assign
    version_counter    (N)   bumped on every write
    embedder_fingerprint (S) stamped on first open
    embedder_dim       (N)
    meta               (M)   write_meta() values
    quotas             (M)   namespace -> max_entries

  Entry items (id >= 1):
    query_hash         (S)
    namespace          (S)
    embedding          (B)   raw float32 bytes
    response           (S)
    metadata           (S)   JSON-encoded
    confidence_score   (N)
    created_at         (N)
    last_accessed_at   (N)
    ttl                (N, optional)
    access_count       (N)

GSI gsi_hash : PK=namespace, SK=query_hash      → get_by_hash
GSI gsi_lru  : PK=namespace, SK=last_accessed_at → iter_lru_ids (per-namespace LRU)
```

The reserved `id=0` item is the cache's metadata - `next_id`, `version_counter`, fingerprint, dim, meta map, quotas map. All cache-wide metadata lives in this single item.

## Atomic writes

Every mutation is a `TransactWriteItems` pairing the data op with a counter `Update`:

```python
client.transact_write_items(TransactItems=[
    {"Put": {"TableName": ..., "Item": entry_item, "ConditionExpression": "attribute_not_exists(id)"}},
    {"Update": {
        "TableName": ..., "Key": {"id": {"N": "0"}},
        "UpdateExpression": "SET next_id = :nid, version_counter = :nver",
        "ConditionExpression": "next_id = :cur_nid AND version_counter = :cur_ver",
        ...
    }},
])
```

The `ConditionExpression` on the counter detects a concurrent insert: if another writer claimed the same `next_id`, the transaction fails with `TransactionCanceledException` and `mneme` retries (up to 5 times).

## Snapshot / restore

`DynamoDBStore.snapshot_to` raises `CheckpointError`. AWS provides better tooling than scan-and-copy for this use case:

- **On-demand backups** via `CreateBackup` - point-and-shoot, restorable to a new table.
- **Point-in-Time Recovery** (PITR) - continuous, restore to any second in the last 35 days.
- **Export to S3** via `ExportTableToPointInTime` - DynamoDB JSON or Ion format, queryable from Athena.

The cache assumes operators run AWS-native backup tooling out of band.

## Cost levers

- **`PAY_PER_REQUEST` billing** is the default and right for variable workloads. Costs scale linearly with actual reads/writes; no minimum.
- **`PROVISIONED` billing** is cheaper at sustained high throughput. You pay for capacity even when idle. Combine with auto-scaling for production fleets.
- **TTL on entries.** DynamoDB has native TTL - set the entry's `ttl` field and the row vanishes within ~48 hours of expiry without consuming write capacity. Cheaper than calling `cache.vacuum()` on a schedule.
- **`Scan` is expensive.** `iter_all`, `iter_since`, and `count(global)` use Scan. For very large tables, watch the cost; pick a different store if you Scan often.

## Auth

`mneme` uses boto3's default credential chain:

1. Env vars (`AWS_ACCESS_KEY_ID`, etc.)
2. Shared config (`~/.aws/credentials`, `~/.aws/config`)
3. IAM role (EC2, ECS, Lambda, EKS service account)

The role needs (minimum):

```
dynamodb:DescribeTable
dynamodb:GetItem
dynamodb:PutItem
dynamodb:UpdateItem
dynamodb:DeleteItem
dynamodb:Query
dynamodb:Scan
dynamodb:TransactWriteItems
dynamodb:CreateTable          (only if create_table=True)
```

KMS read access if the table is encrypted with a customer-managed key.

## Failure modes

| Failure | Behavior |
| --- | --- |
| Table missing + `create_table=False` | `StoreBackendError` with remediation pointing at `create_table=True` |
| IAM role missing required actions | `StoreBackendError` with the boto3 error chained |
| Counter contention exceeds 5 retries | `StoreBackendError("DynamoDB insert lost the counter race after 5 retries")` |
| `boto3` not installed | `StoreBackendError("requires the optional 'dynamodb' extra")` |

## Testing

For local tests without a real AWS account, use **moto**:

```python
from moto import mock_aws

with mock_aws():
    store = DynamoDBStore(table_name="test", region_name="us-east-1", create_table=True)
    # ... use the cache ...
```

Or **DynamoDB Local** via Docker for higher fidelity:

```bash
docker run -d -p 8000:8000 amazon/dynamodb-local:latest
```

```python
DynamoDBStore(
    table_name="test",
    region_name="us-east-1",
    endpoint_url="http://localhost:8000",
    create_table=True,
)
```

## Where to go next

- **[Postgres](postgres.md)** - durable alternative if you don't want AWS-native.
- **[Performance tuning](../guides/performance-tuning.md)** - DynamoDB-specific cost knobs.
- **[showcase/dynamodb_quickstart.py](https://github.com/anthonynystrom/mneme/blob/main/examples/dynamodb_quickstart.py)** - runnable example with moto.
