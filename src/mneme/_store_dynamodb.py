"""``DynamoDBStore``: AWS DynamoDB-backed Store. Optional ``[dynamodb]`` extra.

Cross-host shared cache backed by a managed serverless NoSQL store. Every
mutation pairs the data op with a ``version_counter`` bump inside a single
``TransactWriteItems`` call so multi-process readers can detect changes
(per the same-txn invariant the other server-backed stores satisfy).

Schema (single table + 2 GSIs):

- ``id`` is the partition key (Number).
- The reserved item at ``id=0`` is the *counter*: it stores ``next_id``,
  ``version_counter``, the embedder fingerprint/dim, the ``meta`` map, and
  the namespace quotas map.
- Real entries occupy ``id >= 1``.
- GSI ``gsi_hash`` (PK=``namespace``, SK=``query_hash``) backs ``get_by_hash``.
- GSI ``gsi_lru`` (PK=``namespace``, SK=``last_accessed_at``) backs
  ``iter_lru_ids``.

``snapshot_to`` and ``restore_from`` raise ``CheckpointError`` (matches
``PostgresStore`` / ``RedisStore``) — use AWS native backups instead.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ._exceptions import (
    CacheClosedError,
    CheckpointError,
    EmbedderDimensionError,
    EmbedderMismatchError,
    StoreBackendError,
)
from ._types import StoredEntry

if TYPE_CHECKING:
    pass


def _import_boto3() -> Any:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover
        raise StoreBackendError(
            "DynamoDBStore requires the optional 'dynamodb' extra. "
            "Remediation: pip install mneme[dynamodb]"
        ) from exc
    return boto3


_COUNTER_ID = 0
_GSI_HASH = "gsi_hash"
_GSI_LRU = "gsi_lru"
_TXN_RETRY_LIMIT = 5
_TXN_RETRY_BACKOFF_SEC = 0.01


def _to_int(v: Any) -> int:
    """boto3 returns ``Decimal`` for Number attrs; coerce to ``int``."""
    if isinstance(v, Decimal):
        return int(v)
    return int(v)


def _to_dec(v: int) -> Decimal:
    """DynamoDB Number attrs require ``Decimal`` from the resource API."""
    return Decimal(v)


class DynamoDBStore:
    """DynamoDB-backed Store. Optional ``[dynamodb]`` extra.

    Authentication uses boto3's default credential chain (env vars / shared
    config / IAM role). Pass ``endpoint_url`` to target moto or a local
    DynamoDB emulator.
    """

    def __init__(
        self,
        table_name: str,
        *,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        aws_profile: str | None = None,
        create_table: bool = False,
        billing_mode: Literal["PAY_PER_REQUEST", "PROVISIONED"] = "PAY_PER_REQUEST",
        provisioned_capacity: tuple[int, int] | None = None,
    ) -> None:
        if not table_name:
            raise ValueError(
                "DynamoDBStore: table_name must be a non-empty string. "
                "Remediation: pass the DynamoDB table name."
            )
        if billing_mode == "PROVISIONED" and provisioned_capacity is None:
            raise ValueError(
                "DynamoDBStore: provisioned_capacity=(rcu, wcu) is required "
                "when billing_mode='PROVISIONED'. Remediation: pass a tuple, "
                "or use billing_mode='PAY_PER_REQUEST'."
            )
        if billing_mode == "PAY_PER_REQUEST" and provisioned_capacity is not None:
            raise ValueError(
                "DynamoDBStore: provisioned_capacity is only valid with "
                "billing_mode='PROVISIONED'. Remediation: drop the parameter "
                "or set billing_mode='PROVISIONED'."
            )
        self._table_name = table_name
        self._region_name = region_name
        self._endpoint_url = endpoint_url
        self._aws_profile = aws_profile
        self._create_table = create_table
        self._billing_mode = billing_mode
        self._provisioned_capacity = provisioned_capacity
        self._client: Any = None
        self._table: Any = None
        self._closed = False

    # --- Lifecycle ---

    def open(self, embedder_fingerprint: str, embedder_dim: int) -> None:
        if self._closed:
            raise CacheClosedError("DynamoDBStore was closed. Remediation: create a new instance.")
        if self._client is None:
            boto3 = _import_boto3()
            try:
                session_kwargs: dict[str, Any] = {}
                if self._aws_profile is not None:
                    session_kwargs["profile_name"] = self._aws_profile
                session = boto3.session.Session(**session_kwargs)
                client_kwargs: dict[str, Any] = {}
                if self._region_name is not None:
                    client_kwargs["region_name"] = self._region_name
                if self._endpoint_url is not None:
                    client_kwargs["endpoint_url"] = self._endpoint_url
                self._client = session.client("dynamodb", **client_kwargs)
                self._table = session.resource("dynamodb", **client_kwargs).Table(self._table_name)
            except Exception as exc:
                raise StoreBackendError(
                    f"Failed to construct DynamoDB client for table "
                    f"{self._table_name!r}. Remediation: verify credentials, "
                    "region, and (if used) endpoint_url."
                ) from exc

        self._ensure_table()
        self._ensure_counter(embedder_fingerprint, embedder_dim)

    def _ensure_table(self) -> None:
        try:
            self._client.describe_table(TableName=self._table_name)
            return
        except self._client.exceptions.ResourceNotFoundException:
            pass
        except Exception as exc:
            raise StoreBackendError(
                f"DescribeTable failed for {self._table_name!r}: {exc}. Remediation: see __cause__."
            ) from exc

        if not self._create_table:
            raise StoreBackendError(
                f"DynamoDB table {self._table_name!r} does not exist. "
                "Remediation: pass create_table=True, or pre-provision the "
                "table via CDK/Terraform/console with PK 'id' (Number), "
                f"GSI {_GSI_HASH!r} on (namespace, query_hash), and GSI "
                f"{_GSI_LRU!r} on (namespace, last_accessed_at)."
            )

        attr_defs = [
            {"AttributeName": "id", "AttributeType": "N"},
            {"AttributeName": "namespace", "AttributeType": "S"},
            {"AttributeName": "query_hash", "AttributeType": "S"},
            {"AttributeName": "last_accessed_at", "AttributeType": "N"},
        ]
        key_schema = [{"AttributeName": "id", "KeyType": "HASH"}]
        gsi_hash_def: dict[str, Any] = {
            "IndexName": _GSI_HASH,
            "KeySchema": [
                {"AttributeName": "namespace", "KeyType": "HASH"},
                {"AttributeName": "query_hash", "KeyType": "RANGE"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        }
        gsi_lru_def: dict[str, Any] = {
            "IndexName": _GSI_LRU,
            "KeySchema": [
                {"AttributeName": "namespace", "KeyType": "HASH"},
                {"AttributeName": "last_accessed_at", "KeyType": "RANGE"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        }
        kwargs: dict[str, Any] = {
            "TableName": self._table_name,
            "AttributeDefinitions": attr_defs,
            "KeySchema": key_schema,
            "GlobalSecondaryIndexes": [gsi_hash_def, gsi_lru_def],
        }
        if self._billing_mode == "PAY_PER_REQUEST":
            kwargs["BillingMode"] = "PAY_PER_REQUEST"
        else:
            assert self._provisioned_capacity is not None
            rcu, wcu = self._provisioned_capacity
            kwargs["BillingMode"] = "PROVISIONED"
            kwargs["ProvisionedThroughput"] = {
                "ReadCapacityUnits": rcu,
                "WriteCapacityUnits": wcu,
            }
            gsi_hash_def["ProvisionedThroughput"] = {
                "ReadCapacityUnits": rcu,
                "WriteCapacityUnits": wcu,
            }
            gsi_lru_def["ProvisionedThroughput"] = {
                "ReadCapacityUnits": rcu,
                "WriteCapacityUnits": wcu,
            }
        try:
            self._client.create_table(**kwargs)
            waiter = self._client.get_waiter("table_exists")
            waiter.wait(TableName=self._table_name)
        except Exception as exc:
            raise StoreBackendError(
                f"CreateTable failed for {self._table_name!r}: {exc}. "
                "Remediation: see __cause__; ensure the IAM role has "
                "dynamodb:CreateTable on the target account/region."
            ) from exc

    def _ensure_counter(self, embedder_fingerprint: str, embedder_dim: int) -> None:
        resp = self._table.get_item(Key={"id": _to_dec(_COUNTER_ID)}, ConsistentRead=True)
        item = resp.get("Item")
        if item is None:
            self._table.put_item(
                Item={
                    "id": _to_dec(_COUNTER_ID),
                    "next_id": _to_dec(1),
                    "version_counter": _to_dec(0),
                    "embedder_fingerprint": embedder_fingerprint,
                    "embedder_dim": _to_dec(embedder_dim),
                    "meta": {},
                    "quotas": {},
                },
                ConditionExpression="attribute_not_exists(id)",
            )
            return
        stored_fp = str(item.get("embedder_fingerprint", ""))
        if stored_fp != embedder_fingerprint:
            raise EmbedderMismatchError(
                f"Stored fingerprint {stored_fp!r} does not match supplied "
                f"{embedder_fingerprint!r}. Remediation: open with the original "
                "embedder, or use mneme.tools.migrate.reembed() to migrate."
            )
        stored_dim = _to_int(item.get("embedder_dim", 0))
        if stored_dim != embedder_dim:
            raise EmbedderDimensionError(
                f"Stored embedder_dim={stored_dim} does not match supplied "
                f"{embedder_dim}. Remediation: use reembed() to change dimension."
            )

    def close(self) -> None:
        self._client = None
        self._table = None
        self._closed = True

    def _table_or_fail(self) -> Any:
        if self._closed:
            raise CacheClosedError("DynamoDBStore is closed.")
        if self._table is None:
            raise CacheClosedError("DynamoDBStore not opened. Remediation: call open() first.")
        return self._table

    def _client_or_fail(self) -> Any:
        if self._closed:
            raise CacheClosedError("DynamoDBStore is closed.")
        if self._client is None:
            raise CacheClosedError("DynamoDBStore not opened. Remediation: call open() first.")
        return self._client

    # --- Internal helpers ---

    def _read_counter(self) -> dict[str, Any]:
        resp = self._table_or_fail().get_item(Key={"id": _to_dec(_COUNTER_ID)}, ConsistentRead=True)
        item = resp.get("Item")
        if item is None:
            raise StoreBackendError(
                "DynamoDBStore counter row is missing. Remediation: open() "
                "must be called before any other method."
            )
        return item

    @staticmethod
    def _item_to_entry(item: dict[str, Any]) -> StoredEntry:
        meta_raw = item.get("metadata", "{}")
        meta = meta_raw if isinstance(meta_raw, dict) else json.loads(str(meta_raw))
        embedding = item.get("embedding", b"")
        if hasattr(embedding, "value"):
            embedding = embedding.value  # boto3 Binary wrapper
        ttl_val = item.get("ttl")
        return StoredEntry(
            id=_to_int(item["id"]),
            namespace=str(item["namespace"]),
            query_hash=str(item["query_hash"]),
            query=str(item.get("query", "")),
            response=str(item.get("response", "")),
            embedding=bytes(embedding),
            metadata=meta,
            created_at=_to_int(item.get("created_at", 0)),
            last_accessed_at=_to_int(item.get("last_accessed_at", 0)),
            ttl=None if ttl_val is None else _to_int(ttl_val),
            access_count=_to_int(item.get("access_count", 0)),
        )

    def _entry_to_item(self, entry: StoredEntry, row_id: int) -> dict[str, Any]:
        from boto3.dynamodb.types import Binary

        item: dict[str, Any] = {
            "id": _to_dec(row_id),
            "namespace": entry.namespace,
            "query_hash": entry.query_hash,
            "query": entry.query,
            "response": entry.response,
            "embedding": Binary(entry.embedding),
            "metadata": json.dumps(entry.metadata),
            "created_at": _to_dec(entry.created_at),
            "last_accessed_at": _to_dec(entry.last_accessed_at),
            "access_count": _to_dec(entry.access_count),
        }
        if entry.ttl is not None:
            item["ttl"] = _to_dec(entry.ttl)
        return item

    def _bump_version_only(self) -> None:
        """Increment version_counter on the counter item. Used after data ops
        that aren't already part of a TransactWriteItems with the counter."""
        self._table_or_fail().update_item(
            Key={"id": _to_dec(_COUNTER_ID)},
            UpdateExpression="ADD version_counter :one",
            ExpressionAttributeValues={":one": _to_dec(1)},
        )

    # --- Read ---

    def get_by_hash(self, namespace: str, query_hash: str) -> StoredEntry | None:
        resp = self._table_or_fail().query(
            IndexName=_GSI_HASH,
            KeyConditionExpression=("#ns = :ns AND query_hash = :qh"),
            ExpressionAttributeNames={"#ns": "namespace"},
            ExpressionAttributeValues={":ns": namespace, ":qh": query_hash},
            Limit=1,
        )
        items = resp.get("Items", [])
        if not items:
            return None
        return self._item_to_entry(items[0])

    def get_by_id(self, id: int) -> StoredEntry | None:
        if id <= 0:
            return None
        resp = self._table_or_fail().get_item(Key={"id": _to_dec(id)})
        item = resp.get("Item")
        if item is None:
            return None
        return self._item_to_entry(item)

    def count(self, namespace: str | None = None) -> int:
        # DynamoDB has no constant-time row count; Scan with Select=COUNT.
        client = self._client_or_fail()
        kwargs: dict[str, Any] = {
            "TableName": self._table_name,
            "Select": "COUNT",
            "FilterExpression": "id > :zero",
            "ExpressionAttributeValues": {":zero": {"N": "0"}},
        }
        if namespace is not None:
            kwargs["FilterExpression"] = "id > :zero AND #ns = :ns"
            kwargs["ExpressionAttributeNames"] = {"#ns": "namespace"}
            kwargs["ExpressionAttributeValues"] = {
                ":zero": {"N": "0"},
                ":ns": {"S": namespace},
            }
        total = 0
        last: dict[str, Any] | None = None
        while True:
            if last is not None:
                kwargs["ExclusiveStartKey"] = last
            resp = client.scan(**kwargs)
            total += int(resp.get("Count", 0))
            last = resp.get("LastEvaluatedKey")
            if last is None:
                break
        return total

    def list_namespaces(self) -> list[str]:
        seen: set[str] = set()
        last: dict[str, Any] | None = None
        kwargs: dict[str, Any] = {
            "ProjectionExpression": "#ns",
            "ExpressionAttributeNames": {"#ns": "namespace"},
            "FilterExpression": "id > :zero",
            "ExpressionAttributeValues": {":zero": _to_dec(0)},
        }
        while True:
            if last is not None:
                kwargs["ExclusiveStartKey"] = last
            resp = self._table_or_fail().scan(**kwargs)
            for item in resp.get("Items", []):
                ns = item.get("namespace")
                if ns is not None:
                    seen.add(str(ns))
            last = resp.get("LastEvaluatedKey")
            if last is None:
                break
        return sorted(seen)

    def iter_lru_ids(self, n: int, namespace: str | None = None) -> Iterator[int]:
        if n <= 0:
            return iter(())
        if namespace is not None:
            resp = self._table_or_fail().query(
                IndexName=_GSI_LRU,
                KeyConditionExpression="#ns = :ns",
                ExpressionAttributeNames={"#ns": "namespace"},
                ExpressionAttributeValues={":ns": namespace},
                ScanIndexForward=True,
                Limit=n,
            )
            return iter([_to_int(item["id"]) for item in resp.get("Items", [])])
        # Global LRU: query each namespace, merge sort by last_accessed_at, take n.
        merged: list[tuple[int, int]] = []  # (last_accessed_at, id)
        for ns in self.list_namespaces():
            resp = self._table_or_fail().query(
                IndexName=_GSI_LRU,
                KeyConditionExpression="#ns = :ns",
                ExpressionAttributeNames={"#ns": "namespace"},
                ExpressionAttributeValues={":ns": ns},
                ScanIndexForward=True,
                Limit=n,
            )
            for item in resp.get("Items", []):
                merged.append((_to_int(item["last_accessed_at"]), _to_int(item["id"])))
        merged.sort()
        return iter([row_id for _ts, row_id in merged[:n]])

    def iter_all(self) -> Iterator[StoredEntry]:
        return self._iter_scan(after_id=0)

    def iter_since(self, last_id: int) -> Iterator[StoredEntry]:
        return self._iter_scan(after_id=last_id)

    def _iter_scan(self, *, after_id: int) -> Iterator[StoredEntry]:
        items: list[dict[str, Any]] = []
        last: dict[str, Any] | None = None
        kwargs: dict[str, Any] = {
            "FilterExpression": "id > :cutoff",
            "ExpressionAttributeValues": {":cutoff": _to_dec(max(after_id, 0))},
        }
        while True:
            if last is not None:
                kwargs["ExclusiveStartKey"] = last
            resp = self._table_or_fail().scan(**kwargs)
            items.extend(resp.get("Items", []))
            last = resp.get("LastEvaluatedKey")
            if last is None:
                break
        items.sort(key=lambda it: _to_int(it["id"]))
        return iter([self._item_to_entry(it) for it in items])

    # --- Write ---

    def insert(self, entry: StoredEntry) -> int:
        client = self._client_or_fail()
        # Hash collision → treat as upsert: same id, no counter bump for next_id
        # but version_counter still bumped since data changed.
        existing = self.get_by_hash(entry.namespace, entry.query_hash)
        if existing is not None:
            row_id = existing.id
            try:
                client.transact_write_items(
                    TransactItems=[
                        {
                            "Put": {
                                "TableName": self._table_name,
                                "Item": _to_dynamodb_item(self._entry_to_item(entry, row_id)),
                            }
                        },
                        {
                            "Update": {
                                "TableName": self._table_name,
                                "Key": {"id": {"N": str(_COUNTER_ID)}},
                                "UpdateExpression": "ADD version_counter :one",
                                "ExpressionAttributeValues": {
                                    ":one": {"N": "1"},
                                },
                            }
                        },
                    ]
                )
            except Exception as exc:
                raise StoreBackendError(
                    f"DynamoDB upsert failed: {exc}. Remediation: see __cause__."
                ) from exc
            return row_id

        # Fresh insert: read counter, transact (Put new entry + bump counter
        # under conditional check). Retry on conflict.
        for attempt in range(_TXN_RETRY_LIMIT):
            counter = self._read_counter()
            new_id = _to_int(counter["next_id"])
            cur_version = _to_int(counter.get("version_counter", 0))
            try:
                client.transact_write_items(
                    TransactItems=[
                        {
                            "Put": {
                                "TableName": self._table_name,
                                "Item": _to_dynamodb_item(self._entry_to_item(entry, new_id)),
                                "ConditionExpression": "attribute_not_exists(id)",
                            }
                        },
                        {
                            "Update": {
                                "TableName": self._table_name,
                                "Key": {"id": {"N": str(_COUNTER_ID)}},
                                "UpdateExpression": ("SET next_id = :nid, version_counter = :nver"),
                                "ConditionExpression": (
                                    "next_id = :cur_nid AND version_counter = :cur_ver"
                                ),
                                "ExpressionAttributeValues": {
                                    ":nid": {"N": str(new_id + 1)},
                                    ":nver": {"N": str(cur_version + 1)},
                                    ":cur_nid": {"N": str(new_id)},
                                    ":cur_ver": {"N": str(cur_version)},
                                },
                            }
                        },
                    ]
                )
                return new_id
            except client.exceptions.TransactionCanceledException as exc:
                if attempt == _TXN_RETRY_LIMIT - 1:
                    raise StoreBackendError(
                        f"DynamoDB insert lost the counter race after "
                        f"{_TXN_RETRY_LIMIT} retries. Remediation: reduce "
                        "concurrent writers, or check for orphaned rows at "
                        f"id={new_id} in {self._table_name!r}."
                    ) from exc
                time.sleep(_TXN_RETRY_BACKOFF_SEC * (attempt + 1))
            except Exception as exc:
                raise StoreBackendError(
                    f"DynamoDB insert failed: {exc}. Remediation: see __cause__."
                ) from exc
        # Unreachable: loop returns or raises.
        raise StoreBackendError("DynamoDB insert: exhausted retries.")

    def update_access(self, id: int, now: int) -> None:
        if id <= 0:
            return
        client = self._client_or_fail()
        try:
            client.transact_write_items(
                TransactItems=[
                    {
                        "Update": {
                            "TableName": self._table_name,
                            "Key": {"id": {"N": str(id)}},
                            "UpdateExpression": (
                                "SET last_accessed_at = :now ADD access_count :one"
                            ),
                            "ConditionExpression": "attribute_exists(id)",
                            "ExpressionAttributeValues": {
                                ":now": {"N": str(now)},
                                ":one": {"N": "1"},
                            },
                        }
                    },
                    {
                        "Update": {
                            "TableName": self._table_name,
                            "Key": {"id": {"N": str(_COUNTER_ID)}},
                            "UpdateExpression": "ADD version_counter :one",
                            "ExpressionAttributeValues": {":one": {"N": "1"}},
                        }
                    },
                ]
            )
        except client.exceptions.TransactionCanceledException:
            return  # row didn't exist → no-op per the protocol contract
        except Exception as exc:
            raise StoreBackendError(
                f"DynamoDB update_access failed: {exc}. Remediation: see __cause__."
            ) from exc

    def delete_by_id(self, id: int) -> bool:
        if id <= 0:
            return False
        client = self._client_or_fail()
        try:
            client.transact_write_items(
                TransactItems=[
                    {
                        "Delete": {
                            "TableName": self._table_name,
                            "Key": {"id": {"N": str(id)}},
                            "ConditionExpression": "attribute_exists(id)",
                        }
                    },
                    {
                        "Update": {
                            "TableName": self._table_name,
                            "Key": {"id": {"N": str(_COUNTER_ID)}},
                            "UpdateExpression": "ADD version_counter :one",
                            "ExpressionAttributeValues": {":one": {"N": "1"}},
                        }
                    },
                ]
            )
        except client.exceptions.TransactionCanceledException:
            return False
        except Exception as exc:
            raise StoreBackendError(
                f"DynamoDB delete_by_id failed: {exc}. Remediation: see __cause__."
            ) from exc
        return True

    def delete_expired(self, now: int, namespace: str | None = None) -> int:
        # DynamoDB FilterExpression has no arithmetic, so we can't push
        # ``created_at + ttl <= now`` server-side. Pull the candidate rows
        # (those with a ttl set, optionally namespace-scoped) and filter in
        # Python. The version_counter bump happens atomically per row inside
        # ``delete_by_id``.
        ids_to_delete: list[int] = []
        last: dict[str, Any] | None = None
        kwargs: dict[str, Any] = {
            "FilterExpression": (
                "id > :zero AND attribute_exists(#t)"
                + (" AND #ns = :ns" if namespace is not None else "")
            ),
            "ExpressionAttributeNames": {"#t": "ttl"},
            "ExpressionAttributeValues": {":zero": _to_dec(0)},
            "ProjectionExpression": "id, created_at, #t",
        }
        if namespace is not None:
            kwargs["ExpressionAttributeNames"]["#ns"] = "namespace"
            kwargs["ExpressionAttributeValues"][":ns"] = namespace
        while True:
            if last is not None:
                kwargs["ExclusiveStartKey"] = last
            resp = self._table_or_fail().scan(**kwargs)
            for item in resp.get("Items", []):
                created = _to_int(item.get("created_at", 0))
                ttl = _to_int(item.get("ttl", 0))
                if created + ttl <= now:
                    ids_to_delete.append(_to_int(item["id"]))
            last = resp.get("LastEvaluatedKey")
            if last is None:
                break
        deleted = 0
        for row_id in ids_to_delete:
            if self.delete_by_id(row_id):
                deleted += 1
        return deleted

    def clear_namespace(self, namespace: str) -> int:
        ids_to_delete: list[int] = []
        last: dict[str, Any] | None = None
        kwargs: dict[str, Any] = {
            "IndexName": _GSI_HASH,
            "KeyConditionExpression": "#ns = :ns",
            "ExpressionAttributeNames": {"#ns": "namespace"},
            "ExpressionAttributeValues": {":ns": namespace},
            "ProjectionExpression": "id",
        }
        while True:
            if last is not None:
                kwargs["ExclusiveStartKey"] = last
            resp = self._table_or_fail().query(**kwargs)
            for item in resp.get("Items", []):
                ids_to_delete.append(_to_int(item["id"]))
            last = resp.get("LastEvaluatedKey")
            if last is None:
                break
        deleted = 0
        for row_id in ids_to_delete:
            if self.delete_by_id(row_id):
                deleted += 1
        return deleted

    # --- Quotas ---

    def set_namespace_quota(self, namespace: str, max_entries: int) -> None:
        try:
            self._table_or_fail().update_item(
                Key={"id": _to_dec(_COUNTER_ID)},
                UpdateExpression="SET quotas.#ns = :v",
                ExpressionAttributeNames={"#ns": namespace},
                ExpressionAttributeValues={":v": _to_dec(max_entries)},
            )
        except Exception as exc:
            raise StoreBackendError(
                f"DynamoDB set_namespace_quota failed: {exc}. Remediation: see __cause__."
            ) from exc

    def get_namespace_quota(self, namespace: str) -> int | None:
        item = self._read_counter()
        quotas = item.get("quotas", {})
        if namespace not in quotas:
            return None
        return _to_int(quotas[namespace])

    # --- Coordination ---

    def read_version_counter(self) -> int:
        item = self._read_counter()
        return _to_int(item.get("version_counter", 0))

    def read_meta(self, key: str) -> str | None:
        item = self._read_counter()
        meta = item.get("meta", {})
        if key not in meta:
            return None
        return str(meta[key])

    def write_meta(self, key: str, value: str) -> None:
        try:
            self._table_or_fail().update_item(
                Key={"id": _to_dec(_COUNTER_ID)},
                UpdateExpression="SET meta.#k = :v",
                ExpressionAttributeNames={"#k": key},
                ExpressionAttributeValues={":v": value},
            )
        except Exception as exc:
            raise StoreBackendError(
                f"DynamoDB write_meta failed: {exc}. Remediation: see __cause__."
            ) from exc

    # --- Health ---

    def integrity_check(self) -> bool:
        try:
            self._client_or_fail().describe_table(TableName=self._table_name)
        except Exception:
            return False
        return True

    # --- Backup ---

    def snapshot_to(self, dest_path: str | Path) -> None:
        del dest_path
        raise CheckpointError(
            "DynamoDBStore.snapshot_to is not implemented in v1; the library "
            "does not bundle DynamoDB backup tooling. Remediation: use AWS "
            "on-demand backup or Point-in-Time Recovery, or scan-and-export "
            "to S3 via ExportTableToPointInTime."
        )

    @classmethod
    def restore_from(cls, source_path: str | Path, dest_path: str | Path) -> DynamoDBStore:
        del source_path, dest_path
        raise CheckpointError(
            "DynamoDBStore.restore_from is not implemented in v1. Remediation: "
            "use AWS native restore (RestoreTableFromBackup or "
            "RestoreTableToPointInTime), then construct DynamoDBStore pointing "
            "at the restored table."
        )


def _to_dynamodb_item(item: dict[str, Any]) -> dict[str, Any]:
    """Convert a resource-style item (Decimal/Binary/str/dict) to the
    low-level client-style item dict required by ``transact_write_items``.

    The resource API auto-serializes; the low-level client API does not.
    Keeping this conversion local avoids bringing in
    ``boto3.dynamodb.types.TypeSerializer`` at module import time (still
    behind the lazy `_import_boto3` boundary).
    """
    from boto3.dynamodb.types import TypeSerializer

    serializer = TypeSerializer()
    return {k: serializer.serialize(v) for k, v in item.items()}


__all__ = ["DynamoDBStore"]
