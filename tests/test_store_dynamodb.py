"""Backend-specific tests for ``DynamoDBStore``.

These tests use moto (in-process AWS mock) so no Docker / real AWS account
is required. The full ``Store`` protocol surface is exercised against
DynamoDB via the conformance battery in ``test_store_protocol_compliance.py``
under the ``DynamoDBStore[moto]`` and ``DynamoDBStore[local]`` parameters.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mneme._exceptions import (
    CacheClosedError,
    CheckpointError,
    EmbedderDimensionError,
    EmbedderMismatchError,
    StoreBackendError,
)
from mneme._types import StoredEntry

# Set fake AWS creds at module load so any boto3 import that reaches for
# config doesn't error out before moto's mock takes over.
for _var, _val in [
    ("AWS_ACCESS_KEY_ID", "testing"),
    ("AWS_SECRET_ACCESS_KEY", "testing"),
    ("AWS_SESSION_TOKEN", "testing"),
    ("AWS_DEFAULT_REGION", "us-east-1"),
]:
    os.environ.setdefault(_var, _val)


pytest.importorskip("moto", reason="moto extra not installed")
from moto import mock_aws  # noqa: E402


@pytest.fixture
def table_name() -> str:
    return f"mneme_test_{uuid.uuid4().hex[:12]}"


def _make_entry(
    *,
    namespace: str = "default",
    query_hash: str | None = None,
    query: str = "hi",
    response: str = "bye",
    metadata: dict[str, Any] | None = None,
    created_at: int = 1_700_000_000,
    last_accessed_at: int | None = None,
    ttl: int | None = None,
    access_count: int = 0,
) -> StoredEntry:
    if query_hash is None:
        query_hash = uuid.uuid4().hex * 2  # 64 chars
    if last_accessed_at is None:
        last_accessed_at = created_at
    return StoredEntry(
        id=0,
        namespace=namespace,
        query_hash=query_hash,
        query=query,
        response=response,
        embedding=np.zeros(4, dtype=np.float32).tobytes(),
        metadata=metadata if metadata is not None else {},
        created_at=created_at,
        last_accessed_at=last_accessed_at,
        ttl=ttl,
        access_count=access_count,
    )


# --- Constructor validation ---


def test_constructor_rejects_empty_table_name() -> None:
    from mneme._store_dynamodb import DynamoDBStore

    with pytest.raises(ValueError, match="non-empty"):
        DynamoDBStore(table_name="")


def test_constructor_requires_capacity_when_provisioned() -> None:
    from mneme._store_dynamodb import DynamoDBStore

    with pytest.raises(ValueError, match="provisioned_capacity"):
        DynamoDBStore(table_name="t", billing_mode="PROVISIONED")


def test_constructor_rejects_capacity_when_pay_per_request() -> None:
    from mneme._store_dynamodb import DynamoDBStore

    with pytest.raises(ValueError, match="only valid"):
        DynamoDBStore(
            table_name="t",
            billing_mode="PAY_PER_REQUEST",
            provisioned_capacity=(5, 5),
        )


# --- Auto-create vs missing table ---


def test_open_raises_when_table_missing_and_create_false(table_name: str) -> None:
    from mneme._store_dynamodb import DynamoDBStore

    with mock_aws():
        store = DynamoDBStore(
            table_name=table_name,
            region_name="us-east-1",
            create_table=False,
        )
        with pytest.raises(StoreBackendError, match="does not exist"):
            store.open(embedder_fingerprint="fp:v1", embedder_dim=4)


def test_open_auto_creates_when_create_true(table_name: str) -> None:
    from mneme._store_dynamodb import DynamoDBStore

    with mock_aws():
        store = DynamoDBStore(
            table_name=table_name,
            region_name="us-east-1",
            create_table=True,
        )
        store.open(embedder_fingerprint="fp:v1", embedder_dim=4)
        assert store.read_version_counter() == 0
        store.close()


def test_open_auto_creates_with_provisioned_billing(table_name: str) -> None:
    from mneme._store_dynamodb import DynamoDBStore

    with mock_aws():
        store = DynamoDBStore(
            table_name=table_name,
            region_name="us-east-1",
            create_table=True,
            billing_mode="PROVISIONED",
            provisioned_capacity=(5, 5),
        )
        store.open(embedder_fingerprint="fp:v1", embedder_dim=4)
        store.close()


# --- Fingerprint / dim validation on reopen ---


def test_reopen_with_different_fingerprint_raises(table_name: str) -> None:
    from mneme._store_dynamodb import DynamoDBStore

    with mock_aws():
        s1 = DynamoDBStore(
            table_name=table_name,
            region_name="us-east-1",
            create_table=True,
        )
        s1.open(embedder_fingerprint="fp:v1", embedder_dim=4)
        s1.close()

        s2 = DynamoDBStore(
            table_name=table_name,
            region_name="us-east-1",
            create_table=False,
        )
        with pytest.raises(EmbedderMismatchError):
            s2.open(embedder_fingerprint="fp:v2", embedder_dim=4)


def test_reopen_with_different_dim_raises(table_name: str) -> None:
    from mneme._store_dynamodb import DynamoDBStore

    with mock_aws():
        s1 = DynamoDBStore(
            table_name=table_name,
            region_name="us-east-1",
            create_table=True,
        )
        s1.open(embedder_fingerprint="fp:v1", embedder_dim=4)
        s1.close()

        s2 = DynamoDBStore(
            table_name=table_name,
            region_name="us-east-1",
            create_table=False,
        )
        with pytest.raises(EmbedderDimensionError):
            s2.open(embedder_fingerprint="fp:v1", embedder_dim=8)


# --- Closed state guard ---


def test_method_after_close_raises_cache_closed(table_name: str) -> None:
    from mneme._store_dynamodb import DynamoDBStore

    with mock_aws():
        store = DynamoDBStore(
            table_name=table_name,
            region_name="us-east-1",
            create_table=True,
        )
        store.open(embedder_fingerprint="fp:v1", embedder_dim=4)
        store.close()
        with pytest.raises(CacheClosedError):
            store.get_by_hash("default", "x")


def test_reopen_after_close_raises_cache_closed(table_name: str) -> None:
    from mneme._store_dynamodb import DynamoDBStore

    with mock_aws():
        store = DynamoDBStore(
            table_name=table_name,
            region_name="us-east-1",
            create_table=True,
        )
        store.open(embedder_fingerprint="fp:v1", embedder_dim=4)
        store.close()
        with pytest.raises(CacheClosedError):
            store.open(embedder_fingerprint="fp:v1", embedder_dim=4)


# --- Snapshot/restore stubs ---


def test_snapshot_to_raises_checkpoint_error(table_name: str, tmp_path: Path) -> None:
    from mneme._store_dynamodb import DynamoDBStore

    with mock_aws():
        store = DynamoDBStore(
            table_name=table_name,
            region_name="us-east-1",
            create_table=True,
        )
        store.open(embedder_fingerprint="fp:v1", embedder_dim=4)
        try:
            with pytest.raises(CheckpointError, match="not implemented"):
                store.snapshot_to(tmp_path / "snap.tar.gz")
        finally:
            store.close()


def test_restore_from_raises_checkpoint_error(tmp_path: Path) -> None:
    from mneme._store_dynamodb import DynamoDBStore

    with pytest.raises(CheckpointError, match="not implemented"):
        DynamoDBStore.restore_from(tmp_path / "src.tar.gz", tmp_path / "dst.tar.gz")


# --- Insert + version_counter atomicity ---


def test_insert_bumps_version_counter(table_name: str) -> None:
    from mneme._store_dynamodb import DynamoDBStore

    with mock_aws():
        store = DynamoDBStore(
            table_name=table_name,
            region_name="us-east-1",
            create_table=True,
        )
        store.open(embedder_fingerprint="fp:v1", embedder_dim=4)
        try:
            before = store.read_version_counter()
            new_id = store.insert(_make_entry(query_hash="a" * 64))
            after = store.read_version_counter()
            assert new_id >= 1
            assert after == before + 1
        finally:
            store.close()


def test_insert_assigns_monotonic_ids(table_name: str) -> None:
    from mneme._store_dynamodb import DynamoDBStore

    with mock_aws():
        store = DynamoDBStore(
            table_name=table_name,
            region_name="us-east-1",
            create_table=True,
        )
        store.open(embedder_fingerprint="fp:v1", embedder_dim=4)
        try:
            ids = [store.insert(_make_entry(query_hash=f"{i:064d}")) for i in range(5)]
            assert ids == sorted(ids)
            assert len(set(ids)) == len(ids)
        finally:
            store.close()


def test_insert_upserts_on_hash_collision(table_name: str) -> None:
    from mneme._store_dynamodb import DynamoDBStore

    with mock_aws():
        store = DynamoDBStore(
            table_name=table_name,
            region_name="us-east-1",
            create_table=True,
        )
        store.open(embedder_fingerprint="fp:v1", embedder_dim=4)
        try:
            id1 = store.insert(_make_entry(query_hash="z" * 64, response="first"))
            id2 = store.insert(_make_entry(query_hash="z" * 64, response="second"))
            assert id1 == id2
            fetched = store.get_by_hash("default", "z" * 64)
            assert fetched is not None
            assert fetched.response == "second"
        finally:
            store.close()


# --- Lazy-import gate ---


def test_import_mneme_does_not_pull_boto3() -> None:
    """``import mneme`` must not pull in boto3 / DynamoDBStore — §30 invariant.

    Run in a subprocess so we observe a clean import graph without disturbing
    the parent's ``sys.modules`` (which would break class identity for any
    later test that uses pytest.raises against mneme exceptions).
    """
    import subprocess
    import sys

    code = (
        "import sys; import mneme; "
        "assert 'mneme._store_dynamodb' not in sys.modules, "
        "'DynamoDB module was eagerly imported'; "
        "assert 'boto3' not in sys.modules, 'boto3 was eagerly imported'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, (
        f"Lazy-import gate failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
