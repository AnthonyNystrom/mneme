"""Phase-1 tests: dataclasses, protocols, type aliases, exception hierarchy."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, fields, is_dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pytest

from mneme._exceptions import (
    CacheClosedError,
    CheckpointError,
    CorruptCacheError,
    EmbedderDimensionError,
    EmbedderMismatchError,
    IndexBackendUnavailableError,
    MnemeError,
    MultiProcessLockError,
    NamespaceQuotaExceededError,
    QuantizationError,
    SchemaMigrationError,
    StoreBackendError,
)
from mneme._types import (
    AsyncEmbedder,
    ConfidenceFn,
    Embedder,
    Health,
    Hit,
    Index,
    MetricsHook,
    Stats,
    Store,
    StoredEntry,
    Validator,
)

# --- Dataclass shape ---


def _sample_hit() -> Hit:
    return Hit(
        response="hello",
        similarity=0.91,
        confidence=0.87,
        age_seconds=42,
        layer="exact",
        namespace="default",
        metadata={"source": "unit"},
    )


def _sample_stats() -> Stats:
    return Stats(
        namespace="default",
        entries=10,
        hits_exact=3,
        hits_semantic=2,
        misses=1,
        evictions=0,
        expirations=0,
        embedder_fingerprint="fake:embed:v1",
        vector_dtype="float32",
        memory_bytes_estimate=12_345,
    )


def _sample_health() -> Health:
    return Health(
        healthy=True,
        schema_version=1,
        integrity_ok=True,
        embedder_fingerprint_match=True,
        entries=10,
        namespaces=2,
        oldest_entry_age_seconds=100,
        index_backend="numpy",
        store_backend="sqlite",
        vector_dtype="float32",
        multi_process_mode="single",
    )


def _sample_stored_entry() -> StoredEntry:
    return StoredEntry(
        id=1,
        namespace="default",
        query_hash="a" * 64,
        query="hi",
        response="hello",
        embedding=b"\x00" * 16,
        metadata={"k": "v"},
        created_at=1_700_000_000,
        last_accessed_at=1_700_000_001,
        ttl=86_400,
        access_count=2,
    )


@pytest.mark.parametrize(
    ("cls", "factory"),
    [
        (Hit, _sample_hit),
        (Stats, _sample_stats),
        (Health, _sample_health),
        (StoredEntry, _sample_stored_entry),
    ],
)
def test_public_dataclasses_are_dataclasses(cls, factory):
    assert is_dataclass(cls)
    inst = factory()
    assert is_dataclass(inst)


@pytest.mark.parametrize(
    ("cls", "factory"),
    [
        (Hit, _sample_hit),
        (Stats, _sample_stats),
        (Health, _sample_health),
        (StoredEntry, _sample_stored_entry),
    ],
)
def test_public_dataclasses_are_frozen(cls, factory):
    inst = factory()
    first_field = fields(cls)[0].name
    with pytest.raises((FrozenInstanceError, AttributeError)):
        setattr(inst, first_field, "tampered")


@pytest.mark.parametrize(
    ("cls", "factory"),
    [
        (Hit, _sample_hit),
        (Stats, _sample_stats),
        (Health, _sample_health),
        (StoredEntry, _sample_stored_entry),
    ],
)
def test_public_dataclasses_are_slotted(cls, factory):
    # __slots__ presence on the class is the source of truth for slots=True.
    assert hasattr(cls, "__slots__")
    inst = factory()
    # Assigning a non-field attribute must fail. Python 3.12 raises TypeError
    # for frozen+slots dataclasses (cpython quirk); other versions raise
    # AttributeError / FrozenInstanceError. Accept either family.
    with pytest.raises((AttributeError, TypeError)):
        inst.this_is_not_a_field = "nope"  # type: ignore[attr-defined]


def test_hit_field_names():
    expected = {
        "response",
        "similarity",
        "confidence",
        "age_seconds",
        "layer",
        "namespace",
        "metadata",
    }
    assert {f.name for f in fields(Hit)} == expected


def test_hit_layer_accepts_documented_values():
    for layer in ("exact", "semantic"):
        Hit(
            response="r",
            similarity=1.0,
            confidence=1.0,
            age_seconds=0,
            layer=layer,
            namespace="ns",
            metadata={},
        )


def test_stats_field_names():
    expected = {
        "namespace",
        "entries",
        "hits_exact",
        "hits_semantic",
        "misses",
        "evictions",
        "expirations",
        "embedder_fingerprint",
        "vector_dtype",
        "memory_bytes_estimate",
        "index_memory_bytes",
        "index_tombstone_count",
    }
    assert {f.name for f in fields(Stats)} == expected


def test_health_field_names():
    expected = {
        "healthy",
        "schema_version",
        "integrity_ok",
        "embedder_fingerprint_match",
        "entries",
        "namespaces",
        "oldest_entry_age_seconds",
        "index_backend",
        "store_backend",
        "vector_dtype",
        "multi_process_mode",
    }
    assert {f.name for f in fields(Health)} == expected


def test_stored_entry_field_names():
    expected = {
        "id",
        "namespace",
        "query_hash",
        "query",
        "response",
        "embedding",
        "metadata",
        "created_at",
        "last_accessed_at",
        "ttl",
        "access_count",
    }
    assert {f.name for f in fields(StoredEntry)} == expected


# --- Type aliases ---


def test_confidence_fn_alias_callable():
    def cf(sim: float, age: int, meta: dict[str, Any]) -> float:
        return sim * 0.9

    fn: ConfidenceFn = cf
    assert fn(0.9, 100, {}) == pytest.approx(0.81)


def test_validator_alias_callable():
    def v(r: str) -> bool:
        return bool(r and r.strip())

    fn: Validator = v
    assert fn("hello") is True
    assert fn("") is False


def test_metrics_hook_alias_callable():
    received: list[tuple[str, dict[str, Any]]] = []

    def hook(event: str, attrs: dict[str, Any]) -> None:
        received.append((event, attrs))

    fn: MetricsHook = hook
    fn("hit", {"layer": "exact"})
    assert received == [("hit", {"layer": "exact"})]


# --- Embedder protocols ---


class _FakeSyncEmbedder:
    @property
    def dim(self) -> int:
        return 4

    @property
    def fingerprint(self) -> str:
        return "fake:sync:v1"

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(self.dim).astype(np.float32)
        v /= np.linalg.norm(v)
        return v


class _FakeAsyncEmbedder:
    @property
    def dim(self) -> int:
        return 4

    @property
    def fingerprint(self) -> str:
        return "fake:async:v1"

    async def embed(self, text: str) -> npt.NDArray[np.float32]:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(self.dim).astype(np.float32)
        v /= np.linalg.norm(v)
        return v


def test_sync_embedder_protocol_conformance():
    assert isinstance(_FakeSyncEmbedder(), Embedder)


def test_async_embedder_protocol_conformance():
    assert isinstance(_FakeAsyncEmbedder(), AsyncEmbedder)


async def test_async_embedder_returns_correct_shape():
    e = _FakeAsyncEmbedder()
    v = await e.embed("hello")
    assert v.shape == (4,)
    assert v.dtype == np.float32


def test_sync_embedder_returns_correct_shape():
    e = _FakeSyncEmbedder()
    v = e.embed("hello")
    assert v.shape == (4,)
    assert v.dtype == np.float32


def test_sync_embed_is_not_coroutine():
    e = _FakeSyncEmbedder()
    result = e.embed("x")
    assert not asyncio.iscoroutine(result)


# --- Index protocol ---


class _NoopIndex:
    @property
    def dim(self) -> int:
        return 4

    @property
    def size(self) -> int:
        return 0

    @property
    def dtype(self) -> Literal["float32", "float16", "int8"]:
        return "float32"

    def append(self, row_id: int, vec: npt.NDArray[Any], namespace: str) -> None:
        del row_id, vec, namespace

    def remove(self, row_id: int) -> None:
        del row_id

    def search(
        self, query: npt.NDArray[Any], namespace: str, *, k: int = 1
    ) -> list[tuple[int, float]]:
        del query, namespace, k
        return []

    def rebuild_from(self, rows: Any) -> None:
        del rows

    def compact(self) -> None:
        return None

    def requantize(self, dtype: Literal["float32", "float16", "int8"]) -> None:
        del dtype


def test_index_protocol_conformance():
    assert isinstance(_NoopIndex(), Index)


# --- Store protocol ---


class _NoopStore:
    def open(self, embedder_fingerprint: str, embedder_dim: int) -> None:
        del embedder_fingerprint, embedder_dim

    def close(self) -> None:
        return None

    def get_by_hash(self, namespace: str, query_hash: str) -> StoredEntry | None:
        del namespace, query_hash
        return None

    def get_by_id(self, id: int) -> StoredEntry | None:
        del id
        return None

    def count(self, namespace: str | None = None) -> int:
        del namespace
        return 0

    def list_namespaces(self) -> list[str]:
        return []

    def iter_lru_ids(self, n: int, namespace: str | None = None) -> Any:
        del n, namespace
        return iter([])

    def iter_all(self) -> Any:
        return iter([])

    def iter_since(self, last_id: int) -> Any:
        del last_id
        return iter([])

    def insert(self, entry: StoredEntry) -> int:
        del entry
        return 1

    def update_access(self, id: int, now: int) -> None:
        del id, now

    def delete_by_id(self, id: int) -> bool:
        del id
        return True

    def delete_expired(self, now: int, namespace: str | None = None) -> int:
        del now, namespace
        return 0

    def clear_namespace(self, namespace: str) -> int:
        del namespace
        return 0

    def set_namespace_quota(self, namespace: str, max_entries: int) -> None:
        del namespace, max_entries

    def get_namespace_quota(self, namespace: str) -> int | None:
        del namespace
        return None

    def read_version_counter(self) -> int:
        return 0

    def read_meta(self, key: str) -> str | None:
        del key
        return None

    def write_meta(self, key: str, value: str) -> None:
        del key, value

    def integrity_check(self) -> bool:
        return True

    def snapshot_to(self, dest_path: Any) -> None:
        del dest_path

    @classmethod
    def restore_from(cls, source_path: Any, dest_path: Any) -> _NoopStore:
        del source_path, dest_path
        return cls()


def test_store_protocol_conformance():
    assert isinstance(_NoopStore(), Store)


def test_store_protocol_classmethod_present():
    """Verify restore_from is callable as a classmethod."""
    inst = _NoopStore.restore_from("/tmp/src", "/tmp/dst")
    assert isinstance(inst, _NoopStore)


# --- Exception hierarchy ---


_PUBLIC_EXCEPTIONS = [
    EmbedderMismatchError,
    EmbedderDimensionError,
    CorruptCacheError,
    SchemaMigrationError,
    CacheClosedError,
    CheckpointError,
    IndexBackendUnavailableError,
    StoreBackendError,
    NamespaceQuotaExceededError,
    MultiProcessLockError,
    QuantizationError,
]


@pytest.mark.parametrize("exc", _PUBLIC_EXCEPTIONS)
def test_public_exceptions_inherit_mneme_error(exc):
    assert issubclass(exc, MnemeError)
    assert issubclass(exc, Exception)


@pytest.mark.parametrize("exc", _PUBLIC_EXCEPTIONS)
def test_public_exceptions_carry_message(exc):
    e = exc("remediation: do thing X")
    assert "remediation" in str(e)


def test_mneme_error_inherits_exception():
    assert issubclass(MnemeError, Exception)
    assert not issubclass(MnemeError, BaseException) or issubclass(MnemeError, Exception)


def test_exceptions_are_raisable():
    for exc in _PUBLIC_EXCEPTIONS:
        with pytest.raises(exc):
            raise exc("boom")
        with pytest.raises(MnemeError):
            raise exc("boom")


def test_exception_docstrings_mention_remediation():
    """Each public exception class documents its remediation in its docstring."""
    for exc in _PUBLIC_EXCEPTIONS:
        assert exc.__doc__ is not None
        assert "remediation" in exc.__doc__.lower(), (
            f"{exc.__name__} docstring missing remediation guidance"
        )
