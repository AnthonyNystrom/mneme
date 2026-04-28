"""Phase-5 tests: HnswIndex correctness, namespace isolation, lifecycle.

Skipped if the optional ``[hnsw]`` extra is missing.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

# Skip the whole module if hnswlib is not installed.
hnswlib = pytest.importorskip("hnswlib")

from mneme._exceptions import (  # noqa: E402
    IndexBackendUnavailableError,
    QuantizationError,
)
from mneme._index_hnsw import HnswIndex  # noqa: E402
from mneme._types import Index  # noqa: E402


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n).astype(np.float32)


def _rand(rng: np.random.Generator, dim: int = 8) -> np.ndarray:
    return _l2(rng.standard_normal(dim).astype(np.float32))


# --- Protocol & construction ---


def test_hnsw_index_satisfies_protocol():
    idx = HnswIndex(dim=4)
    assert isinstance(idx, Index)


def test_dim_property():
    idx = HnswIndex(dim=128)
    assert idx.dim == 128


def test_dtype_is_float32():
    idx = HnswIndex(dim=4)
    assert idx.dtype == "float32"


def test_initial_size_zero():
    idx = HnswIndex(dim=4)
    assert idx.size == 0


def test_dim_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        HnswIndex(dim=0)


def test_only_float32_dtype_supported():
    with pytest.raises(QuantizationError, match="float32"):
        HnswIndex(dim=4, dtype="float16")
    with pytest.raises(QuantizationError, match="float32"):
        HnswIndex(dim=4, dtype="int8")


def test_index_options_are_applied():
    idx = HnswIndex(
        dim=4,
        index_options={"M": 32, "ef_construction": 400, "ef": 100},
    )
    assert idx._M == 32  # type: ignore[attr-defined]
    assert idx._ef_construction == 400  # type: ignore[attr-defined]
    assert idx._ef == 100  # type: ignore[attr-defined]


# --- Append / search correctness ---


def test_search_finds_exact_match():
    idx = HnswIndex(dim=8)
    target = _l2(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    idx.append(1, target, "default")
    rng = np.random.default_rng(0)
    for i in range(10):
        idx.append(i + 2, _rand(rng, 8), "default")
    hits = idx.search(target, "default", k=1)
    assert len(hits) == 1
    assert hits[0][0] == 1
    assert hits[0][1] == pytest.approx(1.0, abs=1e-3)


def test_search_returns_top_k_in_descending_score_order():
    idx = HnswIndex(dim=8)
    target = _l2(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    for i, scale in enumerate([0.99, 0.95, 0.90, 0.80, 0.70]):
        v = _l2(np.array([scale, np.sqrt(1 - scale**2), 0, 0, 0, 0, 0, 0], dtype=np.float32))
        idx.append(i + 1, v, "default")
    hits = idx.search(target, "default", k=3)
    assert len(hits) == 3
    for i in range(len(hits) - 1):
        assert hits[i][1] >= hits[i + 1][1]


def test_search_k_exceeds_size_returns_all_live():
    idx = HnswIndex(dim=8)
    rng = np.random.default_rng(0)
    for i in range(3):
        idx.append(i + 1, _rand(rng, 8), "default")
    hits = idx.search(_rand(rng, 8), "default", k=100)
    assert len(hits) == 3


def test_search_with_zero_k_returns_empty():
    idx = HnswIndex(dim=4)
    rng = np.random.default_rng(0)
    idx.append(1, _rand(rng, 4), "default")
    assert idx.search(_rand(rng, 4), "default", k=0) == []


def test_search_unknown_namespace_returns_empty():
    idx = HnswIndex(dim=4)
    rng = np.random.default_rng(0)
    idx.append(1, _rand(rng, 4), "default")
    assert idx.search(_rand(rng, 4), "other", k=5) == []


def test_search_with_wrong_dim_raises():
    idx = HnswIndex(dim=4)
    rng = np.random.default_rng(0)
    idx.append(1, _rand(rng, 4), "default")
    with pytest.raises(ValueError, match="shape"):
        idx.search(np.zeros(8, dtype=np.float32), "default", k=1)


def test_append_with_wrong_dim_raises():
    idx = HnswIndex(dim=4)
    with pytest.raises(ValueError, match="shape"):
        idx.append(1, np.zeros(8, dtype=np.float32), "default")


# --- Namespace isolation ---


def test_namespace_isolation():
    idx = HnswIndex(dim=4)
    a = _l2(np.array([1, 0, 0, 0], dtype=np.float32))
    idx.append(1, a, "tenant_a")
    idx.append(2, a, "tenant_b")
    hits_a = idx.search(a, "tenant_a", k=10)
    hits_b = idx.search(a, "tenant_b", k=10)
    assert {row_id for row_id, _ in hits_a} == {1}
    assert {row_id for row_id, _ in hits_b} == {2}


# --- Remove / soft-delete ---


def test_remove_excludes_from_search():
    idx = HnswIndex(dim=4)
    target = _l2(np.array([1, 0, 0, 0], dtype=np.float32))
    idx.append(1, target, "default")
    idx.append(2, _l2(np.array([0, 1, 0, 0], dtype=np.float32)), "default")
    idx.remove(1)
    hits = idx.search(target, "default", k=10)
    assert 1 not in {row_id for row_id, _ in hits}


def test_remove_nonexistent_is_noop():
    idx = HnswIndex(dim=4)
    idx.remove(99999)


def test_remove_decreases_size():
    idx = HnswIndex(dim=4)
    rng = np.random.default_rng(0)
    idx.append(1, _rand(rng, 4), "default")
    idx.append(2, _rand(rng, 4), "default")
    assert idx.size == 2
    idx.remove(1)
    assert idx.size == 1


# --- Rebuild + compact ---


def test_rebuild_from_replaces_state():
    idx = HnswIndex(dim=4)
    rng = np.random.default_rng(0)
    idx.append(1, _rand(rng, 4), "old")
    idx.append(2, _rand(rng, 4), "old")
    idx.rebuild_from(
        [
            (10, _rand(rng, 4), "new"),
            (11, _rand(rng, 4), "new"),
        ]
    )
    assert idx.size == 2
    assert idx.search(_rand(rng, 4), "old", k=5) == []
    hits = idx.search(_rand(rng, 4), "new", k=5)
    assert {row_id for row_id, _ in hits} == {10, 11}


def test_compact_no_tombstones_is_noop():
    idx = HnswIndex(dim=4)
    rng = np.random.default_rng(0)
    idx.append(1, _rand(rng, 4), "default")
    idx.compact()
    assert idx.size == 1


def test_compact_drops_tombstones_after_removes():
    idx = HnswIndex(dim=4)
    rng = np.random.default_rng(0)
    target = _l2(np.array([1, 0, 0, 0], dtype=np.float32))
    idx.append(1, _rand(rng, 4), "default")
    idx.append(2, target, "default")
    idx.append(3, _rand(rng, 4), "default")
    idx.remove(1)
    idx.remove(3)
    pre_size = idx.size
    idx.compact()
    assert idx.size == pre_size  # live count unchanged
    hits = idx.search(target, "default", k=10)
    assert {row_id for row_id, _ in hits} == {2}


# --- Capacity growth ---


def test_index_grows_past_initial_max_elements():
    idx = HnswIndex(dim=4, index_options={"initial_max_elements": 2})
    rng = np.random.default_rng(0)
    for i in range(20):
        idx.append(i + 1, _rand(rng, 4), "default")
    assert idx.size == 20


# --- Requantize ---


def test_requantize_to_same_dtype_is_noop():
    idx = HnswIndex(dim=4)
    idx.requantize("float32")  # must not raise


def test_requantize_to_other_dtype_raises():
    idx = HnswIndex(dim=4)
    with pytest.raises(QuantizationError, match="NumpyIndex"):
        idx.requantize("int8")
    with pytest.raises(QuantizationError, match="NumpyIndex"):
        idx.requantize("float16")


# --- Lazy import guard ---


def test_missing_hnswlib_raises_index_backend_unavailable():
    """Even with hnswlib installed, simulate the missing-extra path."""
    with (
        patch(
            "mneme._index_hnsw._import_hnswlib",
            side_effect=IndexBackendUnavailableError(
                "hnsw extra missing. Remediation: pip install mneme[hnsw]"
            ),
        ),
        pytest.raises(IndexBackendUnavailableError, match="hnsw"),
    ):
        HnswIndex(dim=4)


# --- Scale check (small) ---


def test_search_correctness_at_dim_768_with_distractors():
    """End-to-end correctness at a realistic embedding dimension."""
    idx = HnswIndex(dim=768)
    rng = np.random.default_rng(42)
    target = _l2(rng.standard_normal(768).astype(np.float32))
    for i in range(200):
        idx.append(i + 1, _l2(rng.standard_normal(768).astype(np.float32)), "default")
    idx.append(999, target, "default")
    hits = idx.search(target, "default", k=1)
    assert hits[0][0] == 999
    assert hits[0][1] == pytest.approx(1.0, abs=1e-3)


# --- Cross-namespace id move ---


def test_append_same_id_moves_between_namespaces():
    idx = HnswIndex(dim=4)
    rng = np.random.default_rng(0)
    v = _rand(rng, 4)
    idx.append(1, v, "tenant_a")
    # Same id, different namespace: prior entry must be soft-deleted in tenant_a.
    idx.append(1, _rand(rng, 4), "tenant_b")
    assert 1 not in {row_id for row_id, _ in idx.search(v, "tenant_a", k=10)}
    assert 1 in {row_id for row_id, _ in idx.search(_rand(rng, 4), "tenant_b", k=10)}


# --- Revive after remove ---


def test_remove_then_append_revives_with_new_vector():
    idx = HnswIndex(dim=4)
    target = _l2(np.array([1, 0, 0, 0], dtype=np.float32))
    other = _l2(np.array([0, 1, 0, 0], dtype=np.float32))
    idx.append(1, target, "default")
    idx.remove(1)
    # Same namespace + same id: revive (unmark_deleted + replace vector).
    idx.append(1, other, "default")
    hits = idx.search(other, "default", k=10)
    assert 1 in {row_id for row_id, _ in hits}


def test_search_returns_empty_when_all_namespace_entries_deleted():
    idx = HnswIndex(dim=4)
    rng = np.random.default_rng(0)
    idx.append(1, _rand(rng, 4), "default")
    idx.append(2, _rand(rng, 4), "default")
    idx.remove(1)
    idx.remove(2)
    assert idx.search(_rand(rng, 4), "default", k=5) == []


def test_remove_after_namespace_already_dropped_is_safe():
    idx = HnswIndex(dim=4)
    rng = np.random.default_rng(0)
    idx.append(1, _rand(rng, 4), "default")
    # Simulate manual cleanup of the namespace's underlying index.
    idx._indexes.pop("default", None)  # type: ignore[attr-defined]
    idx.remove(1)  # must not raise
