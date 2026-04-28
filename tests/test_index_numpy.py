"""Phase-4 tests: NumpyIndex behavior across dtypes and namespaces."""

from __future__ import annotations

import numpy as np
import pytest

from mneme._index import NumpyIndex
from mneme._types import Index


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n).astype(np.float32)


def _rand(rng: np.random.Generator, dim: int = 8) -> np.ndarray:
    return _l2(rng.standard_normal(dim).astype(np.float32))


# --- Protocol conformance ---


def test_numpy_index_satisfies_protocol():
    idx = NumpyIndex(dim=4)
    assert isinstance(idx, Index)


def test_dim_property():
    idx = NumpyIndex(dim=128)
    assert idx.dim == 128


def test_initial_size_zero():
    idx = NumpyIndex(dim=4)
    assert idx.size == 0


def test_dim_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        NumpyIndex(dim=0)
    with pytest.raises(ValueError, match="positive"):
        NumpyIndex(dim=-1)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_dtype_property_matches_constructor(dtype):
    idx = NumpyIndex(dim=4, dtype=dtype)
    assert idx.dtype == dtype


# --- Append / size ---


def test_append_grows_size():
    idx = NumpyIndex(dim=4)
    rng = np.random.default_rng(0)
    idx.append(1, _rand(rng, 4), "default")
    idx.append(2, _rand(rng, 4), "default")
    assert idx.size == 2


def test_append_grows_capacity():
    idx = NumpyIndex(dim=4, initial_capacity=2)
    rng = np.random.default_rng(0)
    for i in range(10):
        idx.append(i + 1, _rand(rng, 4), "default")
    assert idx.size == 10


def test_append_with_wrong_dim_raises():
    idx = NumpyIndex(dim=4)
    bad = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        idx.append(1, bad, "default")


def test_append_same_id_updates_in_place():
    idx = NumpyIndex(dim=4)
    rng = np.random.default_rng(0)
    v1 = _rand(rng, 4)
    v2 = _rand(rng, 4)
    idx.append(1, v1, "default")
    idx.append(1, v2, "default")
    assert idx.size == 1


# --- Search correctness ---


def test_cosine_similarity_correctness():
    idx = NumpyIndex(dim=4)
    # Build orthogonal-ish vectors
    a = _l2(np.array([1, 0, 0, 0], dtype=np.float32))
    b = _l2(np.array([0, 1, 0, 0], dtype=np.float32))
    c = _l2(np.array([0.9, 0.1, 0, 0], dtype=np.float32))
    idx.append(1, a, "default")
    idx.append(2, b, "default")
    idx.append(3, c, "default")
    hits = idx.search(a, "default", k=1)
    assert hits[0][0] == 1
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    # c is closer to a than b is
    hits = idx.search(a, "default", k=2)
    assert hits[0][0] == 1  # a == a
    assert hits[1][0] == 3  # c closer than b


def test_search_k_top_results_in_sorted_order():
    idx = NumpyIndex(dim=4)
    target = _l2(np.array([1, 0, 0, 0], dtype=np.float32))
    for i, scale in enumerate([0.99, 0.95, 0.90, 0.80, 0.70]):
        v = _l2(np.array([scale, np.sqrt(1 - scale**2), 0, 0], dtype=np.float32))
        idx.append(i + 1, v, "default")
    hits = idx.search(target, "default", k=3)
    assert len(hits) == 3
    # Scores in descending order
    for i in range(len(hits) - 1):
        assert hits[i][1] >= hits[i + 1][1]


def test_search_k_exceeds_size_returns_all():
    idx = NumpyIndex(dim=4)
    rng = np.random.default_rng(0)
    idx.append(1, _rand(rng, 4), "default")
    idx.append(2, _rand(rng, 4), "default")
    hits = idx.search(_rand(rng, 4), "default", k=10)
    assert len(hits) == 2


def test_search_with_zero_k_returns_empty():
    idx = NumpyIndex(dim=4)
    rng = np.random.default_rng(0)
    idx.append(1, _rand(rng, 4), "default")
    assert idx.search(_rand(rng, 4), "default", k=0) == []


def test_search_unknown_namespace_returns_empty():
    idx = NumpyIndex(dim=4)
    rng = np.random.default_rng(0)
    idx.append(1, _rand(rng, 4), "default")
    assert idx.search(_rand(rng, 4), "other", k=5) == []


def test_search_with_wrong_dim_raises():
    idx = NumpyIndex(dim=4)
    rng = np.random.default_rng(0)
    idx.append(1, _rand(rng, 4), "default")
    bad_q = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        idx.search(bad_q, "default", k=1)


# --- Namespace isolation ---


def test_namespace_isolation():
    idx = NumpyIndex(dim=4)
    a = _l2(np.array([1, 0, 0, 0], dtype=np.float32))
    idx.append(1, a, "tenant_a")
    idx.append(2, a, "tenant_b")
    hits_a = idx.search(a, "tenant_a", k=10)
    hits_b = idx.search(a, "tenant_b", k=10)
    assert {row_id for row_id, _ in hits_a} == {1}
    assert {row_id for row_id, _ in hits_b} == {2}


# --- Remove / tombstones ---


def test_remove_excludes_from_search():
    idx = NumpyIndex(dim=4)
    rng = np.random.default_rng(0)
    target = _l2(np.array([1, 0, 0, 0], dtype=np.float32))
    idx.append(1, target, "default")
    idx.append(2, _rand(rng, 4), "default")
    idx.remove(1)
    hits = idx.search(target, "default", k=10)
    assert 1 not in {row_id for row_id, _ in hits}


def test_remove_nonexistent_is_noop():
    idx = NumpyIndex(dim=4)
    idx.remove(99999)
    idx.remove(99999)


def test_remove_decreases_size():
    idx = NumpyIndex(dim=4)
    rng = np.random.default_rng(0)
    idx.append(1, _rand(rng, 4), "default")
    idx.append(2, _rand(rng, 4), "default")
    assert idx.size == 2
    idx.remove(1)
    assert idx.size == 1


def test_remove_then_append_revives_slot():
    idx = NumpyIndex(dim=4)
    rng = np.random.default_rng(0)
    v1 = _rand(rng, 4)
    v2 = _rand(rng, 4)
    idx.append(1, v1, "default")
    idx.remove(1)
    assert idx.size == 0
    idx.append(1, v2, "default")
    assert idx.size == 1


# --- Compaction ---


def test_compact_with_no_tombstones_is_noop():
    idx = NumpyIndex(dim=4)
    rng = np.random.default_rng(0)
    idx.append(1, _rand(rng, 4), "default")
    pre_matrix = idx._matrix.copy()  # type: ignore[attr-defined]
    idx.compact()
    np.testing.assert_array_equal(idx._matrix, pre_matrix)  # type: ignore[attr-defined]


def test_compact_drops_tombstones():
    idx = NumpyIndex(dim=4)
    rng = np.random.default_rng(0)
    target = _l2(np.array([1, 0, 0, 0], dtype=np.float32))
    idx.append(1, _rand(rng, 4), "default")
    idx.append(2, target, "default")
    idx.append(3, _rand(rng, 4), "default")
    idx.remove(1)
    idx.remove(3)
    assert idx.size == 1
    idx.compact()
    assert idx.size == 1
    # Search still works for the surviving row
    hits = idx.search(target, "default", k=10)
    assert {row_id for row_id, _ in hits} == {2}


# --- Rebuild ---


def test_rebuild_from_replaces_state():
    idx = NumpyIndex(dim=4)
    rng = np.random.default_rng(0)
    idx.append(1, _rand(rng, 4), "old")
    idx.append(2, _rand(rng, 4), "old")
    new_rows = [
        (10, _rand(rng, 4), "new"),
        (11, _rand(rng, 4), "new"),
        (12, _rand(rng, 4), "new"),
    ]
    idx.rebuild_from(new_rows)
    assert idx.size == 3
    assert idx.search(_rand(rng, 4), "old", k=5) == []
    hits = idx.search(_rand(rng, 4), "new", k=5)
    assert {row_id for row_id, _ in hits} == {10, 11, 12}


# --- Quantized search across dtypes ---


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_search_correctness_across_dtypes(dtype):
    """Top-1 hit on the exact query vector should still be itself."""
    idx = NumpyIndex(dim=64, dtype=dtype)
    rng = np.random.default_rng(0)
    target = _l2(rng.standard_normal(64).astype(np.float32))
    for i in range(20):
        v = _l2(rng.standard_normal(64).astype(np.float32))
        idx.append(i + 1, v, "default")
    idx.append(999, target, "default")
    hits = idx.search(target, "default", k=1)
    assert hits[0][0] == 999
    if dtype == "float32":
        assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    elif dtype == "float16":
        assert hits[0][1] == pytest.approx(1.0, abs=5e-3)
    else:  # int8
        assert hits[0][1] == pytest.approx(1.0, abs=2e-2)


def test_dtype_consistency_after_int8_storage():
    """The matrix dtype matches the chosen vector_dtype."""
    idx = NumpyIndex(dim=4, dtype="int8")
    rng = np.random.default_rng(0)
    idx.append(1, _rand(rng, 4), "default")
    assert idx._matrix.dtype == np.int8  # type: ignore[attr-defined]


def test_dtype_consistency_after_float16_storage():
    idx = NumpyIndex(dim=4, dtype="float16")
    rng = np.random.default_rng(0)
    idx.append(1, _rand(rng, 4), "default")
    assert idx._matrix.dtype == np.float16  # type: ignore[attr-defined]


# --- Requantize ---


def test_requantize_changes_dtype():
    idx = NumpyIndex(dim=8, dtype="float32")
    rng = np.random.default_rng(0)
    for i in range(5):
        idx.append(i + 1, _rand(rng, 8), "default")
    idx.requantize("int8")
    assert idx.dtype == "int8"
    assert idx._matrix.dtype == np.int8  # type: ignore[attr-defined]


def test_requantize_is_noop_for_same_dtype():
    idx = NumpyIndex(dim=8, dtype="float32")
    rng = np.random.default_rng(0)
    for i in range(3):
        idx.append(i + 1, _rand(rng, 8), "default")
    pre_matrix = idx._matrix.copy()  # type: ignore[attr-defined]
    idx.requantize("float32")
    np.testing.assert_array_equal(idx._matrix, pre_matrix)  # type: ignore[attr-defined]


def test_requantize_preserves_top_hit():
    idx = NumpyIndex(dim=64, dtype="float32")
    rng = np.random.default_rng(0)
    target = _l2(rng.standard_normal(64).astype(np.float32))
    for i in range(20):
        idx.append(i + 1, _l2(rng.standard_normal(64).astype(np.float32)), "default")
    idx.append(999, target, "default")
    pre_top = idx.search(target, "default", k=1)[0][0]
    idx.requantize("float16")
    post_top = idx.search(target, "default", k=1)[0][0]
    assert pre_top == post_top == 999


# --- Memory ---


def test_memory_bytes_reflects_dtype():
    idx_f32 = NumpyIndex(dim=4, dtype="float32", initial_capacity=100)
    idx_f16 = NumpyIndex(dim=4, dtype="float16", initial_capacity=100)
    idx_i8 = NumpyIndex(dim=4, dtype="int8", initial_capacity=100)
    assert idx_f32.memory_bytes == 100 * 4 * 4
    assert idx_f16.memory_bytes == 100 * 4 * 2
    assert idx_i8.memory_bytes == 100 * 4 * 1
