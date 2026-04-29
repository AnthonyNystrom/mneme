"""``NumpyIndex``: in-memory vector matrix with namespace filtering.

Cosine similarity on L2-normalized vectors only (via ``M @ q``). The store
holds the source-of-truth float32 vectors; the index keeps a quantized
in-memory copy per ``vector_dtype``.

Per PRD §10.2 / §11.3:

- Append grows capacity geometrically; removed rows become tombstones until
  ``compact()`` rebuilds.
- Search restricts to a single namespace via per-namespace offset arrays.
- For ``int8``, the matrix is stored as int8 and dequantized on-the-fly to
  float32 for the matvec (faster than maintaining a parallel float32 copy
  on x86 CPUs without AVX-512 FP16).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt

from ._quantization import dequantize, np_dtype_for, quantize
from ._types import VectorDtype

_INITIAL_CAPACITY = 256
# Chunk size (rows) for the dequantize-then-matvec path. 4096 rows balances
# matvec setup cost (which favors larger chunks) against cache pressure (which
# favors smaller). Empirically the sweet spot on M-series + x86 for typical
# embedding dims (768-3072). The per-chunk fp32 buffer is reused across calls.
_MATVEC_CHUNK = 4096


def _chunked_matvec(
    matrix: npt.NDArray[Any],
    query_f32: npt.NDArray[np.float32],
    dtype: VectorDtype,
) -> npt.NDArray[np.float32]:
    """Compute ``dequantize(matrix, dtype) @ query`` without materializing the
    full dequantized copy. For float32, calls matmul directly (no copy).

    For int8: the standard dequant is ``arr.astype(f32) / 127.0`` -- two
    600MB+ allocations at 100k x 1536. Instead we pre-scale the query by
    ``1/127`` (small) and only do ``chunk.astype(f32) @ scaled_q``,
    halving memory traffic on the big array.
    """
    if dtype == "float32":
        result: npt.NDArray[np.float32] = matrix @ query_f32
        return result
    n = matrix.shape[0]
    dim = matrix.shape[1]
    scores = np.empty(n, dtype=np.float32)
    # Reused L2-resident buffer for the fp32 cast.
    buf = np.empty((_MATVEC_CHUNK, dim), dtype=np.float32)
    if dtype == "int8":
        # Push the 1/127 scale onto the query side; keeps the big array
        # path to a single fp32 cast per chunk.
        scaled_q = (query_f32 / 127.0).astype(np.float32)
        for start in range(0, n, _MATVEC_CHUNK):
            end = min(start + _MATVEC_CHUNK, n)
            count = end - start
            np.copyto(buf[:count], matrix[start:end], casting="unsafe")
            scores[start:end] = buf[:count] @ scaled_q
        return scores
    # float16: cast-only dequant; reuse the buffer the same way.
    for start in range(0, n, _MATVEC_CHUNK):
        end = min(start + _MATVEC_CHUNK, n)
        count = end - start
        np.copyto(buf[:count], matrix[start:end], casting="unsafe")
        scores[start:end] = buf[:count] @ query_f32
    return scores


class NumpyIndex:
    """NumPy-backed Index over L2-normalized vectors."""

    def __init__(
        self,
        dim: int,
        *,
        dtype: VectorDtype = "float32",
        initial_capacity: int = _INITIAL_CAPACITY,
    ) -> None:
        if dim <= 0:
            raise ValueError(
                f"NumpyIndex: dim must be positive (got {dim}). "
                f"Remediation: pass the embedder's output dimension."
            )
        if initial_capacity <= 0:
            raise ValueError(
                f"NumpyIndex: initial_capacity must be positive (got {initial_capacity})."
            )
        self._dim = dim
        self._dtype: VectorDtype = dtype
        self._initial_capacity = initial_capacity
        self._matrix: npt.NDArray[Any] = np.zeros(
            (initial_capacity, dim), dtype=np_dtype_for(dtype)
        )
        self._row_ids: npt.NDArray[np.int64] = np.zeros(initial_capacity, dtype=np.int64)
        # Per-namespace list of matrix-row offsets (kept in insertion order).
        self._namespace_offsets: dict[str, list[int]] = {}
        # row_id → matrix-row offset, for O(1) updates and removes.
        self._row_id_to_offset: dict[int, int] = {}
        # Offset → namespace, for compact() and remove() bookkeeping.
        self._offset_namespace: dict[int, str] = {}
        self._tombstones: set[int] = set()
        # Logical capacity used (matrix rows allocated, including tombstones).
        self._size: int = 0

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def size(self) -> int:
        """Number of live (non-tombstoned) rows."""
        return self._size - len(self._tombstones)

    @property
    def dtype(self) -> VectorDtype:
        return self._dtype

    @property
    def memory_bytes(self) -> int:
        """Bytes occupied by the matrix (excludes Python overhead)."""
        return int(self._matrix.nbytes)

    # --- internal: capacity growth ---

    def _grow_to(self, new_capacity: int) -> None:
        old = self._matrix
        self._matrix = np.zeros((new_capacity, self._dim), dtype=old.dtype)
        self._matrix[: old.shape[0]] = old
        new_ids = np.zeros(new_capacity, dtype=np.int64)
        new_ids[: self._row_ids.shape[0]] = self._row_ids
        self._row_ids = new_ids

    def _ensure_capacity(self, needed: int) -> None:
        if self._matrix.shape[0] >= needed:
            return
        new_capacity = max(self._matrix.shape[0] * 2, needed)
        self._grow_to(new_capacity)

    # --- public API (Index protocol) ---

    def append(self, row_id: int, vec: npt.NDArray[Any], namespace: str) -> None:
        if vec.shape != (self._dim,):
            raise ValueError(
                f"NumpyIndex: vector shape {vec.shape} does not match dim={self._dim}. "
                f"Remediation: pass a 1-D array of length {self._dim}."
            )
        # Stage 1: ensure float32 input for downstream quantization.
        v32 = vec.astype(np.float32, copy=False)
        existing_offset = self._row_id_to_offset.get(row_id)
        if existing_offset is None:
            self._ensure_capacity(self._size + 1)
            offset = self._size
            self._size += 1
            self._row_id_to_offset[row_id] = offset
            self._offset_namespace[offset] = namespace
            self._namespace_offsets.setdefault(namespace, []).append(offset)
        else:
            offset = existing_offset
            # Update namespace bookkeeping if it changed (rare).
            old_ns = self._offset_namespace[offset]
            if old_ns != namespace:
                self._namespace_offsets[old_ns].remove(offset)
                if not self._namespace_offsets[old_ns]:
                    del self._namespace_offsets[old_ns]
                self._namespace_offsets.setdefault(namespace, []).append(offset)
                self._offset_namespace[offset] = namespace
            # Reviving from tombstone after a remove+append cycle.
            self._tombstones.discard(offset)
        self._matrix[offset] = quantize(v32, self._dtype)
        self._row_ids[offset] = row_id

    def remove(self, row_id: int) -> None:
        offset = self._row_id_to_offset.pop(row_id, None)
        if offset is None:
            return
        ns = self._offset_namespace.pop(offset, None)
        if ns is not None and offset in self._namespace_offsets.get(ns, []):
            self._namespace_offsets[ns].remove(offset)
            if not self._namespace_offsets[ns]:
                del self._namespace_offsets[ns]
        self._tombstones.add(offset)

    def search(
        self,
        query: npt.NDArray[Any],
        namespace: str,
        *,
        k: int = 1,
    ) -> list[tuple[int, float]]:
        if query.shape != (self._dim,):
            raise ValueError(
                f"NumpyIndex: query shape {query.shape} does not match dim={self._dim}."
            )
        if k <= 0:
            return []
        offsets_list = self._namespace_offsets.get(namespace)
        if not offsets_list:
            return []

        # Fast path: when the namespace contains every live row (single ns,
        # no tombstones), skip fancy indexing and matvec the matrix directly.
        # This saves a 100k x 768 fp32 copy (~300MB) for the common case and
        # is the difference between 40ms and 0.5ms search at 100k entries.
        single_ns_full = (
            not self._tombstones
            and len(offsets_list) == self._size
            and len(self._namespace_offsets) == 1
        )
        q32 = query.astype(np.float32, copy=False)
        if single_ns_full:
            slice_view = self._matrix[: self._size]
            scores = _chunked_matvec(slice_view, q32, self._dtype)
            if k >= len(scores):
                order = np.argsort(-scores)
            else:
                top = np.argpartition(-scores, k)[:k]
                order = top[np.argsort(-scores[top])]
            return [(int(self._row_ids[i]), float(scores[i])) for i in order[:k]]

        # General path: filter tombstones, fancy-index the matrix.
        live = [o for o in offsets_list if o not in self._tombstones]
        if not live:
            return []
        offsets = np.array(live, dtype=np.int64)
        slice_ = self._matrix[offsets]
        scores = _chunked_matvec(slice_, q32, self._dtype)
        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            top = np.argpartition(-scores, k)[:k]
            order = top[np.argsort(-scores[top])]
        return [(int(self._row_ids[offsets[i]]), float(scores[i])) for i in order[:k]]

    def rebuild_from(self, rows: Iterable[tuple[int, npt.NDArray[Any], str]]) -> None:
        """Discard current state and rebuild from a stream of (id, vec, ns).

        Bulk-vectorized: stacks all input vectors into a single ndarray and
        quantizes once (~10x faster than per-row ``append`` at 100k entries,
        which keeps the Phase-14 open-time targets in reach).
        """
        rows = list(rows)
        n = len(rows)
        capacity = max(self._initial_capacity, n)
        self._matrix = np.zeros((capacity, self._dim), dtype=np_dtype_for(self._dtype))
        self._row_ids = np.zeros(capacity, dtype=np.int64)
        self._namespace_offsets.clear()
        self._row_id_to_offset.clear()
        self._offset_namespace.clear()
        self._tombstones.clear()
        self._size = n

        if n == 0:
            return

        # Stack all input vectors into one matrix and quantize once.
        stacked = np.empty((n, self._dim), dtype=np.float32)
        ids = np.empty(n, dtype=np.int64)
        for i, (row_id, vec, ns) in enumerate(rows):
            stacked[i] = vec.astype(np.float32, copy=False)
            ids[i] = row_id
            self._row_id_to_offset[row_id] = i
            self._offset_namespace[i] = ns
            self._namespace_offsets.setdefault(ns, []).append(i)
        self._row_ids[:n] = ids
        if self._dtype == "float32":
            self._matrix[:n] = stacked
        else:
            self._matrix[:n] = quantize(stacked, self._dtype)

    def compact(self) -> None:
        """Rebuild the matrix dropping tombstoned rows. Idempotent if no
        tombstones exist."""
        if not self._tombstones:
            return
        live_rows: list[tuple[int, npt.NDArray[Any], str]] = []
        for offset in range(self._size):
            if offset in self._tombstones:
                continue
            row_id = int(self._row_ids[offset])
            ns = self._offset_namespace[offset]
            vec = dequantize(self._matrix[offset], self._dtype)
            live_rows.append((row_id, vec, ns))
        self.rebuild_from(live_rows)

    def requantize(self, dtype: VectorDtype) -> None:
        """Convert the in-memory matrix to a new dtype.

        Lossy when going to a less-precise dtype (e.g. float32 → int8). For
        full-precision rebuild, the cache layer should call ``rebuild_from``
        with float32 vectors from the Store.
        """
        if dtype == self._dtype:
            return
        # Dequantize each row to float32, then re-quantize at new dtype.
        new_matrix = np.zeros(self._matrix.shape, dtype=np_dtype_for(dtype))
        for offset in range(self._size):
            if offset in self._tombstones:
                continue
            f32 = dequantize(self._matrix[offset], self._dtype)
            new_matrix[offset] = quantize(f32, dtype)
        self._matrix = new_matrix
        self._dtype = dtype


__all__ = ["NumpyIndex"]
