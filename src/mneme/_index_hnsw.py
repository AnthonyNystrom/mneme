"""``HnswIndex``: hnswlib-backed Index. Optional ``[hnsw]`` extra.

One ``hnswlib.Index`` per namespace, all over cosine space. Lazy-imports
``hnswlib`` so the core install stays NumPy-only. Quantization is not
supported (hnswlib uses fp32 internally); use ``NumpyIndex`` for fp16/int8.

Per PRD §10.3 / §16:

- Cosine space, with hnswlib distance = ``1 - cos_sim``; we convert back to
  similarity at query time.
- ``index_options`` exposes ``M``, ``ef_construction``, ``ef``, and
  ``initial_max_elements``. The index resizes geometrically on demand.
- ``remove(row_id)`` marks the element deleted (hnswlib soft-delete).
  ``compact()`` is a no-op; deletions reclaim space only on full rebuild.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt

from ._exceptions import IndexBackendUnavailableError, QuantizationError
from ._types import VectorDtype

_DEFAULTS = {
    "M": 16,
    "ef_construction": 200,
    "ef": 50,
    "initial_max_elements": 10_000,
}


def _import_hnswlib() -> Any:
    try:
        import hnswlib
    except ImportError as exc:  # pragma: no cover - extras-gated
        raise IndexBackendUnavailableError(
            "HnswIndex requires the optional 'hnsw' extra. "
            "Remediation: pip install mneme[hnsw], or use index_backend='numpy'."
        ) from exc
    return hnswlib


class HnswIndex:
    """hnswlib-backed Index with one underlying index per namespace."""

    def __init__(
        self,
        dim: int,
        *,
        dtype: VectorDtype = "float32",
        index_options: dict[str, Any] | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError(
                f"HnswIndex: dim must be positive (got {dim}). "
                f"Remediation: pass the embedder's output dimension."
            )
        if dtype != "float32":
            raise QuantizationError(
                f"HnswIndex does not support vector_dtype={dtype!r}; hnswlib "
                f"uses float32 internally. Remediation: use NumpyIndex with "
                f"int8 or float16 for memory-constrained workloads."
            )
        self._hnswlib = _import_hnswlib()
        self._dim = dim
        self._dtype: VectorDtype = "float32"
        opts = {**_DEFAULTS, **(index_options or {})}
        self._M = int(opts["M"])
        self._ef_construction = int(opts["ef_construction"])
        self._ef = int(opts["ef"])
        self._initial_max = int(opts["initial_max_elements"])
        self._indexes: dict[str, Any] = {}
        self._row_id_to_ns: dict[int, str] = {}
        # Per-namespace soft-deletion tracking (mirrors hnswlib's mark_deleted
        # state). Lives across cross-namespace id moves: an id moved from
        # tenant_a → tenant_b is still soft-deleted in tenant_a but live in
        # tenant_b.
        self._deleted_per_ns: dict[str, set[int]] = {}

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def size(self) -> int:
        total = sum(idx.element_count for idx in self._indexes.values())
        deleted = sum(len(s) for s in self._deleted_per_ns.values())
        return int(total) - deleted

    @property
    def dtype(self) -> VectorDtype:
        return self._dtype

    # --- internal: per-namespace index helpers ---

    def _make_index(self) -> Any:
        idx = self._hnswlib.Index(space="cosine", dim=self._dim)
        # ``allow_replace_deleted=True`` lets ``append`` revive a soft-deleted
        # row in place (used in the remove-then-append-same-id revival path).
        try:
            idx.init_index(
                max_elements=self._initial_max,
                ef_construction=self._ef_construction,
                M=self._M,
                allow_replace_deleted=True,
            )
        except TypeError:  # pragma: no cover - older hnswlib without kw
            idx.init_index(
                max_elements=self._initial_max,
                ef_construction=self._ef_construction,
                M=self._M,
            )
        idx.set_ef(self._ef)
        return idx

    def _ensure_capacity(self, idx: Any) -> None:
        # hnswlib max_elements is a hard cap; resize geometrically when full.
        if idx.element_count + 1 > idx.max_elements:
            idx.resize_index(max(idx.max_elements * 2, idx.element_count + 1))

    # --- public API (Index protocol) ---

    def append(self, row_id: int, vec: npt.NDArray[Any], namespace: str) -> None:
        if vec.shape != (self._dim,):
            raise ValueError(f"HnswIndex: vector shape {vec.shape} does not match dim={self._dim}.")
        v32 = vec.astype(np.float32, copy=False)
        existing_ns = self._row_id_to_ns.get(row_id)
        if existing_ns is not None and existing_ns != namespace:
            # Soft-delete the prior row in the old namespace's index. The
            # per-ns deletion entry stays even after _row_id_to_ns is updated
            # so size and search filtering remain correct for tenant_a.
            with contextlib.suppress(RuntimeError):
                self._indexes[existing_ns].mark_deleted(row_id)
            self._deleted_per_ns.setdefault(existing_ns, set()).add(row_id)
        if namespace not in self._indexes:
            self._indexes[namespace] = self._make_index()
        idx = self._indexes[namespace]
        ns_deleted = self._deleted_per_ns.setdefault(namespace, set())
        if existing_ns == namespace and row_id in ns_deleted:
            # Reviving from a prior remove() in the same namespace.
            with contextlib.suppress(RuntimeError):
                idx.unmark_deleted(row_id)
            ns_deleted.discard(row_id)
            with contextlib.suppress(TypeError, RuntimeError):
                idx.add_items(
                    v32.reshape(1, -1),
                    np.array([row_id], dtype=np.int64),
                    replace_deleted=True,
                )
            self._row_id_to_ns[row_id] = namespace
            return
        self._ensure_capacity(idx)
        idx.add_items(v32.reshape(1, -1), np.array([row_id], dtype=np.int64))
        self._row_id_to_ns[row_id] = namespace

    def remove(self, row_id: int) -> None:
        ns = self._row_id_to_ns.get(row_id)
        if ns is None:
            return
        idx = self._indexes.get(ns)
        if idx is None:
            return
        with contextlib.suppress(RuntimeError):
            idx.mark_deleted(row_id)
        self._deleted_per_ns.setdefault(ns, set()).add(row_id)

    def search(
        self,
        query: npt.NDArray[Any],
        namespace: str,
        *,
        k: int = 1,
    ) -> list[tuple[int, float]]:
        if query.shape != (self._dim,):
            raise ValueError(
                f"HnswIndex: query shape {query.shape} does not match dim={self._dim}."
            )
        if k <= 0:
            return []
        idx = self._indexes.get(namespace)
        if idx is None:
            return []
        ns_deleted = self._deleted_per_ns.get(namespace, set())
        live = int(idx.element_count) - len(ns_deleted)
        if live <= 0:
            return []
        actual_k = min(k, live)
        idx.set_ef(max(self._ef, actual_k))
        q32 = query.astype(np.float32, copy=False).reshape(1, -1)
        labels, distances = idx.knn_query(q32, k=actual_k)
        # cosine space: hnswlib returns distance = 1 - cos_sim.
        return [
            (int(label), float(1.0 - dist))
            for label, dist in zip(labels[0], distances[0], strict=False)
            if int(label) not in ns_deleted
        ]

    def rebuild_from(self, rows: Iterable[tuple[int, npt.NDArray[Any], str]]) -> None:
        rows = list(rows)
        self._indexes.clear()
        self._row_id_to_ns.clear()
        self._deleted_per_ns.clear()
        for row_id, vec, ns in rows:
            self.append(row_id, vec, ns)

    def compact(self) -> None:
        """Rebuild each namespace index, dropping soft-deleted rows.

        hnswlib does not reclaim deleted slots in place; this is the only way
        to actually shrink memory after many ``remove`` calls.
        """
        if not any(self._deleted_per_ns.values()):
            return
        live: list[tuple[int, npt.NDArray[Any], str]] = []
        for row_id, ns in list(self._row_id_to_ns.items()):
            if row_id in self._deleted_per_ns.get(ns, set()):
                continue
            idx = self._indexes.get(ns)
            if idx is None:
                continue
            try:
                vec_list = idx.get_items([row_id])
            except RuntimeError:
                continue
            arr = np.asarray(vec_list, dtype=np.float32)
            if arr.size == 0:
                continue
            # hnswlib may return shape (1, dim) or (dim,) depending on version.
            row_vec = arr[0] if arr.ndim == 2 else arr
            live.append((row_id, row_vec, ns))
        self.rebuild_from(live)

    def requantize(self, dtype: VectorDtype) -> None:
        if dtype == self._dtype:
            return
        raise QuantizationError(
            f"HnswIndex only supports vector_dtype='float32'; cannot requantize "
            f"to {dtype!r}. Remediation: switch to NumpyIndex via "
            f"index_backend='numpy' for fp16/int8 quantization."
        )


__all__ = ["HnswIndex"]
