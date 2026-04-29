"""Multi-process coordinators.

Two modes:

- ``StaleTolerantCoordinator``: each process keeps its own in-memory index
  and polls ``store.read_version_counter()``. On bump, it pulls deltas via
  ``store.iter_since`` (or full rebuild if delta >= 100). Cheap, eventually
  consistent. Production default for multi-worker servers.

- ``MmapSharedCoordinator``: shared-memory matrix in a sidecar file,
  arbitrated by an advisory file lock. Strong cross-process consistency at
  the cost of a write-side lock contention point. Use when latency matters
  more than write throughput.

Both modes are *coordinators* — they sit between the cache and the
in-memory index, intercepting reads/writes to keep state synchronized. The
cache constructs the right coordinator from ``multi_process_mode`` and
dispatches index ops through it.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import struct
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ._exceptions import (
    EmbedderDimensionError,
    EmbedderMismatchError,
    MultiProcessLockError,
    QuantizationError,
)
from ._quantization import (
    bytes_per_element,
    dequantize,
    np_dtype_for,
    quantize,
)
from ._types import VectorDtype

if TYPE_CHECKING:
    from ._types import Index, Store

logger = logging.getLogger("mneme.multiproc")

# Above this many missed version bumps, do a full rebuild instead of streaming
# deltas.
_REBUILD_THRESHOLD = 100


# =============================================================================
# Stale-tolerant
# =============================================================================


class StaleTolerantCoordinator:
    """Multi-process coordinator using ``version_counter`` polling.

    Each process holds its own in-memory ``Index`` mirror of the store. On
    every cache operation (under the cache RLock), the coordinator reads
    the store's ``version_counter``. If it has advanced since the last
    check, it pulls deltas via ``iter_since`` (or rebuilds from scratch if
    too far behind).
    """

    REBUILD_THRESHOLD = _REBUILD_THRESHOLD

    def __init__(
        self,
        store: Store,
        index: Index,
        *,
        stale_check_interval: float = 0.0,
    ) -> None:
        self._store = store
        self._index = index
        self._stale_check_interval = float(stale_check_interval)
        # Snapshot the store's current version + max id so we don't refetch
        # entries that were already in the index when this coordinator
        # attached (the cache already loaded them).
        self._local_version = store.read_version_counter()
        self._last_seen_id = self._compute_last_id()
        self._last_check_time = 0.0

    @property
    def local_version(self) -> int:
        return self._local_version

    @property
    def last_seen_id(self) -> int:
        return self._last_seen_id

    def _compute_last_id(self) -> int:
        max_id = 0
        for entry in self._store.iter_all():
            if entry.id > max_id:
                max_id = entry.id
        return max_id

    def refresh(self) -> int:
        """Sync the local index from the store. Returns number of entries
        newly added (0 if nothing changed)."""
        now = time.monotonic()
        if (
            self._stale_check_interval > 0.0
            and (now - self._last_check_time) < self._stale_check_interval
        ):
            return 0
        self._last_check_time = now

        counter = self._store.read_version_counter()
        if counter <= self._local_version:
            return 0

        delta = counter - self._local_version
        if delta >= self.REBUILD_THRESHOLD:
            return self._full_rebuild(counter)
        return self._incremental_sync(counter)

    def _full_rebuild(self, counter: int) -> int:
        """Drop all in-memory state and re-load from the store."""
        rows: list[tuple[int, npt.NDArray[np.float32], str]] = []
        max_id = 0
        for entry in self._store.iter_all():
            vec = np.frombuffer(entry.embedding, dtype=np.float32).copy()
            rows.append((entry.id, vec, entry.namespace))
            if entry.id > max_id:
                max_id = entry.id
        self._index.rebuild_from(rows)
        added = len(rows)
        self._last_seen_id = max_id
        self._local_version = counter
        logger.info(
            "stale-tolerant: full rebuild, %d rows from store (delta=%d)",
            added,
            counter - self._local_version,
        )
        return added

    def _incremental_sync(self, counter: int) -> int:
        """Stream entries with id > last_seen_id and append to the index."""
        added = 0
        for entry in self._store.iter_since(self._last_seen_id):
            vec = np.frombuffer(entry.embedding, dtype=np.float32).copy()
            self._index.append(entry.id, vec, entry.namespace)
            self._last_seen_id = entry.id
            added += 1
        self._local_version = counter
        return added


# =============================================================================
# Mmap-shared
# =============================================================================
#
# File layout:
#
#   [ header (64 bytes) ]
#     magic         (8) — b"MNEMEMSC"
#     version       (4) — file format version
#     dim           (4) — embedding dimension
#     count         (4) — populated rows
#     capacity      (4) — allocated rows
#     dtype_code    (4) — 0=fp32, 1=fp16, 2=int8
#     fingerprint   (32) — SHA-256 of embedder fingerprint
#     reserved      (4)
#   [ row_id table   ] capacity * 8 bytes (int64)
#   [ namespace tbl  ] capacity * 16 bytes (utf-8 left-padded)
#   [ tombstones     ] ceil(capacity / 8) bytes (bitmap)
#   [ vector data    ] capacity * dim * dtype_size bytes


_MAGIC = b"MNEMEMSC"
_FILE_FORMAT_VERSION = 1
_HEADER_SIZE = 64
_NS_TOKEN_LEN = 16
_DTYPE_CODES: dict[VectorDtype, int] = {"float32": 0, "float16": 1, "int8": 2}
_DTYPE_FROM_CODE: dict[int, VectorDtype] = {v: k for k, v in _DTYPE_CODES.items()}


def _ns_to_token(ns: str) -> bytes:
    raw = ns.encode("utf-8")
    if len(raw) > _NS_TOKEN_LEN:
        # Hash-prefix overflow so two different long namespaces don't collide.
        return raw[: _NS_TOKEN_LEN - 6] + hashlib.sha256(raw).digest()[:6]
    return raw.ljust(_NS_TOKEN_LEN, b"\x00")


def _token_to_ns(token: bytes) -> str:
    return token.rstrip(b"\x00").decode("utf-8", errors="replace")


def _bitmap_set(buf: bytearray, idx: int, value: bool) -> None:
    byte_idx, bit = divmod(idx, 8)
    if value:
        buf[byte_idx] |= 1 << bit
    else:
        buf[byte_idx] &= ~(1 << bit)


def _bitmap_get(buf: bytes, idx: int) -> bool:
    byte_idx, bit = divmod(idx, 8)
    if byte_idx >= len(buf):
        return False
    return bool(buf[byte_idx] & (1 << bit))


@contextmanager
def _flock_exclusive(fd: int) -> Any:
    """Cross-platform advisory exclusive file lock.

    POSIX: ``fcntl.flock``. Windows: ``msvcrt.locking`` (best-effort).
    """
    if os.name == "posix":
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as exc:
            raise MultiProcessLockError(
                f"Failed to acquire exclusive flock: {exc}. Remediation: "
                f"check process permissions on the lock file."
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    elif os.name == "nt":  # pragma: no cover - tested only on POSIX in CI
        import msvcrt

        try:
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
        except OSError as exc:
            raise MultiProcessLockError(
                f"Failed to acquire msvcrt lock: {exc}. Remediation: check file access permissions."
            ) from exc
        try:
            yield
        finally:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
    else:  # pragma: no cover - defensive
        # No locking primitive: behave as if locked (single-process fallback).
        yield


def _header_pack(
    dim: int,
    count: int,
    capacity: int,
    dtype: VectorDtype,
    fp_hash: bytes,
) -> bytes:
    body = struct.pack(
        "<8sIIIII32s4s",
        _MAGIC,
        _FILE_FORMAT_VERSION,
        dim,
        count,
        capacity,
        _DTYPE_CODES[dtype],
        fp_hash,
        b"\x00" * 4,
    )
    assert len(body) == _HEADER_SIZE, f"header size {len(body)} != {_HEADER_SIZE}"
    return body


def _header_unpack(buf: bytes) -> tuple[int, int, int, VectorDtype, bytes]:
    """Returns (dim, count, capacity, dtype, fp_hash). Raises if magic invalid."""
    if len(buf) < _HEADER_SIZE:
        raise MultiProcessLockError(
            f"Header read short ({len(buf)} < {_HEADER_SIZE}). Remediation: "
            f"the vectors file is corrupt; delete it and let the cache "
            f"rebuild it from the Store."
        )
    (
        magic,
        version,
        dim,
        count,
        capacity,
        dtype_code,
        fp_hash,
        _reserved,
    ) = struct.unpack("<8sIIIII32s4s", buf[:_HEADER_SIZE])
    if magic != _MAGIC:
        raise MultiProcessLockError(
            f"Vectors file magic {magic!r} != expected {_MAGIC!r}. "
            f"Remediation: delete the file or check that you're pointing "
            f"at a mneme-managed vectors file."
        )
    if version != _FILE_FORMAT_VERSION:
        raise MultiProcessLockError(
            f"Vectors file version {version} != supported {_FILE_FORMAT_VERSION}. "
            f"Remediation: upgrade the mneme library."
        )
    if dtype_code not in _DTYPE_FROM_CODE:
        raise QuantizationError(
            f"Unknown dtype_code {dtype_code} in vectors file. Remediation: "
            f"file may be from a future library version."
        )
    return dim, count, capacity, _DTYPE_FROM_CODE[dtype_code], fp_hash


class MmapSharedCoordinator:
    """Shared-matrix multi-process coordinator backed by a sidecar file.

    All processes attaching to the same ``base_path`` see the same matrix
    bytes via memory-mapped I/O. Writers acquire an exclusive ``flock`` on
    a sidecar ``.lock`` file before mutating; readers re-mmap when the
    header's ``count`` advances.

    dtype is fixed at file creation and validated on attach;
    mismatched dtypes across processes raise ``QuantizationError``.
    """

    def __init__(
        self,
        base_path: str | Path,
        dim: int,
        *,
        dtype: VectorDtype = "float32",
        embedder_fingerprint: str = "",
        initial_capacity: int = 1024,
    ) -> None:
        self._base_path = Path(base_path)
        self._data_path = Path(str(base_path) + ".vectors")
        self._lock_path = Path(str(base_path) + ".vectors.lock")
        self._dim = dim
        self._dtype: VectorDtype = dtype
        self._fp_hash = hashlib.sha256(embedder_fingerprint.encode("utf-8")).digest()
        self._initial_capacity = max(int(initial_capacity), 1)
        self._closed = False
        # Lock file fd kept open for the lifetime of the coordinator.
        self._lock_fd: int | None = None
        # Local capacity snapshot (for re-mmap detection).
        self._capacity_seen = 0
        self._open_or_create()

    # --- file management ---

    def _open_or_create(self) -> None:
        # Lock fd
        self._lock_fd = os.open(
            self._lock_path,
            os.O_RDWR | os.O_CREAT,
            0o600,
        )
        if not self._data_path.exists() or self._data_path.stat().st_size < _HEADER_SIZE:
            # Initialize new file under exclusive lock.
            with _flock_exclusive(self._lock_fd):
                self._initialize_new_file()
        else:
            self._validate_existing_file()

    def _initialize_new_file(self) -> None:
        cap = self._initial_capacity
        size = self._file_size_for(cap)
        with open(self._data_path, "wb") as f:
            f.write(_header_pack(self._dim, 0, cap, self._dtype, self._fp_hash))
            # Zero-fill the rest.
            f.write(b"\x00" * (size - _HEADER_SIZE))
        os.chmod(self._data_path, 0o600)
        self._capacity_seen = cap

    def _validate_existing_file(self) -> None:
        with open(self._data_path, "rb") as f:
            header = f.read(_HEADER_SIZE)
        dim, _count, cap, dtype, fp_hash = _header_unpack(header)
        if dim != self._dim:
            raise EmbedderDimensionError(
                f"Vectors file dim={dim} != expected {self._dim}. Remediation: "
                f"either point at a different file, or use reembed() to migrate."
            )
        if dtype != self._dtype:
            raise QuantizationError(
                f"Vectors file dtype={dtype} != expected {self._dtype}. "
                f"Remediation: ensure all processes use the same vector_dtype, "
                f"or delete the .vectors sidecar so it gets re-created."
            )
        if fp_hash != self._fp_hash:
            raise EmbedderMismatchError(
                "Vectors file fingerprint hash mismatch. Remediation: open with "
                "the original embedder, or delete the .vectors file and let it "
                "rebuild from the Store."
            )
        self._capacity_seen = cap

    def _file_size_for(self, capacity: int) -> int:
        elem_bytes = bytes_per_element(self._dtype)
        bitmap_bytes = (capacity + 7) // 8
        return (
            _HEADER_SIZE
            + capacity * 8  # row_id table
            + capacity * _NS_TOKEN_LEN  # namespace table
            + bitmap_bytes
            + capacity * self._dim * elem_bytes
        )

    def _offsets_for(self, capacity: int) -> tuple[int, int, int, int]:
        elem_bytes = bytes_per_element(self._dtype)
        ids_off = _HEADER_SIZE
        ns_off = ids_off + capacity * 8
        tomb_off = ns_off + capacity * _NS_TOKEN_LEN
        vec_off = tomb_off + ((capacity + 7) // 8)
        del elem_bytes  # used implicitly via _file_size_for
        return ids_off, ns_off, tomb_off, vec_off

    # --- public coordinator API ---

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dtype(self) -> VectorDtype:
        return self._dtype

    def read_header(self) -> tuple[int, int, int, VectorDtype, bytes]:
        """Read the current header (under no lock; for inspection only)."""
        with open(self._data_path, "rb") as f:
            return _header_unpack(f.read(_HEADER_SIZE))

    def append(self, row_id: int, vec: npt.NDArray[Any], namespace: str) -> None:
        """Append a row under exclusive lock. Grows the file if needed."""
        if vec.shape != (self._dim,):
            raise ValueError(
                f"MmapSharedCoordinator.append: vector shape {vec.shape} != ({self._dim},)."
            )
        if self._lock_fd is None:
            raise MultiProcessLockError("MmapSharedCoordinator is closed.")
        v32 = vec.astype(np.float32, copy=False)
        v32 = self._l2_normalize(v32)
        qvec = quantize(v32, self._dtype).tobytes()
        token = _ns_to_token(namespace)

        with _flock_exclusive(self._lock_fd):
            # Read header to decide whether we need to grow.
            with open(self._data_path, "rb") as f:
                header = f.read(_HEADER_SIZE)
            dim, count, capacity, _dt, _fp = _header_unpack(header)
            if count >= capacity:
                capacity = self._grow_locked(capacity)
            ids_off, ns_off, tomb_off, vec_off = self._offsets_for(capacity)
            slot = count
            elem_bytes = bytes_per_element(self._dtype)
            with open(self._data_path, "r+b") as f:
                # row_id
                f.seek(ids_off + slot * 8)
                f.write(struct.pack("<q", int(row_id)))
                # namespace token
                f.seek(ns_off + slot * _NS_TOKEN_LEN)
                f.write(token)
                # Tombstone bit cleared (explicit for revival cases).
                f.seek(tomb_off + slot // 8)
                byte = f.read(1)
                buf = bytearray(byte if byte else b"\x00")
                _bitmap_set(buf, slot % 8, False)
                f.seek(tomb_off + slot // 8)
                f.write(bytes(buf))
                # vector data
                f.seek(vec_off + slot * dim * elem_bytes)
                f.write(qvec)
                # Bump count in header.
                count += 1
                f.seek(0)
                f.write(_header_pack(self._dim, count, capacity, self._dtype, self._fp_hash))
                f.flush()
                os.fsync(f.fileno())
        self._capacity_seen = capacity

    def _grow_locked(self, current_capacity: int) -> int:
        """Double the file's capacity. Caller holds the file lock and must
        not have any open handle to the data file (we replace it via
        ``os.replace``). Returns the new capacity."""
        new_capacity = current_capacity * 2
        new_size = self._file_size_for(new_capacity)
        tmp_path = self._data_path.with_suffix(self._data_path.suffix + ".grow")
        # Snapshot current contents.
        with open(self._data_path, "rb") as f:
            old_bytes = f.read()
        old_ids_off, old_ns_off, old_tomb_off, old_vec_off = self._offsets_for(current_capacity)
        new_ids_off, new_ns_off, new_tomb_off, new_vec_off = self._offsets_for(new_capacity)
        elem_bytes = bytes_per_element(self._dtype)
        with open(tmp_path, "wb") as g:
            old_count = struct.unpack("<I", old_bytes[12:16])[0]
            g.write(
                _header_pack(
                    self._dim,
                    old_count,
                    new_capacity,
                    self._dtype,
                    self._fp_hash,
                )
            )
            # Row-id table
            g.seek(new_ids_off)
            g.write(old_bytes[old_ids_off : old_ids_off + current_capacity * 8])
            g.write(b"\x00" * ((new_capacity - current_capacity) * 8))
            # Namespace table
            g.seek(new_ns_off)
            g.write(old_bytes[old_ns_off : old_ns_off + current_capacity * _NS_TOKEN_LEN])
            g.write(b"\x00" * ((new_capacity - current_capacity) * _NS_TOKEN_LEN))
            # Tombstone bitmap
            g.seek(new_tomb_off)
            old_bm_size = (current_capacity + 7) // 8
            new_bm_size = (new_capacity + 7) // 8
            g.write(old_bytes[old_tomb_off : old_tomb_off + old_bm_size])
            g.write(b"\x00" * (new_bm_size - old_bm_size))
            # Vector area
            g.seek(new_vec_off)
            old_vec_bytes = current_capacity * self._dim * elem_bytes
            g.write(old_bytes[old_vec_off : old_vec_off + old_vec_bytes])
            g.write(b"\x00" * ((new_capacity - current_capacity) * self._dim * elem_bytes))
            g.flush()
            current_pos = g.tell()
            if current_pos < new_size:
                g.write(b"\x00" * (new_size - current_pos))
            os.fsync(g.fileno())
        os.replace(tmp_path, self._data_path)
        with contextlib.suppress(OSError):  # pragma: no cover - filesystem may not support
            os.chmod(self._data_path, 0o600)
        return new_capacity

    @staticmethod
    def _l2_normalize(v32: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        n = float(np.linalg.norm(v32))
        if n == 0.0:
            return v32
        return (v32 / n).astype(np.float32, copy=False)

    def remove(self, row_id: int) -> bool:
        """Mark a row tombstoned. Returns True if a matching row was found."""
        if self._lock_fd is None:
            raise MultiProcessLockError("MmapSharedCoordinator is closed.")
        with _flock_exclusive(self._lock_fd), open(self._data_path, "r+b") as f:
            header = f.read(_HEADER_SIZE)
            dim, count, capacity, _dt, _fp = _header_unpack(header)
            del dim
            ids_off, _ns_off, tomb_off, _v = self._offsets_for(capacity)
            f.seek(ids_off)
            ids_buf = f.read(count * 8)
            ids = np.frombuffer(ids_buf, dtype="<i8")
            slot = -1
            for i, rid in enumerate(ids):
                if int(rid) == int(row_id):
                    slot = i
                    break
            if slot < 0:
                return False
            f.seek(tomb_off + slot // 8)
            byte = f.read(1)
            buf = bytearray(byte if byte else b"\x00")
            _bitmap_set(buf, slot % 8, True)
            f.seek(tomb_off + slot // 8)
            f.write(bytes(buf))
            f.flush()
            os.fsync(f.fileno())
            return True

    def search(
        self,
        query: npt.NDArray[Any],
        namespace: str,
        *,
        k: int = 1,
    ) -> list[tuple[int, float]]:
        """Read-only search across the shared matrix."""
        if query.shape != (self._dim,):
            raise ValueError(
                f"MmapSharedCoordinator.search: query shape {query.shape} != ({self._dim},)."
            )
        if k <= 0:
            return []
        target_token = _ns_to_token(namespace)
        with open(self._data_path, "rb") as f:
            header = f.read(_HEADER_SIZE)
            dim, count, capacity, _dt, _fp = _header_unpack(header)
            del dim
            if count == 0:
                return []
            ids_off, ns_off, tomb_off, vec_off = self._offsets_for(capacity)
            # Read full row-id table, namespace table, tombstone bitmap.
            f.seek(ids_off)
            ids = np.frombuffer(f.read(count * 8), dtype="<i8").copy()
            f.seek(ns_off)
            ns_tokens = f.read(count * _NS_TOKEN_LEN)
            f.seek(tomb_off)
            tomb_bytes = f.read((capacity + 7) // 8)
            # Filter to live rows in namespace.
            live: list[tuple[int, int]] = []  # (row_id, slot)
            for i in range(count):
                if _bitmap_get(tomb_bytes, i):
                    continue
                if ns_tokens[i * _NS_TOKEN_LEN : (i + 1) * _NS_TOKEN_LEN] != target_token:
                    continue
                live.append((int(ids[i]), i))
            if not live:
                return []
            # Read vector slice.
            elem_bytes = bytes_per_element(self._dtype)
            np_dtype = np_dtype_for(self._dtype)
            slots = np.array([s for _, s in live], dtype=np.int64)
            slice_bytes_each = self._dim * elem_bytes
            vecs = np.empty((len(live), self._dim), dtype=np_dtype)
            for j, slot in enumerate(slots):
                f.seek(vec_off + int(slot) * slice_bytes_each)
                buf = f.read(slice_bytes_each)
                vecs[j] = np.frombuffer(buf, dtype=np_dtype)
        # Score
        slice_f32 = dequantize(vecs, self._dtype)
        q32 = query.astype(np.float32, copy=False)
        q32 = self._l2_normalize(q32)
        scores = slice_f32 @ q32
        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            top = np.argpartition(-scores, k)[:k]
            order = top[np.argsort(-scores[top])]
        return [(live[i][0], float(scores[i])) for i in order[:k]]

    def close(self) -> None:
        if self._lock_fd is not None:
            with contextlib.suppress(OSError):  # pragma: no cover - best-effort
                os.close(self._lock_fd)
            self._lock_fd = None
        self._closed = True


__all__ = ["MmapSharedCoordinator", "StaleTolerantCoordinator"]
