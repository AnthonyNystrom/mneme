"""Vector quantization for the in-memory index.

Per PRD §10.4 and §11.5, the Store always persists ``float32``; quantization
to ``float16`` or ``int8`` is an in-memory representation choice.

- ``float16``: trivial cast. < 0.1% similarity drift on typical embeddings.
- ``int8``: per-vector scalar quantization. Assumes L2-normalized input
  (values in [-1, 1]). Symmetric scheme with implicit scale 127. Typical
  similarity drift 1-3%; calibrate similarity_threshold against the same
  dtype used in production.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ._exceptions import QuantizationError
from ._types import VectorDtype

_NP_DTYPES: dict[VectorDtype, np.dtype[Any]] = {
    "float32": np.dtype(np.float32),
    "float16": np.dtype(np.float16),
    "int8": np.dtype(np.int8),
}

_BYTES_PER_ELEMENT: dict[VectorDtype, int] = {
    "float32": 4,
    "float16": 2,
    "int8": 1,
}


def np_dtype_for(vector_dtype: VectorDtype) -> np.dtype[Any]:
    """Return the numpy dtype matching a ``VectorDtype`` literal."""
    if vector_dtype not in _NP_DTYPES:
        raise QuantizationError(
            f"Unknown vector_dtype {vector_dtype!r}. Remediation: use one of {list(_NP_DTYPES)}."
        )
    return _NP_DTYPES[vector_dtype]


def bytes_per_element(vector_dtype: VectorDtype) -> int:
    return _BYTES_PER_ELEMENT[vector_dtype]


def quantize_float16(vec: npt.NDArray[np.float32]) -> npt.NDArray[np.float16]:
    return vec.astype(np.float16)


def dequantize_float16(qvec: npt.NDArray[np.float16]) -> npt.NDArray[np.float32]:
    return qvec.astype(np.float32)


def quantize_int8(vec: npt.NDArray[np.float32]) -> npt.NDArray[np.int8]:
    """Symmetric int8 quantization with implicit scale 127.

    Assumes L2-normalized input. Values outside [-1, 1] still produce valid
    int8 output (clipped) but the implicit scale is no longer accurate, so
    callers must L2-normalize before calling this.
    """
    return np.clip(np.round(vec * 127.0), -128, 127).astype(np.int8)


def dequantize_int8(qvec: npt.NDArray[np.int8]) -> npt.NDArray[np.float32]:
    return qvec.astype(np.float32) / 127.0


def quantize(vec: npt.NDArray[np.float32], dtype: VectorDtype) -> npt.NDArray[Any]:
    """Dispatch to the right quantizer based on dtype literal."""
    if dtype == "float32":
        return vec.astype(np.float32, copy=False)
    if dtype == "float16":
        return quantize_float16(vec)
    if dtype == "int8":
        return quantize_int8(vec)
    raise QuantizationError(  # pragma: no cover - exhaustive
        f"Unknown dtype {dtype!r}. Remediation: use float32, float16, or int8."
    )


def dequantize(qvec: npt.NDArray[Any], dtype: VectorDtype) -> npt.NDArray[np.float32]:
    """Inverse of ``quantize``: returns float32."""
    if dtype == "float32":
        return qvec.astype(np.float32, copy=False)
    if dtype == "float16":
        return dequantize_float16(qvec)
    if dtype == "int8":
        return dequantize_int8(qvec)
    raise QuantizationError(  # pragma: no cover - exhaustive
        f"Unknown dtype {dtype!r}."
    )


def memory_bytes_estimate(n_entries: int, dim: int, dtype: VectorDtype) -> int:
    """Bytes needed for the in-memory matrix at the given size and dtype.

    Excludes Python object overhead and auxiliary arrays.
    """
    return n_entries * dim * bytes_per_element(dtype)


__all__ = [
    "bytes_per_element",
    "dequantize",
    "dequantize_float16",
    "dequantize_int8",
    "memory_bytes_estimate",
    "np_dtype_for",
    "quantize",
    "quantize_float16",
    "quantize_int8",
]
