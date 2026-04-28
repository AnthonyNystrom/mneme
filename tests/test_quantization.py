"""Phase-4 tests: quantization correctness and accuracy bounds."""

from __future__ import annotations

import numpy as np
import pytest

from mneme._exceptions import QuantizationError
from mneme._quantization import (
    bytes_per_element,
    dequantize,
    dequantize_float16,
    dequantize_int8,
    memory_bytes_estimate,
    np_dtype_for,
    quantize,
    quantize_float16,
    quantize_int8,
)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


# --- Helpers / dtype mapping ---


def test_np_dtype_for_known_dtypes():
    assert np_dtype_for("float32") == np.dtype(np.float32)
    assert np_dtype_for("float16") == np.dtype(np.float16)
    assert np_dtype_for("int8") == np.dtype(np.int8)


def test_np_dtype_for_unknown_raises():
    with pytest.raises(QuantizationError, match="Unknown"):
        np_dtype_for("float64")  # type: ignore[arg-type]


def test_bytes_per_element():
    assert bytes_per_element("float32") == 4
    assert bytes_per_element("float16") == 2
    assert bytes_per_element("int8") == 1


def test_memory_bytes_estimate():
    # 100k entries * 768 dim * 4 bytes = 307_200_000
    assert memory_bytes_estimate(100_000, 768, "float32") == 307_200_000
    assert memory_bytes_estimate(100_000, 768, "float16") == 153_600_000
    assert memory_bytes_estimate(100_000, 768, "int8") == 76_800_000


# --- float16 round-trip ---


def test_float16_round_trip_within_tolerance():
    rng = np.random.default_rng(0)
    v32 = rng.standard_normal(768).astype(np.float32)
    v32 = _l2_normalize(v32)
    v16 = quantize_float16(v32)
    rt = dequantize_float16(v16)
    assert rt.dtype == np.float32
    # < 0.1% drift on per-element comparison for L2-normalized inputs.
    np.testing.assert_allclose(rt, v32, atol=1e-3, rtol=1e-3)


def test_float16_dispatch_via_quantize():
    v32 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    qv = quantize(v32, "float16")
    assert qv.dtype == np.float16
    rt = dequantize(qv, "float16")
    assert rt.dtype == np.float32


# --- int8 round-trip ---


def test_int8_round_trip_within_tolerance():
    rng = np.random.default_rng(1)
    v32 = rng.standard_normal(768).astype(np.float32)
    v32 = _l2_normalize(v32)
    qv = quantize_int8(v32)
    assert qv.dtype == np.int8
    rt = dequantize_int8(qv)
    # < 1/127 ≈ 0.79% per-element drift on L2-normalized inputs.
    np.testing.assert_allclose(rt, v32, atol=1.0 / 127.0)


def test_int8_normalized_input_no_clip():
    """Values in [-1, 1] (post-L2-normalization) round-trip without saturating."""
    v32 = np.array([1.0, -1.0, 0.5, -0.5, 0.0], dtype=np.float32)
    qv = quantize_int8(v32)
    # 1.0 * 127 → 127 (max int8 positive)
    # -1.0 * 127 → -127 (within int8 range)
    assert qv[0] == 127
    assert qv[1] == -127
    assert qv[2] == round(0.5 * 127)
    assert qv[3] == round(-0.5 * 127)
    assert qv[4] == 0


def test_int8_clips_when_out_of_range():
    """Defensive: out-of-[-1,1] values clip rather than overflow."""
    v32 = np.array([2.0, -2.0, 1.5, -1.5], dtype=np.float32)
    qv = quantize_int8(v32)
    assert qv[0] == 127  # clipped from round(2.0 * 127) = 254
    assert qv[1] == -128  # clipped from -254
    assert qv[2] == 127
    assert qv[3] == -128


def test_int8_dispatch_via_quantize():
    v32 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    qv = quantize(v32, "int8")
    assert qv.dtype == np.int8
    rt = dequantize(qv, "int8")
    assert rt.dtype == np.float32


# --- Cosine-similarity drift across dtypes ---


def test_quantization_threshold_drift_int8_within_3_percent():
    """For L2-normalized random vectors, int8 cosine drift < 3% on average."""
    rng = np.random.default_rng(42)
    drifts = []
    for _ in range(200):
        a = rng.standard_normal(768).astype(np.float32)
        b = rng.standard_normal(768).astype(np.float32)
        a = _l2_normalize(a)
        b = _l2_normalize(b)
        sim_f32 = float(a @ b)
        a8 = dequantize_int8(quantize_int8(a))
        b8 = dequantize_int8(quantize_int8(b))
        sim_i8 = float(a8 @ b8)
        drifts.append(abs(sim_f32 - sim_i8))
    avg_drift = float(np.mean(drifts))
    assert avg_drift < 0.03, f"int8 drift {avg_drift:.4f} exceeds 3%"


def test_quantization_threshold_drift_float16_within_0_1_percent():
    rng = np.random.default_rng(7)
    drifts = []
    for _ in range(200):
        a = rng.standard_normal(768).astype(np.float32)
        b = rng.standard_normal(768).astype(np.float32)
        a = _l2_normalize(a)
        b = _l2_normalize(b)
        sim_f32 = float(a @ b)
        a16 = dequantize_float16(quantize_float16(a))
        b16 = dequantize_float16(quantize_float16(b))
        sim_f16 = float(a16 @ b16)
        drifts.append(abs(sim_f32 - sim_f16))
    avg_drift = float(np.mean(drifts))
    assert avg_drift < 0.001, f"float16 drift {avg_drift:.4f} exceeds 0.1%"


# --- Dispatch passthrough ---


def test_quantize_float32_is_no_op():
    v32 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = quantize(v32, "float32")
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, v32)


def test_dequantize_float32_is_no_op():
    v32 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = dequantize(v32, "float32")
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, v32)
