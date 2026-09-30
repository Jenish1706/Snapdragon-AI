"""Q4_0 and Q8_0 fake-quantization roundtrips per 32-element block."""

from __future__ import annotations

import numpy as np


def q4_0_roundtrip(w: np.ndarray) -> np.ndarray:
    """Quantize to 4-bit symmetric (-8 to 7) per 32-element block, then dequantize."""
    orig_shape = w.shape
    flat = w.reshape(-1, 32)
    max_val = np.max(np.abs(flat), axis=1, keepdims=True)
    scale = max_val / 8.0
    scale[scale == 0] = 1.0
    quant = np.clip(np.round(flat / scale), -8, 7)
    dequant = quant * scale
    return dequant.reshape(orig_shape).astype(np.float32)


def q8_0_roundtrip(w: np.ndarray) -> np.ndarray:
    """Quantize to 8-bit symmetric (-128 to 127) per 32-element block, then dequantize."""
    orig_shape = w.shape
    flat = w.reshape(-1, 32)
    max_val = np.max(np.abs(flat), axis=1, keepdims=True)
    scale = max_val / 127.0
    scale[scale == 0] = 1.0
    quant = np.clip(np.round(flat / scale), -128, 127)
    dequant = quant * scale
    return dequant.reshape(orig_shape).astype(np.float32)
