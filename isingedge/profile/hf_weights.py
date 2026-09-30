"""HuggingFace safetensors header parsing and BF16 widening utilities."""

from __future__ import annotations

import json
import struct
import numpy as np


def bf16_to_float32(raw: np.ndarray) -> np.ndarray:
    """Widen 16-bit bfloat16 representation to 32-bit single-precision float."""
    u32 = raw.astype(np.uint32) << 16
    return u32.view(np.float32)


def parse_header(content: bytes) -> tuple[dict, int]:
    """Parse safetensors binary header: 8-byte uint64 length followed by JSON header."""
    header_len = struct.unpack("<Q", content[:8])[0]
    header_json = content[8: 8 + header_len].decode("utf-8")
    header = json.loads(header_json)
    return header, 8 + header_len
