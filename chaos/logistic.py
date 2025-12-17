"""Logistic map chaotic sequence utilities."""

import numpy as np
from typing import Dict, Optional


def generate_logistic(
    length: int,
    x0: float,
    params: Optional[Dict] = None
) -> np.ndarray:
    """Generate a chaotic sequence using the Logistic map."""
    if params is None:
        params = {}

    r = params.get('r', 3.99)
    if not (0 < x0 < 1):
        raise ValueError(f"x0 must be in (0, 1), got {x0}")
    if not (3.57 <= r <= 4.0):
        raise ValueError(f"r must be in [3.57, 4.0], got {r}")

    sequence = np.zeros(length)
    x = x0

    for i in range(length):
        sequence[i] = x
        x = r * x * (1 - x)

    return sequence


def to_bits(sequence: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a chaotic sequence to binary bits."""
    return (sequence >= threshold).astype(np.uint8)


def to_bytes(sequence: np.ndarray, bits_per_byte: int = 8) -> np.ndarray:
    """Convert a chaotic sequence to byte values."""
    max_val = 2 ** bits_per_byte - 1
    quantized = (sequence * max_val).astype(np.uint32)

    num_bytes = len(quantized) // bits_per_byte
    if num_bytes == 0:
        return np.array([], dtype=np.uint8)

    bytes_array = np.zeros(num_bytes, dtype=np.uint8)
    for i in range(num_bytes):
        byte_val = 0
        for j in range(bits_per_byte):
            idx = i * bits_per_byte + j
            if idx < len(quantized):
                bit = quantized[idx] % 2
                byte_val |= (bit << j)
        bytes_array[i] = byte_val

    return bytes_array
