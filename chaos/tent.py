"""Tent map chaotic sequence utilities."""

import numpy as np
from typing import Dict, Optional


def generate_tent(
    length: int,
    x0: float,
    params: Optional[Dict] = None
) -> np.ndarray:
    """Generate a chaotic sequence using the Tent map."""
    if params is None:
        params = {}

    r = params.get('r', 2.0)
    if not (0 <= x0 <= 1):
        raise ValueError(f"x0 must be in [0, 1], got {x0}")
    if not (1.8 <= r <= 2.0):
        raise ValueError(f"r must be in [1.8, 2.0], got {r}")

    sequence = np.zeros(length)
    x = x0
    if abs(x - 0.5) < 1e-10:
        x = 0.5 + 1e-8
    x = np.clip(x, 1e-10, 1.0 - 1e-10)

    for i in range(length):
        sequence[i] = x
        if x < 0.5:
            x = r * x
        else:
            x = r * (1 - x)
        if x >= 1.0:
            x = x % 1.0
        if x < 1e-6:
            offset = ((i * 7 + 13) % 1000) * 1e-9
            x = 1e-6 + offset
        elif x > 1.0 - 1e-6:
            offset = ((i * 7 + 13) % 1000) * 1e-9
            x = 1.0 - 1e-6 - offset
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        if i > 0 and i % 100 == 0:
            perturbation = ((i // 100) % 1000) * 1e-10
            x = x + perturbation if (i // 100) % 2 == 0 else x - perturbation
            x = np.clip(x, 1e-6, 1.0 - 1e-6)

    return sequence
