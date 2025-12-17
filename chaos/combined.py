"""Hybrid chaotic map generators mixing Logistic and Tent maps."""

import numpy as np
from typing import Dict, Optional
from .logistic import generate_logistic
from .tent import generate_tent


def generate_interleaved(
    length: int,
    x0: float,
    params: Optional[Dict] = None
) -> np.ndarray:
    """Interleave Logistic and Tent maps with periodic cross-feedback."""
    if params is None:
        params = {}

    logistic_r = params.get('logistic_r', 3.99)
    tent_r = params.get('tent_r', 1.99)

    if not (0 < x0 < 1):
        raise ValueError(f"x0 must be in (0, 1), got {x0}")

    sequence = np.zeros(length)
    x = np.clip(x0, 1e-10, 1.0 - 1e-10)
    prev_logistic = x
    prev_tent = x

    for i in range(length):
        sequence[i] = x
        if i % 2 == 0:
            x = logistic_r * x * (1 - x)
            if x < 0:
                x = abs(x) % 1.0
            elif x > 1.0:
                x = x % 1.0
            x = np.clip(x, 1e-10, 1.0 - 1e-10)
            if i > 0 and i % 10 == 0:
                x = 0.95 * x + 0.05 * prev_tent
        else:
            x_clipped = np.clip(x, 0.0, 1.0)
            if x_clipped < 0.5:
                x = tent_r * x_clipped
            else:
                x = tent_r * (1 - x_clipped)
            if x >= 1.0:
                x = x % 1.0
            x = np.clip(x, 1e-10, 1.0 - 1e-10)
            if i > 1 and i % 10 == 1:
                x = 0.95 * x + 0.05 * prev_logistic

        if i % 2 == 0:
            prev_logistic = x
        else:
            prev_tent = x

    return sequence


def generate_cascade(
    length: int,
    x0: float,
    params: Optional[Dict] = None
) -> np.ndarray:
    """Apply Logistic then Tent map sequentially with light perturbations."""
    if params is None:
        params = {}

    logistic_r = params.get('logistic_r', 3.99)
    tent_r = params.get('tent_r', 1.99)

    if not (0 < x0 < 1):
        raise ValueError(f"x0 must be in (0, 1), got {x0}")

    sequence = np.zeros(length)
    x = np.clip(x0, 1e-10, 1.0 - 1e-10)

    for i in range(length):
        sequence[i] = x

        x_log = logistic_r * x * (1 - x)
        if x_log < 0:
            x_log = abs(x_log) % 1.0
        elif x_log > 1.0:
            x_log = x_log % 1.0
        x_log = np.clip(x_log, 1e-10, 1.0 - 1e-10)

        x_tent_input = np.clip(x_log, 0.0, 1.0)
        if x_tent_input < 0.5:
            x_tent = tent_r * x_tent_input
        else:
            x_tent = tent_r * (1 - x_tent_input)
        if x_tent >= 1.0:
            x_tent = x_tent % 1.0
        x_tent = np.clip(x_tent, 1e-10, 1.0 - 1e-10)

        x = 1.0 - x_tent if i % 3 == 0 else x_tent
        if x < 0:
            x = abs(x) % 1.0
        elif x >= 1.0:
            x = x % 1.0
        x = np.clip(x, 1e-10, 1.0 - 1e-10)

        if i > 0 and i % 100 == 0:
            perturbation = ((i // 100) % 1000) * 1e-10
            x = x + perturbation if (i // 100) % 2 == 0 else x - perturbation
            x = np.clip(x, 1e-10, 1.0 - 1e-10)

    return sequence
