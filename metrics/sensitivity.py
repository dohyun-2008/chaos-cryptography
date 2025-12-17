"""Sensitivity analysis for encrypted images."""

import numpy as np
from typing import Dict


def calculate_sensitivity(image1: np.ndarray, image2: np.ndarray) -> Dict[str, float]:
    """Calculate pixel change rate and Hamming distance between two images."""
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape for sensitivity calculation.")

    diff = image1 != image2
    pixel_change_rate = np.sum(diff) / diff.size * 100

    flat1 = image1.flatten()
    flat2 = image2.flatten()
    hamming_distance = int(np.sum(flat1 != flat2))

    return {
        'pixel_change_rate': pixel_change_rate,
        'hamming_distance': hamming_distance
    }
