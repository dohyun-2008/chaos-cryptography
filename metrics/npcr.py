"""NPCR calculation utilities."""

import numpy as np


def calculate_npcr(image1: np.ndarray, image2: np.ndarray) -> float:
    """Calculate the NPCR (Number of Pixel Change Rate) between two images."""
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape for NPCR calculation.")

    diff = image1 != image2
    change_rate = np.sum(diff) / diff.size
    return change_rate * 100
