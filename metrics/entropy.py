"""Shannon entropy calculation for images."""

import numpy as np


def calculate_entropy(image: np.ndarray) -> float:
    """Calculate Shannon entropy of an image."""
    if len(image.shape) == 3:
        entropies = []
        for channel in range(image.shape[2]):
            flat = image[:, :, channel].flatten()
            entropies.append(_entropy_1d(flat))
        return np.mean(entropies)
    flat = image.flatten()
    return _entropy_1d(flat)


def _entropy_1d(data: np.ndarray) -> float:
    """Calculate Shannon entropy for a 1D array."""
    value_counts = np.bincount(data, minlength=256)
    probabilities = value_counts / len(data)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)
