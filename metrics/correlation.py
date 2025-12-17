"""Correlation coefficient metrics for images."""

import numpy as np
from typing import Dict


def calculate_correlation(
    image: np.ndarray,
    num_samples: int = 5000
) -> Dict[str, float]:
    """Calculate correlation coefficients in horizontal, vertical, and diagonal directions."""
    if len(image.shape) == 3:
        correlations = {'horizontal': [], 'vertical': [], 'diagonal': []}
        for channel in range(image.shape[2]):
            channel_img = image[:, :, channel]
            corr = _correlation_2d(channel_img, num_samples)
            correlations['horizontal'].append(corr['horizontal'])
            correlations['vertical'].append(corr['vertical'])
            correlations['diagonal'].append(corr['diagonal'])
        return {
            'horizontal': np.mean(correlations['horizontal']),
            'vertical': np.mean(correlations['vertical']),
            'diagonal': np.mean(correlations['diagonal'])
        }
    return _correlation_2d(image, num_samples)


def _correlation_2d(image: np.ndarray, num_samples: int) -> Dict[str, float]:
    """Correlation calculation for a 2D grayscale image."""
    h, w = image.shape
    image_float = image.astype(np.float64)

    np.random.seed(42)
    samples = min(num_samples, (h - 1) * (w - 1))

    h_pairs = _sample_pairs(h, w, samples, direction='horizontal')
    h_corr = _compute_correlation(image_float, h_pairs)

    v_pairs = _sample_pairs(h, w, samples, direction='vertical')
    v_corr = _compute_correlation(image_float, v_pairs)

    d_pairs = _sample_pairs(h, w, samples, direction='diagonal')
    d_corr = _compute_correlation(image_float, d_pairs)

    return {
        'horizontal': float(h_corr),
        'vertical': float(v_corr),
        'diagonal': float(d_corr)
    }


def _sample_pairs(h: int, w: int, num_samples: int, direction: str) -> np.ndarray:
    """Sample pixel pairs in the specified direction."""
    pairs = []

    if direction == 'horizontal':
        max_i, max_j = h, w - 1
        for _ in range(num_samples):
            i = np.random.randint(0, max_i)
            j = np.random.randint(0, max_j)
            pairs.append([[i, j], [i, j + 1]])
    elif direction == 'vertical':
        max_i, max_j = h - 1, w
        for _ in range(num_samples):
            i = np.random.randint(0, max_i)
            j = np.random.randint(0, max_j)
            pairs.append([[i, j], [i + 1, j]])
    elif direction == 'diagonal':
        max_i, max_j = h - 1, w - 1
        for _ in range(num_samples):
            i = np.random.randint(0, max_i)
            j = np.random.randint(0, max_j)
            pairs.append([[i, j], [i + 1, j + 1]])

    return np.array(pairs)


def _compute_correlation(image: np.ndarray, pairs: np.ndarray) -> float:
    """Compute correlation coefficient from pixel pairs."""
    x_vals = []
    y_vals = []

    for pair in pairs:
        r1, c1 = pair[0]
        r2, c2 = pair[1]
        x_vals.append(image[r1, c1])
        y_vals.append(image[r2, c2])

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    x_mean = np.mean(x_vals)
    y_mean = np.mean(y_vals)

    numerator = np.sum((x_vals - x_mean) * (y_vals - y_mean))
    x_std = np.std(x_vals)
    y_std = np.std(y_vals)

    if x_std == 0 or y_std == 0:
        return 0.0

    correlation = numerator / (len(x_vals) * x_std * y_std)
    return correlation
