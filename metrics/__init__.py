"""Security metrics for evaluating encryption quality."""

from .entropy import calculate_entropy
from .correlation import calculate_correlation
from .npcr import calculate_npcr
from .sensitivity import calculate_sensitivity

__all__ = [
    'calculate_entropy',
    'calculate_correlation',
    'calculate_npcr',
    'calculate_sensitivity',
]
