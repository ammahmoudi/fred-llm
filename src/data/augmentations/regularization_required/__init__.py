"""
Regularization-required augmentation strategies.

These strategies create ill-posed equations requiring regularization techniques.
"""

from src.data.augmentations.regularization_required.ill_posed import (
    IllPosedAugmentation,
)

__all__ = [
    "IllPosedAugmentation",
]
