"""
Augmentation strategies for Fredholm integral equations.

Provides specialized augmentations including:
- Basic transformations (substitute, scale, shift, compose)
- Edge cases (no-solution, approximate-only, ill-posed)
"""

from src.data.augmentations.approximate_only import ApproximateOnlyAugmentation
from src.data.augmentations.base import BaseAugmentation
from src.data.augmentations.compose import ComposeAugmentation
from src.data.augmentations.ill_posed import IllPosedAugmentation
from src.data.augmentations.no_solution import NoSolutionAugmentation
from src.data.augmentations.scale import ScaleAugmentation
from src.data.augmentations.shift import ShiftAugmentation
from src.data.augmentations.substitute import SubstituteAugmentation

__all__ = [
    "BaseAugmentation",
    "SubstituteAugmentation",
    "ScaleAugmentation",
    "ShiftAugmentation",
    "ComposeAugmentation",
    "NoSolutionAugmentation",
    "ApproximateOnlyAugmentation",
    "IllPosedAugmentation",
]
