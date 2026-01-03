"""
Exact symbolic solution strategies.

These strategies preserve exact closed-form analytical solutions through
algebraic transformations of the original equation.

All strategies in this folder have solution_type='exact_symbolic'.
"""

from src.data.augmentations.exact_symbolic.compose import ComposeAugmentation
from src.data.augmentations.exact_symbolic.scale import ScaleAugmentation
from src.data.augmentations.exact_symbolic.shift import ShiftAugmentation
from src.data.augmentations.exact_symbolic.substitute import SubstituteAugmentation

__all__ = [
    "SubstituteAugmentation",
    "ScaleAugmentation",
    "ShiftAugmentation",
    "ComposeAugmentation",
]
