"""
Augmentation strategies for Fredholm integral equations.

Organized by solution type:
- no_solution: Equations with no solution (eigenvalue, range, divergent)
- numerical_only: Numerical approximation required (complex, singular, oscillatory)
- regularization_required: Ill-posed problems needing regularization
- non_unique_solution: Solution families (resonance)
- Basic transformations: substitute, scale, shift, compose (untested)
"""

from src.data.augmentations.base import BaseAugmentation
from src.data.augmentations.compose import ComposeAugmentation

# Import from solution-type organized folders
from src.data.augmentations.no_solution import (
    DisconnectedSupportAugmentation,
    DivergentKernelAugmentation,
    NoSolutionAugmentation,
    RangeViolationAugmentation,
)
from src.data.augmentations.non_unique_solution import ResonanceAugmentation
from src.data.augmentations.numerical_only import (
    ApproximateOnlyAugmentation,
    BoundaryLayerAugmentation,
    CompactSupportAugmentation,
    MixedTypeAugmentation,
    NearResonanceAugmentation,
    OscillatorySolutionAugmentation,
    WeaklySingularAugmentation,
)
from src.data.augmentations.regularization_required import IllPosedAugmentation
from src.data.augmentations.scale import ScaleAugmentation
from src.data.augmentations.shift import ShiftAugmentation
from src.data.augmentations.substitute import SubstituteAugmentation

__all__ = [
    "BaseAugmentation",
    # Basic transformations (untested)
    "SubstituteAugmentation",
    "ScaleAugmentation",
    "ShiftAugmentation",
    "ComposeAugmentation",
    # No solution
    "NoSolutionAugmentation",
    "RangeViolationAugmentation",
    "DivergentKernelAugmentation",
    "DisconnectedSupportAugmentation",
    # Numerical only
    "ApproximateOnlyAugmentation",
    "WeaklySingularAugmentation",
    "BoundaryLayerAugmentation",
    "OscillatorySolutionAugmentation",
    "MixedTypeAugmentation",
    "CompactSupportAugmentation",
    "NearResonanceAugmentation",
    # Regularization required
    "IllPosedAugmentation",
    # Non-unique solution
    "ResonanceAugmentation",
]
