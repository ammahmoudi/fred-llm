"""
Approximate coefficient solution strategies.

These strategies generate equations with functional forms containing
numerical parameters (e.g., exp(-x/0.01), sin(100*pi*x)).

The solution has a formula but with numerical coefficients.
All strategies have solution_type='approx_coef'.
"""

from src.data.augmentations.approx_coef.boundary_layer import (
    BoundaryLayerAugmentation,
)
from src.data.augmentations.approx_coef.compact_support import (
    CompactSupportAugmentation,
)
from src.data.augmentations.approx_coef.mixed_type import MixedTypeAugmentation
from src.data.augmentations.approx_coef.oscillatory_solution import (
    OscillatorySolutionAugmentation,
)
from src.data.augmentations.approx_coef.weakly_singular import (
    WeaklySingularAugmentation,
)

__all__ = [
    "BoundaryLayerAugmentation",
    "OscillatorySolutionAugmentation",
    "WeaklySingularAugmentation",
    "MixedTypeAugmentation",
    "CompactSupportAugmentation",
]
