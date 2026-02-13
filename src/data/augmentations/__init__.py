"""
Augmentation strategies for Fredholm integral equations.

Strategies are organized by solution type (7 types total):

## Folder Structure (by Solution Type)

- **exact_symbolic/**: Closed-form analytical solutions (substitute, scale, shift, compose)
- **approx_coef/**: Functional forms with numerical params (boundary_layer, oscillatory, etc.)
- **discrete_points/**: Pure point samples (complex_kernels, near_resonance)
- **series/**: Truncated series expansions (neumann_series)
- **family/**: Non-unique solution families (resonance)
- **regularized/**: Ill-posed equations (ill_posed)
- **none_solution/**: No solution exists (eigenvalue_cases, range_violation, etc.)

## Solution Type Mapping

| Solution Type | Folder | Count |
|---------------|--------|-------|
| exact_symbolic | exact_symbolic/ | 4 strategies |
| approx_coef | approx_coef/ | 5 strategies |
| discrete_points | discrete_points/ | 2 strategies |
| series | series/ | 1 strategy |
| family | family/ | 1 strategy |
| regularized | regularized/ | 1 strategy |
| none | none_solution/ | 4 strategies |

**Total: 18 strategies (4 basic + 14 edge cases)**
"""

from src.data.augmentations.approx_coef import (
    BoundaryLayerAugmentation,
    CompactSupportAugmentation,
    MixedTypeAugmentation,
    OscillatorySolutionAugmentation,
    WeaklySingularAugmentation,
)
from src.data.augmentations.base import BaseAugmentation
from src.data.augmentations.discrete_points import (
    ApproximateOnlyAugmentation,
    NearResonanceAugmentation,
)
from src.data.augmentations.family import ResonanceAugmentation
from src.data.augmentations.none_solution import (
    DisconnectedSupportAugmentation,
    DivergentKernelAugmentation,
    NoSolutionAugmentation,
    RangeViolationAugmentation,
)
from src.data.augmentations.regularized import IllPosedAugmentation
from src.data.augmentations.series import NeumannSeriesAugmentation

__all__ = [
    "BaseAugmentation",
    # Approx coef (5)
    "BoundaryLayerAugmentation",
    "OscillatorySolutionAugmentation",
    "WeaklySingularAugmentation",
    "MixedTypeAugmentation",
    "CompactSupportAugmentation",
    # Discrete points (2)
    "ApproximateOnlyAugmentation",
    "NearResonanceAugmentation",
    # Series (1)
    "NeumannSeriesAugmentation",
    # Family (1)
    "ResonanceAugmentation",
    # Regularized (1)
    "IllPosedAugmentation",
    # None (4)
    "NoSolutionAugmentation",
    "RangeViolationAugmentation",
    "DivergentKernelAugmentation",
    "DisconnectedSupportAugmentation",
]
