"""
Numerical-only solution augmentation strategies.

These strategies create equations requiring numerical methods (no closed-form):
- Complex kernels (no symbolic antiderivative) → discrete_points
- Weakly singular kernels (integrable singularities) → approx_coef
- Boundary layer solutions (sharp gradients) → approx_coef
- Oscillatory solutions (high-frequency components) → approx_coef
- Mixed-type equations (Volterra + Fredholm) → approx_coef
- Compact support kernels (sparse structure) → approx_coef
- Near resonance (ill-conditioned) → discrete_points
- Neumann series (iterative expansion) → series
"""

from src.data.augmentations.numerical_only.boundary_layer import (
    BoundaryLayerAugmentation,
)
from src.data.augmentations.numerical_only.compact_support import (
    CompactSupportAugmentation,
)
from src.data.augmentations.numerical_only.complex_kernels import (
    ApproximateOnlyAugmentation,
)
from src.data.augmentations.numerical_only.mixed_type import MixedTypeAugmentation
from src.data.augmentations.numerical_only.near_resonance import (
    NearResonanceAugmentation,
)
from src.data.augmentations.numerical_only.neumann_series import (
    NeumannSeriesAugmentation,
)
from src.data.augmentations.numerical_only.oscillatory_solution import (
    OscillatorySolutionAugmentation,
)
from src.data.augmentations.numerical_only.weakly_singular import (
    WeaklySingularAugmentation,
)

__all__ = [
    "ApproximateOnlyAugmentation",
    "WeaklySingularAugmentation",
    "BoundaryLayerAugmentation",
    "OscillatorySolutionAugmentation",
    "MixedTypeAugmentation",
    "CompactSupportAugmentation",
    "NearResonanceAugmentation",
    "NeumannSeriesAugmentation",
]
