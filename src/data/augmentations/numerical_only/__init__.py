"""
Numerical-only solution augmentation strategies.

These strategies create equations requiring numerical methods (no closed-form):
- Complex kernels (no symbolic antiderivative)
- Weakly singular kernels (integrable singularities)
- Boundary layer solutions (sharp gradients)
- Oscillatory solutions (high-frequency components)
- Mixed-type equations (Volterra + Fredholm)
- Compact support kernels (sparse structure)
"""

from src.data.augmentations.numerical_only.boundary_layer import \
    BoundaryLayerAugmentation
from src.data.augmentations.numerical_only.compact_support import \
    CompactSupportAugmentation
from src.data.augmentations.numerical_only.complex_kernels import \
    ApproximateOnlyAugmentation
from src.data.augmentations.numerical_only.mixed_type import \
    MixedTypeAugmentation
from src.data.augmentations.numerical_only.oscillatory_solution import \
    OscillatorySolutionAugmentation
from src.data.augmentations.numerical_only.weakly_singular import \
    WeaklySingularAugmentation

__all__ = [
    "ApproximateOnlyAugmentation",
    "WeaklySingularAugmentation",
    "BoundaryLayerAugmentation",
    "OscillatorySolutionAugmentation",
    "MixedTypeAugmentation",
    "CompactSupportAugmentation",
]
