"""
No solution augmentation strategies.

These strategies create equations where no solution exists due to:
- Eigenvalue conditions (Fredholm Alternative violated)
- Range violations (RHS not in operator range)
- Divergent kernels (non-integrable singularities)
- Disconnected support (rank-deficient operators)
"""

from src.data.augmentations.no_solution.disconnected_support import (
    DisconnectedSupportAugmentation,
)
from src.data.augmentations.no_solution.divergent_kernel import (
    DivergentKernelAugmentation,
)
from src.data.augmentations.no_solution.eigenvalue_cases import NoSolutionAugmentation
from src.data.augmentations.no_solution.range_violation import (
    RangeViolationAugmentation,
)

__all__ = [
    "NoSolutionAugmentation",
    "RangeViolationAugmentation",
    "DivergentKernelAugmentation",
    "DisconnectedSupportAugmentation",
]
