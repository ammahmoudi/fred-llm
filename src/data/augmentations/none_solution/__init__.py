"""
No solution strategies.

These strategies generate equations with no solution
(violating Fredholm Alternative, range violations, etc.).

The u field is empty, has_solution=False.
All strategies have solution_type='none'.
"""

from src.data.augmentations.none_solution.disconnected_support import (
    DisconnectedSupportAugmentation,
)
from src.data.augmentations.none_solution.divergent_kernel import (
    DivergentKernelAugmentation,
)
from src.data.augmentations.none_solution.eigenvalue_cases import NoSolutionAugmentation
from src.data.augmentations.none_solution.range_violation import (
    RangeViolationAugmentation,
)

__all__ = [
    "NoSolutionAugmentation",
    "RangeViolationAugmentation",
    "DivergentKernelAugmentation",
    "DisconnectedSupportAugmentation",
]
