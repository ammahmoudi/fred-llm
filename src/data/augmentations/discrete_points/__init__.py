"""
Discrete points solution strategies.

These strategies generate equations with solutions represented as
discrete point samples only (no functional form).

The u field is empty, solutions provided as sample_points arrays.
All strategies have solution_type='discrete_points'.
"""

from src.data.augmentations.discrete_points.complex_kernels import (
    ApproximateOnlyAugmentation,
)
from src.data.augmentations.discrete_points.near_resonance import (
    NearResonanceAugmentation,
)

__all__ = [
    "ApproximateOnlyAugmentation",
    "NearResonanceAugmentation",
]
