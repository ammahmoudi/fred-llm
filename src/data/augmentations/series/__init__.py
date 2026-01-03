"""
Series solution strategies.

These strategies generate truncated series expansions
(Neumann series, Taylor series, etc.) with first N terms.

The u field contains the series formula.
All strategies have solution_type='series'.
"""

from src.data.augmentations.series.neumann_series import NeumannSeriesAugmentation

__all__ = [
    "NeumannSeriesAugmentation",
]
