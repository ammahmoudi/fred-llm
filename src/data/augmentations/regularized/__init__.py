"""
Regularized solution strategies.

These strategies generate ill-posed equations requiring
regularization techniques (Tikhonov, TSVD, etc.).

The u field is empty, has_solution=True but unstable without regularization.
All strategies have solution_type='regularized'.
"""

from src.data.augmentations.regularized.ill_posed import IllPosedAugmentation

__all__ = [
    "IllPosedAugmentation",
]
