"""
Data augmentation for Fredholm integral equations.

Generates variations of equations for training data, including edge cases.
"""

from typing import Any

import sympy as sp

from src.data.augmentations import (
    ApproximateOnlyAugmentation,
    ComposeAugmentation,
    IllPosedAugmentation,
    NoSolutionAugmentation,
    ScaleAugmentation,
    ShiftAugmentation,
    SubstituteAugmentation,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def augment_dataset(
    data: list[dict[str, Any]],
    strategies: list[str] | None = None,
    multiplier: int = 2,
) -> list[dict[str, Any]]:
    """
    Augment a dataset with synthetic variations.

    Args:
        data: Original dataset.
        strategies: Augmentation strategies to apply.
        multiplier: Target size multiplier.

    Returns:
        Augmented dataset.
    """
    if strategies is None:
        strategies = ["substitute", "scale", "shift"]

    # Keep originals with metadata marking them as original
    augmented = []
    for item in data:
        original_item = item.copy()
        # Add metadata to mark as original
        if "augmented" not in original_item:
            original_item["augmented"] = False
        if "augmentation_type" not in original_item:
            original_item["augmentation_type"] = "original"
        if "has_solution" not in original_item:
            original_item["has_solution"] = True
        if "solution_type" not in original_item:
            original_item["solution_type"] = "exact"
        augmented.append(original_item)

    for item in data:
        for strategy in strategies:
            try:
                new_items = _apply_augmentation(item, strategy)
                augmented.extend(new_items)

                if len(augmented) >= len(data) * multiplier:
                    break
            except Exception as e:
                logger.debug(f"Augmentation failed for {strategy}: {e}")

    logger.info(f"Augmented dataset from {len(data)} to {len(augmented)} samples")
    return augmented


def _apply_augmentation(
    item: dict[str, Any],
    strategy: str,
) -> list[dict[str, Any]]:
    """Apply a single augmentation strategy."""
    if strategy == "substitute":
        augmenter = SubstituteAugmentation()
        return augmenter.augment(item)
    elif strategy == "scale":
        augmenter = ScaleAugmentation()
        return augmenter.augment(item)
    elif strategy == "shift":
        augmenter = ShiftAugmentation()
        return augmenter.augment(item)
    elif strategy == "compose":
        augmenter = ComposeAugmentation()
        return augmenter.augment(item)
    elif strategy == "no_solution":
        augmenter = NoSolutionAugmentation()
        return augmenter.augment(item)
    elif strategy == "approximate_only":
        augmenter = ApproximateOnlyAugmentation(num_sample_points=10)
        return augmenter.augment(item)
    elif strategy == "ill_posed":
        augmenter = IllPosedAugmentation(num_sample_points=10, regularization_param=0.01)
        return augmenter.augment(item)
    else:
        return []


class DataAugmenter:
    """Configurable data augmenter."""

    def __init__(
        self,
        strategies: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        """
        Initialize the augmenter.

        Args:
            strategies: Augmentation strategies to use. Available:
                - substitute: Variable substitutions (x -> x², 2x, etc.)
                - scale: Scale lambda coefficients
                - shift: Shift integration domain
                - compose: Compose kernels with simple functions
                - no_solution: Generate singular cases (λ is eigenvalue)
                - approximate_only: Generate cases requiring numerical methods
                - ill_posed: Generate Fredholm 1st kind equations
            seed: Random seed for reproducibility.
        """
        self.strategies = strategies or ["substitute", "scale", "shift"]
        self.seed = seed

    def augment(
        self,
        data: list[dict[str, Any]],
        multiplier: int = 2,
    ) -> list[dict[str, Any]]:
        """Augment the dataset."""
        return augment_dataset(data, self.strategies, multiplier)
