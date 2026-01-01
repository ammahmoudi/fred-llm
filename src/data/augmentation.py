"""
Data augmentation for Fredholm integral equations.

Generates variations of equations for training data, including edge cases.
"""

from typing import Any

import sympy as sp

from src.data.augmentations import (
    ApproximateOnlyAugmentation,
    BoundaryLayerAugmentation,
    CompactSupportAugmentation,
    ComposeAugmentation,
    DivergentKernelAugmentation,
    IllPosedAugmentation,
    MixedTypeAugmentation,
    NoSolutionAugmentation,
    OscillatorySolutionAugmentation,
    RangeViolationAugmentation,
    ResonanceAugmentation,
    ScaleAugmentation,
    ShiftAugmentation,
    SubstituteAugmentation,
    WeaklySingularAugmentation,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def augment_dataset(
    data: list[dict[str, Any]],
    strategies: list[str] | None = None,
    multiplier: float = 2.0,
) -> list[dict[str, Any]]:
    """
    Augment a dataset with synthetic variations.

    Args:
        data: Original dataset.
        strategies: Augmentation strategies to apply.
        multiplier: Target size multiplier (e.g., 1.15 for 15% augmentation).
            Recommended: 1.1-1.2 for 11 strategies, 1.25-1.33 for 3 strategies.

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

    # Generate augmented variants up to target size
    target_size = int(len(data) * multiplier)
    for item in data:
        if len(augmented) >= target_size:
            break
        for strategy in strategies:
            if len(augmented) >= target_size:
                break
            try:
                new_items = _apply_augmentation(item, strategy)
                augmented.extend(new_items)
            except Exception as e:
                logger.debug(f"Augmentation failed for {strategy}: {e}")

    logger.info(f"Augmented dataset from {len(data)} to {len(augmented)} samples")
    return augmented


def _apply_augmentation(
    item: dict[str, Any],
    strategy: str,
) -> list[dict[str, Any]]:
    """Apply a single augmentation strategy or strategy group."""
    # Basic transformations
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

    # Solution-type based strategies (run all in folder)
    elif strategy == "no_solution":
        # Run all no-solution strategies
        results = []
        results.extend(NoSolutionAugmentation().augment(item))
        results.extend(RangeViolationAugmentation().augment(item))
        results.extend(DivergentKernelAugmentation().augment(item))
        return results

    elif strategy == "numerical_only":
        # Run all numerical-only strategies
        results = []
        results.extend(ApproximateOnlyAugmentation(num_sample_points=10).augment(item))
        results.extend(WeaklySingularAugmentation(num_sample_points=15).augment(item))
        results.extend(
            BoundaryLayerAugmentation(epsilon=0.01, num_sample_points=20).augment(item)
        )
        results.extend(
            OscillatorySolutionAugmentation(
                base_frequency=10.0, num_sample_points=100
            ).augment(item)
        )
        results.extend(MixedTypeAugmentation().augment(item))
        results.extend(CompactSupportAugmentation(bandwidth=0.1).augment(item))
        return results

    elif strategy == "regularization_required":
        # Run all regularization-required strategies
        augmenter = IllPosedAugmentation(
            num_sample_points=10, regularization_param=0.01
        )
        return augmenter.augment(item)

    elif strategy == "non_unique_solution":
        # Run all non-unique solution strategies
        augmenter = ResonanceAugmentation(perturbation=0.001)
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
            strategies: Augmentation strategies to use (folder-based):

                Basic transformations (untested):
                - substitute, scale, shift, compose

                Solution-type folders (each runs all strategies in that folder):
                - no_solution: Runs eigenvalue_cases + range_violation + divergent_kernel (9 variants)
                - numerical_only: Runs complex_kernels + weakly_singular + boundary_layer +
                                 oscillatory_solution + mixed_type + compact_support (18 variants)
                - regularization_required: Runs ill_posed (3 variants)
                - non_unique_solution: Runs resonance (3 variants)

            seed: Random seed for reproducibility.
        """
        self.strategies = strategies or ["substitute", "scale", "shift"]
        self.seed = seed

    def augment(
        self,
        data: list[dict[str, Any]],
        multiplier: float = 2.0,
    ) -> list[dict[str, Any]]:
        """Augment the dataset.

        Args:
            data: Original dataset.
            multiplier: Target size multiplier (e.g., 1.15, 1.33).

        Returns:
            Augmented dataset with originals + generated variants.
        """
        return augment_dataset(data, self.strategies, multiplier)
