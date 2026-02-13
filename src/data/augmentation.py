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
    DisconnectedSupportAugmentation,
    DivergentKernelAugmentation,
    IllPosedAugmentation,
    MixedTypeAugmentation,
    NearResonanceAugmentation,
    NeumannSeriesAugmentation,
    NoSolutionAugmentation,
    OscillatorySolutionAugmentation,
    RangeViolationAugmentation,
    ResonanceAugmentation,
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
            Recommended: 1.1-1.2 for 14 edge case strategies.

    Returns:
        Augmented dataset.
    """
    if strategies is None:
        strategies = [
            "none_solution",
            "approx_coef",
            "discrete_points",
        ]  # All edge case strategies

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
            original_item["solution_type"] = "exact_symbolic"
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
    # Solution-type based strategies (run all in folder)
    if strategy == "none_solution":
        # Run all none-solution strategies
        results = []
        results.extend(NoSolutionAugmentation().augment(item))
        results.extend(RangeViolationAugmentation().augment(item))
        results.extend(DivergentKernelAugmentation().augment(item))
        results.extend(DisconnectedSupportAugmentation().augment(item))
        return results

    elif strategy == "approx_coef":
        # Run all approx_coef strategies (functional form with numerical params)
        results = []
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

    elif strategy == "discrete_points":
        # Run all discrete_points strategies (pure sample arrays, no functional form)
        results = []
        results.extend(ApproximateOnlyAugmentation(num_sample_points=10).augment(item))
        results.extend(NearResonanceAugmentation(distance=0.1).augment(item))
        return results

    elif strategy == "series":
        # Run all series strategies (truncated series expansions)
        augmenter = NeumannSeriesAugmentation(num_terms=4, lambda_scale=0.3)
        return augmenter.augment(item)

    elif strategy == "regularized":
        # Run all regularized strategies (ill-posed equations)
        augmenter = IllPosedAugmentation(
            num_sample_points=10, regularization_param=0.01
        )
        return augmenter.augment(item)

    elif strategy == "family":
        # Run all family strategies (non-unique solutions)
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
            strategies: Edge case augmentation strategies to use:

                - none_solution: No solution (eigenvalue_cases, range_violation, divergent_kernel, disconnected_support) - 4 strategies
                - approx_coef: Functional form with numerical params (weakly_singular, boundary_layer, oscillatory_solution, mixed_type, compact_support) - 5 strategies
                - discrete_points: Pure sample arrays (approximate_only, near_resonance) - 2 strategies
                - series: Truncated series (neumann_series) - 1 strategy
                - regularized: Ill-posed equations (ill_posed) - 1 strategy
                - family: Non-unique solutions (resonance) - 1 strategy

                Total: 14 edge case strategies

            seed: Random seed for reproducibility.
        """
        self.strategies = strategies or [
            "none_solution",
            "approx_coef",
            "discrete_points",
        ]
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
