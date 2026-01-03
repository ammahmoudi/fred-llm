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
    elif strategy == "none_solution":
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
            strategies: Augmentation strategies to use:

                Basic transformations (exact_symbolic):
                - substitute, scale, shift, compose (4 strategies)

                Solution-type groups (each runs all strategies with that solution type):
                - none_solution: No solution (eigenvalue_cases, range_violation, divergent_kernel,
                        disconnected_support) - 12 variants
                - approx_coef: Functional form with numerical params (weakly_singular,
                              boundary_layer, oscillatory_solution, mixed_type,
                              compact_support) - 15 variants
                - discrete_points: Pure sample arrays (complex_kernels, near_resonance) - 6 variants
                - series: Truncated series (neumann_series) - 3 variants
                - regularized: Ill-posed equations (ill_posed) - 3 variants
                - family: Non-unique solutions (resonance) - 3 variants

                Total: 18 strategies, 42 variants

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
