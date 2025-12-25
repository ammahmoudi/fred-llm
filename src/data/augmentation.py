"""
Data augmentation for Fredholm integral equations.

Generates variations of equations for training data.
"""

from typing import Any

import sympy as sp

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

    augmented = list(data)  # Keep originals

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
        return _substitute_variables(item)
    elif strategy == "scale":
        return _scale_coefficients(item)
    elif strategy == "shift":
        return _shift_domain(item)
    elif strategy == "compose":
        return _compose_kernels(item)
    else:
        return []


def _substitute_variables(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Substitute variables with expressions like x -> x^2, x -> 2*x."""
    results = []
    substitutions = [
        ("x", "2*x", "double_x"),
        ("x", "x**2", "square_x"),
        ("x", "x + 1", "shift_x"),
    ]

    try:
        x = sp.Symbol("x")
        u_expr = sp.sympify(item.get("u", "0"))
        f_expr = sp.sympify(item.get("f", "0"))
        kernel_expr = sp.sympify(item.get("kernel", "0"))

        for old_var, new_var_str, aug_name in substitutions:
            new_var = sp.sympify(new_var_str)
            new_item = item.copy()
            new_item["u"] = str(sp.simplify(u_expr.subs(x, new_var)))
            new_item["f"] = str(sp.simplify(f_expr.subs(x, new_var)))
            new_item["kernel"] = str(sp.simplify(kernel_expr.subs(x, new_var)))
            new_item["augmented"] = True
            new_item["augmentation_type"] = "substitute"
            new_item["augmentation_name"] = aug_name
            results.append(new_item)
    except Exception as e:
        logger.debug(f"Variable substitution failed: {e}")

    return results


def _scale_coefficients(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Scale numerical coefficients by multiplying lambda with different factors."""
    results = []

    try:
        lambda_val = sp.sympify(item.get("lambda_val", "1"))
        for factor in [0.5, 2.0, 0.1, 10.0]:
            new_item = item.copy()
            new_item["lambda_val"] = str(sp.simplify(lambda_val * factor))
            new_item["augmented"] = True
            new_item["augmentation_type"] = "scale"
            new_item["augmentation_factor"] = factor
            results.append(new_item)
    except Exception as e:
        logger.debug(f"Coefficient scaling failed: {e}")

    return results


def _shift_domain(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Shift integration domain to create variations like [a,b] -> [a-1, b-1]."""
    results = []

    try:
        a = sp.sympify(item.get("a", "0"))
        b = sp.sympify(item.get("b", "1"))
        shifts = [
            (a - 1, b - 1, "shift_left"),
            (a + 1, b + 1, "shift_right"),
            (a, b + 1, "extend_right"),
        ]

        for new_a, new_b, shift_name in shifts:
            new_item = item.copy()
            new_item["a"] = str(sp.simplify(new_a))
            new_item["b"] = str(sp.simplify(new_b))
            new_item["augmented"] = True
            new_item["augmentation_type"] = "shift_domain"
            new_item["augmentation_name"] = shift_name
            results.append(new_item)
    except Exception as e:
        logger.debug(f"Domain shifting failed: {e}")

    return results


def _compose_kernels(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Compose kernels by adding or multiplying with simple functions."""
    results = []

    try:
        x = sp.Symbol("x")
        t = sp.Symbol("t")
        kernel_expr = sp.sympify(item.get("kernel", "0"))
        compositions = [
            (kernel_expr + x, "add_x"),
            (kernel_expr + t, "add_t"),
            (kernel_expr * x, "mul_x"),
        ]

        for new_kernel, comp_name in compositions:
            new_item = item.copy()
            new_item["kernel"] = str(sp.simplify(new_kernel))
            new_item["augmented"] = True
            new_item["augmentation_type"] = "compose"
            new_item["augmentation_name"] = comp_name
            results.append(new_item)
    except Exception as e:
        logger.debug(f"Kernel composition failed: {e}")

    return results


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
            strategies: Augmentation strategies to use.
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
