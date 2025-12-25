"""
Variable substitution augmentation strategy.

Substitutes variables with expressions like x -> x², x -> 2x, x -> x+1.
"""

from typing import Any

import sympy as sp

from src.data.augmentations.base import BaseAugmentation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SubstituteAugmentation(BaseAugmentation):
    """
    Substitute variables with expressions.
    
    Applies transformations like:
    - x -> 2x (double the variable)
    - x -> x² (square the variable)
    - x -> x + 1 (shift the variable)
    
    This generates variations that test the model's understanding of
    variable transformations and function composition.
    """

    @property
    def strategy_name(self) -> str:
        return "substitute"

    @property
    def description(self) -> str:
        return "Substitute variables with expressions (x -> x², 2x, x+1)"

    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Substitute variables with expressions."""
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
