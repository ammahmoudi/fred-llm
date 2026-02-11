"""
Kernel composition augmentation strategy.

Composes kernels by adding or multiplying with simple functions.
"""

from typing import Any

import sympy as sp

from src.data.augmentations.base import BaseAugmentation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ComposeAugmentation(BaseAugmentation):
    """
    Compose kernels by adding or multiplying with simple functions.

    Creates variations like:
    - K(x,t) -> K(x,t) + x
    - K(x,t) -> K(x,t) + t
    - K(x,t) -> K(x,t) * x

    This tests the model's ability to handle more complex kernel structures
    and understand kernel composition.
    """

    @property
    def strategy_name(self) -> str:
        return "compose"

    @property
    def description(self) -> str:
        return "Compose kernels by adding/multiplying with simple functions"

    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
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
                new_item["augmentation_variant"] = comp_name
                # Required standard fields
                new_item["has_solution"] = True
                new_item["solution_type"] = "exact_symbolic"
                new_item["edge_case"] = None
                new_item["reason"] = "Kernel composition transformation"
                new_item["recommended_methods"] = []
                new_item["numerical_challenge"] = None
                
                # Generate evaluation points for consistent evaluation
                if new_item.get("has_solution") and new_item.get("u"):
                    try:
                        a_val = float(sp.sympify(new_item.get("a", "0")))
                        b_val = float(sp.sympify(new_item.get("b", "1")))
                        new_item["evaluation_points"] = self._generate_evaluation_points(
                            new_item["u"], a_val, b_val
                        )
                    except Exception as e:
                        logger.debug(f"Failed to generate evaluation points: {e}")
                
                results.append(new_item)
        except Exception as e:
            logger.debug(f"Kernel composition failed: {e}")

        return results
