"""
Domain shifting augmentation strategy.

Shifts the integration domain [a, b] to create variations.
"""

from typing import Any

import sympy as sp

from src.data.augmentations.base import BaseAugmentation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ShiftAugmentation(BaseAugmentation):
    """
    Shift integration domain to create variations.

    Creates variations like:
    - [a, b] -> [a-1, b-1] (shift left)
    - [a, b] -> [a+1, b+1] (shift right)
    - [a, b] -> [a, b+1] (extend right)

    This tests the model's understanding of how integration bounds
    affect the solution.
    """

    @property
    def strategy_name(self) -> str:
        return "shift"

    @property
    def description(self) -> str:
        return "Shift integration domain [a,b] to create variations"

    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Shift integration domain to create variations."""
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
                new_item["augmentation_type"] = "shift"
                new_item["augmentation_variant"] = shift_name
                # Required standard fields
                new_item["has_solution"] = True
                new_item["solution_type"] = "exact_symbolic"
                new_item["edge_case"] = None
                new_item["reason"] = "Integration domain shift transformation"
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
            logger.debug(f"Domain shifting failed: {e}")

        return results
