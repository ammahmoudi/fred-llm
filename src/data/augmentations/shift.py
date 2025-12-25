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
                new_item["augmentation_type"] = "shift_domain"
                new_item["augmentation_name"] = shift_name
                results.append(new_item)
        except Exception as e:
            logger.debug(f"Domain shifting failed: {e}")

        return results
