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
                new_item["augmentation_name"] = comp_name
                results.append(new_item)
        except Exception as e:
            logger.debug(f"Kernel composition failed: {e}")

        return results
