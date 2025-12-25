"""
Coefficient scaling augmentation strategy.

Scales the lambda parameter by different factors.
"""

from typing import Any

import sympy as sp

from src.data.augmentations.base import BaseAugmentation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ScaleAugmentation(BaseAugmentation):
    """
    Scale lambda coefficients by multiplying with different factors.

    Multiplies λ by factors like 0.5, 2.0, 0.1, 10.0 to generate variations
    with different equation scales. This tests the model's sensitivity to
    parameter magnitude.
    """

    @property
    def strategy_name(self) -> str:
        return "scale"

    @property
    def description(self) -> str:
        return "Scale lambda coefficients by multiplying with factors"

    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Scale numerical coefficients by multiplying lambda."""
        results = []

        try:
            lambda_val = sp.sympify(item.get("lambda_val", item.get("lambda", "1")))
            for factor in [0.5, 2.0, 0.1, 10.0]:
                new_item = item.copy()
                new_lambda = str(sp.simplify(lambda_val * factor))
                new_item["lambda_val"] = new_lambda
                new_item["lambda"] = new_lambda
                new_item["augmented"] = True
                new_item["augmentation_type"] = "scale"
                new_item["augmentation_factor"] = factor
                results.append(new_item)
        except Exception as e:
            logger.debug(f"Coefficient scaling failed: {e}")

        return results
