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
                new_item["augmented"] = True
                new_item["augmentation_type"] = "scale"
                new_item["augmentation_variant"] = f"scale_{factor}x"
                # Required standard fields
                new_item["has_solution"] = True
                new_item["solution_type"] = "exact_symbolic"
                new_item["edge_case"] = None
                new_item["reason"] = f"Lambda coefficient scaled by factor {factor}"
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
            logger.debug(f"Coefficient scaling failed: {e}")

        return results
