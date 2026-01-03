"""
Near-resonance augmentation strategy.

Generates equations where λ is close to (but not exactly at) eigenvalue,
creating ill-conditioned problems with very large solution amplitudes.
"""

from typing import Any

import sympy as sp

from src.data.augmentations.base import BaseAugmentation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class NearResonanceAugmentation(BaseAugmentation):
    """
    Generate near-resonance cases with large amplitude solutions.

    Mathematical Background:
        When λ is close to an eigenvalue λ_critical but not exactly equal:
        - Solution still exists and is unique
        - BUT solution has very large amplitude
        - System is ill-conditioned: (I - λK) is nearly singular
        - Small changes in λ or f cause huge changes in u

    Physical Context:
        Like forcing a spring near its natural frequency:
        - At exact resonance: infinite amplitude (no unique solution)
        - Near resonance: finite but very large amplitude
        - Requires careful numerical treatment

    Example:
        u(x) - 2.1∫[0,1] sin(πx)sin(πt) u(t) dt = sin(πx)
        
        Critical value: λ_c = 2.0
        Actual value: λ = 2.1 (distance = 0.1)
        Condition number: ~1/0.1 = 10 (ill-conditioned)

    The Challenge for LLM:
        Model must:
        - Recognize λ is near eigenvalue
        - Understand solution exists but is ill-conditioned
        - Recommend iterative methods with preconditioning
        - Warn about sensitivity to perturbations

    Label:
        {
            "has_solution": true,
            "solution_type": "numerical",
            "edge_case": "near_resonance",
            "near_critical_value": eigenvalue,
            "distance_to_resonance": |λ - λ_critical|,
            "condition_number_estimate": 1/distance,
            "numerical_challenge": "Large amplitude, ill-conditioned"
        }
    """

    def __init__(self, distance: float = 0.1) -> None:
        """
        Initialize near-resonance augmentation.

        Args:
            distance: How far from exact resonance (larger = better conditioned)
        """
        self.distance = distance

    @property
    def strategy_name(self) -> str:
        return "near_resonance"

    @property
    def description(self) -> str:
        return "Generate ill-conditioned equations near eigenvalue resonance"

    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate near-resonance cases."""
        results = []

        try:
            # Extract base parameters
            a = float(sp.sympify(item.get("a", "0")))
            b = float(sp.sympify(item.get("b", "1")))

            # Case 1: Near separable kernel eigenvalue
            # Critical: λ_c = 2.0, Near: λ = 2.1
            lambda_critical_1 = 2.0
            lambda_near_1 = lambda_critical_1 + self.distance
            eigenfunction_1 = "sin(pi*x)"

            case1 = {
                "u": "",  # Large amplitude - no simple closed form
                "f": eigenfunction_1,  # f = sin(πx) not orthogonal to itself
                "kernel": "sin(pi*x) * sin(pi*t)",
                "lambda_val": str(lambda_near_1),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "numerical",
                "edge_case": "near_resonance",
                "is_critical": False,
                "near_critical_value": lambda_critical_1,
                "distance_to_resonance": self.distance,
                "recommended_methods": [
                    "regularization",
                    "iterative_refinement",
                    "preconditioned_gmres",
                    "continuation_methods",
                ],
                "numerical_challenge": "Solution exists but has very large amplitude near resonance",
                "condition_number_estimate": 1.0 / self.distance,
                "reason": f"λ={lambda_near_1} is close to eigenvalue λ_c={lambda_critical_1}",
                "warning": "Highly sensitive to perturbations in λ and f",
                "augmented": True,
                "augmentation_type": self.strategy_name,
                "augmentation_variant": "separable_near_eigenvalue",
            }
            results.append(case1)

            # Case 2: Near constant kernel eigenvalue
            # Critical: λ_c = 1.0, Near: λ = 1.05
            lambda_critical_2 = 1.0
            lambda_near_2 = lambda_critical_2 + self.distance / 2
            
            case2 = {
                "u": "",  # Large amplitude
                "f": "1",  # Constant forcing
                "kernel": "1",
                "lambda_val": str(lambda_near_2),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "numerical",
                "edge_case": "near_resonance",
                "is_critical": False,
                "near_critical_value": lambda_critical_2,
                "distance_to_resonance": self.distance / 2,
                "recommended_methods": [
                    "regularization",
                    "iterative_methods",
                    "tikhonov_regularization",
                ],
                "numerical_challenge": "Nearly singular operator - solution exists but amplified",
                "condition_number_estimate": 2.0 / self.distance,
                "reason": f"λ={lambda_near_2} is close to eigenvalue λ_c={lambda_critical_2}",
                "warning": "Small changes in λ cause order-of-magnitude changes in u",
                "augmented": True,
                "augmentation_type": self.strategy_name,
                "augmentation_variant": "constant_near_eigenvalue",
            }
            results.append(case2)

            # Case 3: Very close to resonance (more extreme)
            lambda_critical_3 = 2.0
            lambda_near_3 = lambda_critical_3 + self.distance / 10  # 10x closer
            
            case3 = {
                "u": "",
                "f": "sin(pi*x) + 0.1*sin(2*pi*x)",  # Mixed frequencies
                "kernel": "sin(pi*x) * sin(pi*t)",
                "lambda_val": str(lambda_near_3),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "numerical",
                "edge_case": "near_resonance",
                "is_critical": False,
                "near_critical_value": lambda_critical_3,
                "distance_to_resonance": self.distance / 10,
                "recommended_methods": [
                    "strong_regularization",
                    "continuation_from_away",
                    "preconditioned_iterative",
                ],
                "numerical_challenge": "Extremely close to resonance - condition number ~100",
                "condition_number_estimate": 10.0 / self.distance,
                "reason": f"λ={lambda_near_3:.3f} very close to eigenvalue λ_c={lambda_critical_3}",
                "warning": "Extreme ill-conditioning - even roundoff errors matter",
                "augmented": True,
                "augmentation_type": self.strategy_name,
                "augmentation_variant": "very_near_eigenvalue",
            }
            results.append(case3)

        except Exception as e:
            logger.debug(f"Near-resonance augmentation failed: {e}")

        return results
