"""
No-solution augmentation strategy.

Generates equations where λ is an eigenvalue, violating the Fredholm Alternative.
These equations have no solution because they fall into the singular case.
"""

from typing import Any

import sympy as sp

from src.data.augmentations.base import BaseAugmentation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class NoSolutionAugmentation(BaseAugmentation):
    """
    Generate no-solution cases where λ is an eigenvalue.

    Category 1: The "No Solution" (Singular) Case

    The Logic:
        Find a kernel where λ is an eigenvalue. For example:
        - If K(x,t) = 1 and limits are [0,1], then λ = 1 is an eigenvalue
        - The homogeneous equation u(x) - λ∫K(x,t)u(t)dt = 0 has non-trivial solutions

    The Problem:
        If you set f(x) = x (not orthogonal to eigenfunctions), the equation
        u(x) - ∫₀¹ u(t)dt = x has no solution because:
        - The integral of any solution must equal constant
        - But f(x) = x cannot be represented this way

    Label:
        {"has_solution": false, "reason": "Violates Fredholm Alternative - λ is eigenvalue"}
    """

    @property
    def strategy_name(self) -> str:
        return "eigenvalue_cases"

    @property
    def description(self) -> str:
        return "Generate singular cases where λ is an eigenvalue (no solution exists)"

    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate no-solution cases."""
        results = []

        try:
            # Case 1: Constant kernel K(x,t) = 1, λ = 1/(b-a), f(x) = x
            # The eigenvalue is λ = 1/(b-a) for homogeneous case
            # Setting f(x) = x (non-constant) creates no-solution scenario
            a = float(sp.sympify(item.get("a", "0")))
            b = float(sp.sympify(item.get("b", "1")))
            eigenvalue = 1.0 / (b - a) if b != a else 1.0

            case1 = {
                "u": "",  # No solution exists
                "f": "x",
                "kernel": "1",
                "lambda_val": str(eigenvalue),
                "lambda_val": str(eigenvalue),
                "a": str(a),
                "b": str(b),
                "has_solution": False,
                "solution_type": "none",
                "reason": "Violates Fredholm Alternative - λ is eigenvalue of constant kernel",
                "edge_case": "eigenvalue_cases",
                "augmented": True,
                "augmentation_type": self.strategy_name,
                "augmentation_variant": "constant_kernel_eigenvalue",
                "recommended_methods": [
                    "Check Fredholm Alternative conditions",
                    "Verify eigenvalue",
                ],
                "numerical_challenge": None,
            }
            results.append(case1)

            # Case 2: Separable kernel K(x,t) = x*t, λ = 3/(b³-a³), f(x) = x²
            # For kernel x*t on [a,b], eigenvalue approximately 3/(b³-a³)
            if a >= 0:  # Avoid negative issues
                eigenvalue2 = 3.0 / (b**3 - a**3) if b**3 != a**3 else 1.0
                case2 = {
                    "u": "",  # No solution exists
                    "f": "x**2",
                    "kernel": "x*t",
                    "lambda_val": str(eigenvalue2),
                    "lambda_val": str(eigenvalue2),
                    "a": str(a),
                    "b": str(b),
                    "has_solution": False,
                    "solution_type": "none",
                    "reason": "Violates Fredholm Alternative - λ is eigenvalue of separable kernel",
                    "edge_case": "eigenvalue_cases",
                    "augmented": True,
                    "augmentation_type": self.strategy_name,
                    "augmentation_variant": "separable_kernel_eigenvalue",
                    "recommended_methods": [
                        "Check Fredholm Alternative conditions",
                        "Verify eigenvalue",
                    ],
                    "numerical_challenge": None,
                }
                results.append(case2)

            # Case 3: Symmetric kernel K(x,t) = cos(x-t), λ = 1, f(x) = sin(x)
            # For many symmetric kernels, λ = 1 is often an eigenvalue
            case3 = {
                "u": "",  # No solution exists
                "f": "sin(x)",
                "kernel": "cos(x - t)",
                "lambda_val": "1",
                "lambda_val": "1",
                "a": str(a),
                "b": str(b),
                "has_solution": False,
                "solution_type": "none",
                "reason": "Violates Fredholm Alternative - symmetric kernel with eigenvalue λ=1",
                "edge_case": "eigenvalue_cases",
                "augmented": True,
                "augmentation_type": self.strategy_name,
                "augmentation_variant": "symmetric_kernel_eigenvalue",
                "recommended_methods": [
                    "Check Fredholm Alternative conditions",
                    "Verify eigenvalue",
                ],
                "numerical_challenge": None,
            }
            results.append(case3)

        except Exception as e:
            logger.debug(f"No-solution augmentation failed: {e}")

        return results
