"""
Resonance/criticality augmentation strategy.

Generates equations where λ is near bifurcation/resonance points,
leading to non-unique solutions or solution families.
"""

from typing import Any

import sympy as sp

from src.data.augmentations.base import BaseAugmentation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ResonanceAugmentation(BaseAugmentation):
    """
    Generate resonance/critical parameter cases.

    Mathematical Background:
        For Fredholm equation: u(x) - λ∫K(x,t)u(t)dt = f(x)

        The operator (I - λK) has eigenvalues μₙ. When λ = 1/μₙ (eigenvalue),
        the operator becomes singular and:
        - Homogeneous equation has non-trivial solutions
        - Non-homogeneous equation has solutions only if f ⊥ eigenfunctions
        - Solutions are non-unique (differ by eigenfunctions)

    Fredholm Alternative:
        At resonance λ = λ_critical:
        1. If f ⊥ all eigenfunctions: Infinitely many solutions exist
        2. If f not orthogonal: No solution exists

    The Challenge for LLM:
        Model must:
        - Recognize λ is at critical value
        - Identify that solution is non-unique
        - Provide general solution: u = u_particular + c*φ(x)
        - Explain family of solutions parameterized by constant c

    Physical Context:
        Resonance appears in:
        - Vibrating systems at natural frequencies
        - Quantum mechanics at energy eigenvalues
        - Wave propagation at cutoff frequencies

    Example:
        u(x) - λ∫[0,1] sin(πx)sin(πt) u(t) dt = 0

        Kernel K(x,t) = sin(πx)sin(πt) has eigenvalue μ = 1/2
        At λ = 2 (resonance), any u(x) = c*sin(πx) is a solution.

    Label:
        {
            "has_solution": true,
            "solution_type": "family",
            "edge_case": "resonance",
            "is_critical": true,
            "eigenvalue_approx": critical_value,
            "solution_multiplicity": "infinite",
            "general_solution": "u_particular + c*phi(x)"
        }
    """

    def __init__(self, perturbation: float = 0.001) -> None:
        """
        Initialize resonance augmentation.

        Args:
            perturbation: How close to exact resonance (0 = exact, >0 = near)
        """
        self.perturbation = perturbation

    @property
    def strategy_name(self) -> str:
        return "resonance"

    @property
    def description(self) -> str:
        return "Generate equations at resonance/bifurcation points"

    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate resonance cases."""
        results = []

        try:
            # Extract base parameters
            a = float(sp.sympify(item.get("a", "0")))
            b = float(sp.sympify(item.get("b", "1")))

            # Case 1: Separable kernel sin(πx)sin(πt) on [0,1]
            # Eigenvalue: μ = 1/2, so λ_critical = 2
            lambda_critical_1 = 2.0 + self.perturbation
            eigenfunction_1 = "sin(pi*x)"

            case1 = {
                "u": f"C * {eigenfunction_1}",  # Solution family
                "f": "0",  # Homogeneous case for clarity
                "kernel": "sin(pi*x) * sin(pi*t)",
                "lambda_val": str(lambda_critical_1),
                "lambda_val": str(lambda_critical_1),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "family",
                "edge_case": "resonance",
                "is_critical": True,
                "eigenvalue_approximate": 0.5,
                "eigenfunction": eigenfunction_1,
                "solution_multiplicity": "infinite",
                "general_solution": f"C * {eigenfunction_1} for any constant C",
                "recommended_methods": [
                    "recognize_resonance",
                    "eigenvalue_analysis",
                    "fredholm_alternative",
                ],
                "numerical_challenge": "Non-unique solution - any multiple of eigenfunction works",
                "augmented": True,
                "augmentation_type": "resonance",
                "augmentation_variant": "separable_eigenvalue_exact",
            }
            results.append(case1)

            # Case 2: Constant kernel K(x,t) = 1 on [0,1]
            # Eigenvalue: μ = 1, so λ_critical = 1
            lambda_critical_2 = 1.0 + self.perturbation
            eigenfunction_2 = "1"  # Constant function

            case2 = {
                "u": "C",  # Any constant is solution
                "f": "0",
                "kernel": "1",
                "lambda_val": str(lambda_critical_2),
                "lambda_val": str(lambda_critical_2),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "family",
                "edge_case": "resonance",
                "is_critical": True,
                "eigenvalue_approximate": 1.0,
                "eigenfunction": "1",
                "solution_multiplicity": "infinite",
                "general_solution": "u = C (any constant)",
                "recommended_methods": [
                    "recognize_resonance",
                    "eigenvalue_analysis",
                    "fredholm_alternative",
                ],
                "numerical_challenge": "Infinite family of constant solutions",
                "augmented": True,
                "augmentation_type": "resonance",
                "augmentation_variant": "constant_kernel_resonance",
            }
            results.append(case2)

        except Exception as e:
            logger.warning(f"Failed to generate resonance case: {e}")

        return results
