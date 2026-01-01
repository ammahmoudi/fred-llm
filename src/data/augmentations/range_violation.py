"""
Range violation augmentation strategy.

Generates equations where the right-hand side f(x) does not belong to the
range of the integral operator, making the equation unsolvable.
"""

from typing import Any

import sympy as sp

from src.data.augmentations.base import BaseAugmentation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RangeViolationAugmentation(BaseAugmentation):
    """
    Generate range space violation cases.

    Mathematical Background:
        For Fredholm equation: u(x) - λ∫K(x,t)u(t)dt = f(x)

        The integral operator T defined by (Tu)(x) = ∫K(x,t)u(t)dt has a range
        (set of all possible outputs). If f(x) is not in Range(I - λT), then
        no solution exists.

    Common Examples:
        1. **Even/Odd Parity Mismatch**:
           - If K is even in both variables, operator produces only even functions
           - Setting f(x) as odd function → no solution

        2. **Separable Kernel with Orthogonality**:
           - K(x,t) = φ(x)ψ(t) produces outputs proportional to φ(x)
           - If f(x) ⊥ φ(x), then f not in range

        3. **Rank-Deficient Operator**:
           - If K has finite rank n, range is n-dimensional
           - f(x) outside this subspace → no solution

    The Challenge for LLM:
        Model must:
        - Recognize structural properties of kernel (symmetry, separability)
        - Identify range of operator
        - Determine if f lies in that range
        - Conclude no solution exists due to range violation

    Physical Context:
        - Occurs in inverse problems with insufficient data
        - Projection onto incomplete basis sets
        - Measurement compatibility in linear systems

    Label:
        {
            "has_solution": false,
            "solution_type": "none",
            "edge_case": "range_violation",
            "reason": "f not in range of operator",
            "operator_property": "even|odd|separable|finite_rank"
        }
    """

    @property
    def strategy_name(self) -> str:
        return "range_violation"

    @property
    def description(self) -> str:
        return "RHS not in range of integral operator"

    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate range violation cases."""
        results = []

        try:
            # Extract base parameters
            a = float(sp.sympify(item.get("a", "0")))
            b = float(sp.sympify(item.get("b", "1")))
            lambda_val = float(
                sp.sympify(item.get("lambda", item.get("lambda_val", "1")))
            )

            # Case 1: Even kernel, odd forcing function
            # K(x,t) = cos(x)cos(t) is even → range contains only even functions
            # f(x) = x is odd → not in range
            case1 = {
                "u": "None",  # No solution exists
                "f": "x",  # Odd function
                "kernel": "cos(x) * cos(t)",  # Even kernel
                "lambda": str(lambda_val),
                "lambda_val": str(lambda_val),
                "a": str(a),
                "b": str(b),
                "has_solution": False,
                "solution_type": "none",
                "edge_case": "range_violation",
                "reason": "f(x) is odd but kernel produces only even functions",
                "operator_property": "even_symmetry",
                "mathematical_explanation": "Range(T) ⊂ L²_even, but f ∈ L²_odd",
                "recommended_methods": [
                    "symmetry_analysis",
                    "range_space_decomposition",
                    "parity_check",
                ],
                "augmented": True,
                "augmentation_type": "range_violation",
                "augmentation_variant": "even_odd_mismatch",
            }
            results.append(case1)

            # Case 2: Separable kernel with orthogonal forcing
            # K(x,t) = sin(pi*x) * sin(pi*t)
            # Range is 1D: all functions proportional to sin(pi*x)
            # f(x) = cos(pi*x) is orthogonal to sin(pi*x) → not in range
            case2 = {
                "u": "None",
                "f": "cos(pi*x)",  # Orthogonal to kernel function
                "kernel": "sin(pi*x) * sin(pi*t)",  # Rank-1 separable
                "lambda": str(lambda_val * 0.5),
                "lambda_val": str(lambda_val * 0.5),
                "a": str(a),
                "b": str(b),
                "has_solution": False,
                "solution_type": "none",
                "edge_case": "range_violation",
                "reason": "f orthogonal to range of separable kernel",
                "operator_property": "separable_rank_one",
                "mathematical_explanation": "K(x,t) = φ(x)ψ(t), Range = span{φ}, but ⟨f,φ⟩ ≠ 0",
                "orthogonality_violated": "cos(pi*x) ⊥ sin(pi*x)",
                "recommended_methods": [
                    "separability_detection",
                    "orthogonality_test",
                    "range_projection",
                ],
                "augmented": True,
                "augmentation_type": "range_violation",
                "augmentation_variant": "separable_orthogonal",
            }
            results.append(case2)

            # Case 3: Finite rank kernel with incompatible f
            # K(x,t) = sin(x)*sin(t) + cos(x)*cos(t) (rank 2)
            # Range = span{sin(x), cos(x)}
            # f(x) = x² not in span → no solution
            case3 = {
                "u": "None",
                "f": "x**2",  # Polynomial not in trig span
                "kernel": "sin(x)*sin(t) + cos(x)*cos(t)",  # Rank 2
                "lambda": str(lambda_val * 0.3),
                "lambda_val": str(lambda_val * 0.3),
                "a": str(a),
                "b": str(b),
                "has_solution": False,
                "solution_type": "none",
                "edge_case": "range_violation",
                "reason": "f not in finite-dimensional range of kernel",
                "operator_property": "finite_rank",
                "kernel_rank": 2,
                "range_basis": "span{sin(x), cos(x)}",
                "mathematical_explanation": "Operator has rank 2, but f requires infinite basis",
                "recommended_methods": [
                    "rank_analysis",
                    "basis_expansion",
                    "projection_test",
                ],
                "augmented": True,
                "augmentation_type": "range_violation",
                "augmentation_variant": "finite_rank_incompatible",
            }
            results.append(case3)

        except Exception as e:
            logger.warning(f"Failed to generate range violation case: {e}")

        return results
