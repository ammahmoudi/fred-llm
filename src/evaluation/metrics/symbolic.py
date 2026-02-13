"""Symbolic comparison metric for solutions."""

from typing import Any

import sympy as sp

from src.llm.math_verify_adapter import math_verify_compare
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def symbolic_compare(
    solution: sp.Expr,
    ground_truth: sp.Expr,
    tolerance: float = 1e-10,
    use_math_verify: bool = True,
) -> dict[str, Any]:
    """
    Compare two symbolic expressions for equivalence.

    Args:
        solution: Generated solution as SymPy expression.
        ground_truth: Expected solution as SymPy expression.
        tolerance: Tolerance for numerical comparison.

    Returns:
        Dictionary with comparison results.
    """
    result = {
        "equivalent": False,
        "difference": None,
        "simplified_match": False,
    }

    try:
        # Math-Verify fast-path: quick boolean check before heavy simplification
        if use_math_verify:
            mv_result = math_verify_compare(solution, ground_truth)
            if mv_result is True:
                result["equivalent"] = True
                result["simplified_match"] = True
                return result

        # Evaluate any unevaluated Integral objects first
        if solution.has(sp.Integral):
            solution = solution.doit()
        if ground_truth.has(sp.Integral):
            ground_truth = ground_truth.doit()

        # Direct symbolic equality
        if sp.simplify(solution - ground_truth) == 0:
            result["equivalent"] = True
            result["simplified_match"] = True
            return result

        # Try different simplification strategies
        diff = sp.simplify(solution - ground_truth)
        result["difference"] = str(diff)

        # Check if difference simplifies to zero
        if diff.equals(sp.Integer(0)):
            result["equivalent"] = True
            result["simplified_match"] = True

        # Expand and compare
        if sp.expand(solution - ground_truth) == 0:
            result["equivalent"] = True

        # Trigsimp for trigonometric expressions
        if sp.trigsimp(solution - ground_truth) == 0:
            result["equivalent"] = True

    except Exception as e:
        logger.warning(f"Symbolic comparison failed: {e}")
        result["error"] = str(e)

    return result
