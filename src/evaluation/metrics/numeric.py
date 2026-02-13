"""Numeric comparison metric for solutions."""

from typing import Any, Optional

import numpy as np
import sympy as sp

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def numeric_compare(
    solution: sp.Expr,
    ground_truth: sp.Expr | dict[str, Any],
    domain: tuple[float, float] = (0, 1),
    n_points: int = 100,
    tolerance: float = 1e-6,
    include_points: bool = False,
) -> dict[str, Any]:
    """
    Compare two expressions numerically over a domain.

    Args:
        solution: Generated solution as SymPy expression.
        ground_truth: Expected solution as SymPy expression, or dictionary with
            'evaluation_points' field containing pre-computed evaluation points.
        domain: Integration domain (a, b).
        n_points: Number of test points (used if no pre-computed points available).
        tolerance: Tolerance for numeric comparison.

    Returns:
        Dictionary with numeric comparison results.
    """
    result = {
        "match": False,
        "max_error": float("inf"),
        "mean_error": float("inf"),
        "mae": float("inf"),
        "rmse": float("inf"),
        "rel_l2": float("inf"),
    }

    points_source: str | None = None

    try:
        x = sp.Symbol("x")

        # Check if ground_truth is a dictionary with evaluation_points
        if isinstance(ground_truth, dict):
            if "evaluation_points" in ground_truth:
                # Use pre-computed evaluation points for consistent metrics
                eval_points = ground_truth["evaluation_points"]
                test_points = np.array(eval_points["x_values"])
                y_truth = np.array(eval_points["u_values"])

                # Evaluate solution at the same points
                if solution.has(sp.Integral):
                    solution = solution.doit()

                # Check for extra symbols
                extra_sol = solution.free_symbols - {x}
                if extra_sol:
                    result["error"] = (
                        f"Solution contains non-numeric symbols: {extra_sol}"
                    )
                    logger.debug(
                        f"Numeric comparison skipped: non-numeric symbols {extra_sol}"
                    )
                    return result

                f_solution = sp.lambdify(x, solution, modules=["numpy"])
                y_solution = np.array([f_solution(xi) for xi in test_points])
                points_source = "evaluation_points"

            elif "u" in ground_truth and ground_truth["u"]:
                # Fallback: Extract u field and use as ground truth expression
                ground_truth_expr = sp.sympify(ground_truth["u"])
                # Continue to standard evaluation below
            else:
                # No evaluation data available
                result["error"] = "No evaluation data available in ground_truth dict"
                logger.debug("Numeric comparison skipped: no evaluation data")
                return result
        else:
            # ground_truth is a SymPy expression - standard behavior
            ground_truth_expr = ground_truth

        # Standard evaluation path (when not using pre-computed points)
        if "y_solution" not in locals():
            # Evaluate any unevaluated Integral objects before lambdify
            if solution.has(sp.Integral):
                solution = solution.doit()
            if ground_truth_expr.has(sp.Integral):
                ground_truth_expr = ground_truth_expr.doit()

            # Check that expressions only depend on x (no other free symbols)
            extra_sol = solution.free_symbols - {x}
            extra_gt = ground_truth_expr.free_symbols - {x}
            if extra_sol or extra_gt:
                extra = extra_sol | extra_gt
                result["error"] = f"Expressions contain non-numeric symbols: {extra}"
                logger.debug(f"Numeric comparison skipped: non-numeric symbols {extra}")
                return result

            # Convert to numeric functions
            f_solution = sp.lambdify(x, solution, modules=["numpy"])
            f_truth = sp.lambdify(x, ground_truth_expr, modules=["numpy"])

            # Generate test points
            a, b = domain
            test_points = np.linspace(a, b, n_points)

            # Evaluate
            y_solution = np.array([f_solution(xi) for xi in test_points])
            y_truth = np.array([f_truth(xi) for xi in test_points])
            points_source = "generated"

        # Compute errors
        errors = np.abs(y_solution - y_truth)
        result["max_error"] = float(np.max(errors))
        result["mean_error"] = float(np.mean(errors))
        result["mae"] = result["mean_error"]
        result["rmse"] = float(np.sqrt(np.mean(errors**2)))

        # Relative L2 error: ||pred - true||_2 / ||true||_2
        # Scale-invariant; standard in PDEBench and CodePDE.
        gt_norm = float(np.sqrt(np.sum(y_truth**2)))
        if gt_norm > 0:
            result["rel_l2"] = float(np.sqrt(np.sum(errors**2))) / gt_norm
        else:
            result["rel_l2"] = 0.0 if result["rmse"] == 0.0 else float("inf")

        # Check if within tolerance
        result["match"] = result["max_error"] < tolerance

        if points_source is None:
            points_source = "generated"
        result["evaluation_points_used"] = int(len(test_points))
        result["points_source"] = points_source

        if include_points:
            result["x_values"] = test_points.tolist()
            result["y_pred"] = y_solution.tolist()
            result["y_true"] = y_truth.tolist()

    except Exception as e:
        logger.warning(f"Numeric comparison failed: {e}")
        result["error"] = str(e)

    return result
