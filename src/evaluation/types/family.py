"""Family type solution evaluator."""

from typing import Any

import numpy as np
import sympy as sp

from src.evaluation.metrics.numeric import numeric_compare
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def family_compare(
    solution: sp.Expr,
    ground_truth: sp.Expr,
) -> bool:
    """
    Check if solution matches a family ground truth up to a free constant.

    Finds free constant symbols (C, c_1, c_2) in ground_truth, substitutes
    them with 1 to get the structural part, then checks if solution / structural_part
    simplifies to an x-free constant.

    Args:
        solution: Predicted solution expression.
        ground_truth: Ground truth family expression with free constants.

    Returns:
        True if solution matches the family structure.
    """
    x = sp.Symbol("x")
    # Identify free constant symbols in ground truth:
    # any symbol that's not the independent variable (x) or integration variable (t)
    standard_vars = {"x", "t"}
    free_constants = [
        s for s in ground_truth.free_symbols if s.name not in standard_vars
    ]

    if not free_constants:
        return False

    try:
        # Substitute all free constants with 1 to get structural part
        structural = ground_truth
        for c in free_constants:
            structural = structural.subs(c, 1)

        # Check if solution / structural simplifies to an x-free constant
        if structural == 0:
            return False

        ratio = sp.simplify(solution / structural)
        # Ratio should be free of x
        if x not in ratio.free_symbols:
            return True

        # Also try difference approach: solution - C * structural = 0
        diff = sp.simplify(solution - structural)
        if x not in diff.free_symbols:
            return True

    except Exception as e:
        logger.debug(f"Family comparison failed: {e}")

    return False


def _substitute_family_constants(expr: sp.Expr, value: float = 1.0) -> sp.Expr:
    """Substitute free constants in a family expression with a fixed value."""
    x = sp.Symbol("x")
    t = sp.Symbol("t")
    constants = expr.free_symbols - {x, t}
    if not constants:
        return expr
    return expr.subs({sym: value for sym in constants})


def _family_param_metadata(
    solution: sp.Expr,
    ground_truth: sp.Expr,
) -> dict[str, Any]:
    """Collect family parameter count and naming metadata."""
    x = sp.Symbol("x")
    t = sp.Symbol("t")
    standard_vars = {x, t}

    gt_params = sorted(
        [s for s in ground_truth.free_symbols if s not in standard_vars],
        key=lambda s: s.name,
    )
    pred_params = sorted(
        [s for s in solution.free_symbols if s not in standard_vars],
        key=lambda s: s.name,
    )

    def _name_valid(sym: sp.Symbol) -> bool:
        return sym.name == "C" or sym.name.startswith("c_")

    naming_valid = (
        all(_name_valid(sym) for sym in pred_params) if pred_params else False
    )

    return {
        "param_count_pred": len(pred_params),
        "param_count_gt": len(gt_params),
        "param_count_match": len(pred_params) == len(gt_params),
        "param_names_pred": [s.name for s in pred_params],
        "param_names_gt": [s.name for s in gt_params],
        "param_naming_valid": naming_valid,
    }


def _family_numeric_compare_samples(
    solution: sp.Expr,
    ground_truth: sp.Expr,
    domain: tuple[float, float],
    n_points: int,
    tolerance: float,
    constant_samples: list[float],
    evaluation_points: dict[str, Any] | None = None,
    include_points: bool = False,
) -> dict[str, Any]:
    """Numeric comparison for family solutions across multiple constant samples."""
    x = sp.Symbol("x")
    t = sp.Symbol("t")
    constants = (solution.free_symbols | ground_truth.free_symbols) - {x, t}

    if not constants:
        if evaluation_points:
            return numeric_compare(
                solution,
                {"evaluation_points": evaluation_points, "u": str(ground_truth)},
                domain,
                n_points,
                tolerance,
                include_points=include_points,
            )
        return numeric_compare(solution, ground_truth, domain, n_points, tolerance)

    if evaluation_points and evaluation_points.get("x_values"):
        x_values = np.array(evaluation_points["x_values"], dtype=float)
        if x_values.size > 0:
            if "u_values_samples" in evaluation_points:
                gt_samples = evaluation_points["u_values_samples"]
                sample_constants = evaluation_points.get(
                    "constant_samples", constant_samples
                )
            else:
                gt_samples = [evaluation_points.get("u_values", [])]
                sample_constants = constant_samples[:1]

            if len(gt_samples) != len(sample_constants):
                sample_constants = sample_constants[: len(gt_samples)]

            sample_results = []
            for sample, y_truth in zip(sample_constants, gt_samples):
                subs_map = {sym: sample for sym in constants}
                sol_sub = solution.subs(subs_map)
                if sol_sub.has(sp.Integral):
                    sol_sub = sol_sub.doit()

                extra_sol = sol_sub.free_symbols - {x}
                if extra_sol:
                    sample_results.append(
                        {
                            "match": False,
                            "max_error": float("inf"),
                            "mean_error": float("inf"),
                            "rmse": float("inf"),
                            "error": f"Solution contains non-numeric symbols: {extra_sol}",
                        }
                    )
                    continue

                f_solution = sp.lambdify(x, sol_sub, modules=["numpy"])
                y_solution = np.array([f_solution(xi) for xi in x_values], dtype=float)
                y_truth_arr = np.array(y_truth, dtype=float)

                if y_solution.shape != y_truth_arr.shape:
                    min_len = min(len(y_solution), len(y_truth_arr))
                    y_solution = y_solution[:min_len]
                    y_truth_arr = y_truth_arr[:min_len]

                errors = np.abs(y_solution - y_truth_arr)
                sample_result = {
                    "match": float(np.max(errors)) < tolerance,
                    "max_error": float(np.max(errors)),
                    "mean_error": float(np.mean(errors)),
                    "mae": float(np.mean(errors)),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                }
                if include_points:
                    sample_result["x_values"] = x_values.tolist()
                    # Convert complex numbers to strings for JSON serialization
                    sample_result["y_pred"] = [
                        str(val) if isinstance(val, complex) else val 
                        for val in y_solution.tolist()
                    ]
                    sample_result["y_true"] = [
                        str(val) if isinstance(val, complex) else val 
                        for val in y_truth_arr.tolist()
                    ]
                    sample_result["points_source"] = "evaluation_points"
                sample_results.append(sample_result)

            max_errors = [r["max_error"] for r in sample_results]
            mean_errors = [r["mean_error"] for r in sample_results]
            rmses = [r["rmse"] for r in sample_results]

            max_error = max(max_errors)
            mean_error = float(np.mean(mean_errors))
            rmse = float(np.mean(rmses))
            match = all(r["match"] for r in sample_results)

            return {
                "match": match,
                "max_error": max_error,
                "mean_error": mean_error,
                "mae": mean_error,
                "rmse": rmse,
                "max_error_std": float(np.std(max_errors)),
                "mean_error_std": float(np.std(mean_errors)),
                "rmse_std": float(np.std(rmses)),
                "sample_results": sample_results,
                "constant_samples": sample_constants,
            }

    sample_results = []
    for sample in constant_samples:
        subs_map = {sym: sample for sym in constants}
        sol_sub = solution.subs(subs_map)
        gt_sub = ground_truth.subs(subs_map)
        sample_results.append(
            numeric_compare(sol_sub, gt_sub, domain, n_points, tolerance)
        )

    max_errors = [r["max_error"] for r in sample_results]
    mean_errors = [r["mean_error"] for r in sample_results]
    rmses = [r["rmse"] for r in sample_results]

    max_error = max(max_errors)
    mean_error = float(np.mean(mean_errors))
    rmse = float(np.mean(rmses))
    match = all(r["match"] for r in sample_results)

    return {
        "match": match,
        "max_error": max_error,
        "mean_error": mean_error,
        "mae": mean_error,
        "rmse": rmse,
        "max_error_std": float(np.std(max_errors)),
        "mean_error_std": float(np.std(mean_errors)),
        "rmse_std": float(np.std(rmses)),
        "sample_results": sample_results,
        "constant_samples": constant_samples,
    }
