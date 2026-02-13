"""Series solution evaluator."""

from typing import Any

import numpy as np
import sympy as sp

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def count_series_terms(expr: sp.Expr) -> int:
    """Count top-level terms in a truncated series expression."""
    try:
        if isinstance(expr, sp.Add):
            return len(expr.as_ordered_terms())
        return 1
    except Exception as e:
        logger.debug(f"Series term count failed: {e}")
        return 0


def _split_series_terms(expr: sp.Expr) -> list[sp.Expr]:
    """Split a series expression into top-level terms."""
    try:
        if isinstance(expr, sp.Add):
            return list(expr.as_ordered_terms())
        return [expr]
    except Exception as e:
        logger.debug(f"Series term split failed: {e}")
        return []


def evaluate_series_terms(
    solution: sp.Expr,
    ground_truth: sp.Expr,
    domain: tuple[float, float] = (0, 1),
    n_points: int = 100,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Evaluate series solutions term-by-term using numeric error metrics."""
    result: dict[str, Any] = {
        "match": False,
        "terms_compared": 0,
        "pred_terms": 0,
        "gt_terms": 0,
        "mean_rmse": float("inf"),
        "max_rmse": float("inf"),
        "term_match_rate": 0.0,
        "per_term_rmse": [],
    }

    try:
        x = sp.Symbol("x")
        pred_terms = _split_series_terms(solution)
        gt_terms = _split_series_terms(ground_truth)

        result["pred_terms"] = len(pred_terms)
        result["gt_terms"] = len(gt_terms)

        n_terms = min(len(pred_terms), len(gt_terms))
        if n_terms == 0:
            result["error"] = "No terms available for comparison"
            return result

        a, b = domain
        test_points = np.linspace(a, b, n_points)

        rmses: list[float] = []
        matches = 0

        for idx in range(n_terms):
            pred_term = pred_terms[idx]
            gt_term = gt_terms[idx]

            if pred_term.has(sp.Integral):
                pred_term = pred_term.doit()
            if gt_term.has(sp.Integral):
                gt_term = gt_term.doit()

            extra_pred = pred_term.free_symbols - {x}
            extra_gt = gt_term.free_symbols - {x}
            if extra_pred or extra_gt:
                extra = extra_pred | extra_gt
                result["error"] = f"Series terms contain non-numeric symbols: {extra}"
                return result

            f_pred = sp.lambdify(x, pred_term, modules=["numpy"])
            f_gt = sp.lambdify(x, gt_term, modules=["numpy"])

            y_pred = np.array([f_pred(xi) for xi in test_points])
            y_gt = np.array([f_gt(xi) for xi in test_points])

            errors = y_pred - y_gt
            rmse = float(np.sqrt(np.mean(errors**2)))
            rmses.append(rmse)
            if rmse < tolerance:
                matches += 1

        result["terms_compared"] = n_terms
        result["per_term_rmse"] = rmses
        result["mean_rmse"] = float(np.mean(rmses))
        result["max_rmse"] = float(np.max(rmses))
        result["term_match_rate"] = matches / n_terms
        result["match"] = matches == n_terms

    except Exception as e:
        logger.warning(f"Series term comparison failed: {e}")
        result["error"] = str(e)

    return result
