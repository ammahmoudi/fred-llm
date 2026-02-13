"""Approximation coefficient solution evaluator."""

from typing import Any

import numpy as np
import sympy as sp

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _extract_term_coeffs(expr: sp.Expr) -> dict[sp.Expr, float]:
    """Extract numeric coefficients keyed by their base term expression."""
    terms = expr.as_ordered_terms() if isinstance(expr, sp.Add) else [expr]
    coeffs: dict[sp.Expr, float] = {}

    for term in terms:
        coeff, base = term.as_coeff_Mul()
        base = sp.simplify(base)
        try:
            coeff_val = float(sp.N(coeff))
        except Exception as e:
            logger.debug(f"Non-numeric coefficient skipped: {coeff} ({e})")
            continue

        if base in coeffs:
            coeffs[base] += coeff_val
        else:
            coeffs[base] = coeff_val

    return coeffs


def evaluate_approx_coeffs(
    solution: sp.Expr,
    ground_truth: sp.Expr,
    tolerance: float = 1e-6,
    relative_tolerance: float = 0.1,
) -> dict[str, Any]:
    """Evaluate approx_coef solutions by comparing per-term coefficients."""
    result: dict[str, Any] = {
        "match": False,
        "terms_compared": 0,
        "match_rate": 0.0,
        "mean_abs_error": float("inf"),
        "max_abs_error": float("inf"),
        "mean_rel_error": float("inf"),
        "max_rel_error": float("inf"),
        "per_term_errors": [],
    }

    try:
        pred_coeffs = _extract_term_coeffs(solution)
        gt_coeffs = _extract_term_coeffs(ground_truth)

        if not gt_coeffs:
            result["error"] = "No ground truth coefficients available"
            return result

        abs_errors: list[float] = []
        rel_errors: list[float] = []
        matches = 0
        per_term: list[dict[str, Any]] = []

        for base, gt_val in gt_coeffs.items():
            pred_val = pred_coeffs.get(base, 0.0)
            abs_error = abs(pred_val - gt_val)
            rel_error = (
                abs_error / abs(gt_val) if abs(gt_val) > tolerance else float("inf")
            )
            term_match = abs_error <= tolerance or rel_error <= relative_tolerance

            abs_errors.append(abs_error)
            if rel_error != float("inf"):
                rel_errors.append(rel_error)

            per_term.append(
                {
                    "term": str(base),
                    "pred_coeff": pred_val,
                    "gt_coeff": gt_val,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                    "match": term_match,
                }
            )

            if term_match:
                matches += 1

        terms_compared = len(gt_coeffs)
        result["terms_compared"] = terms_compared
        result["match_rate"] = matches / terms_compared if terms_compared else 0.0
        result["mean_abs_error"] = (
            float(np.mean(abs_errors)) if abs_errors else float("inf")
        )
        result["max_abs_error"] = (
            float(np.max(abs_errors)) if abs_errors else float("inf")
        )
        result["mean_rel_error"] = (
            float(np.mean(rel_errors)) if rel_errors else float("inf")
        )
        result["max_rel_error"] = (
            float(np.max(rel_errors)) if rel_errors else float("inf")
        )
        result["per_term_errors"] = per_term
        result["match"] = matches == terms_compared and terms_compared > 0

    except Exception as e:
        logger.warning(f"Approx coef comparison failed: {e}")
        result["error"] = str(e)

    return result
