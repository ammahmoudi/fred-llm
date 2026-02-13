"""
Evaluation utilities for Fredholm equation solutions.

Provides both symbolic and numeric evaluation metrics.
"""

import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import sympy as sp
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from scipy import integrate

from src.llm.math_verify_adapter import (
    FREDHOLM_LOCAL_DICT,
    TRANSFORMATIONS,
    math_verify_compare,
    parse_latex_to_sympy,
)
from src.llm.postprocess import ParseError
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Operators we track, matching article Appendix C
_TRACKED_OPERATORS: set[type] = {
    sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.sqrt,
    sp.sinh, sp.cosh, sp.tanh, sp.Abs,
    sp.Add, sp.Mul, sp.Pow,
    sp.Integral,
}

_OPERATOR_NAMES: dict[type, str] = {
    sp.sin: "sin", sp.cos: "cos", sp.tan: "tan",
    sp.exp: "exp", sp.log: "log", sp.sqrt: "sqrt",
    sp.sinh: "sinh", sp.cosh: "cosh", sp.tanh: "tanh",
    sp.Abs: "Abs",
    sp.Add: "Add", sp.Mul: "Mul", sp.Pow: "Pow",
    sp.Integral: "Integral",
}


def extract_operators(expr: sp.Expr) -> set[str]:
    """
    Recursively walk a SymPy expression tree and return the set of
    operator/function names found.

    Args:
        expr: A SymPy expression.

    Returns:
        Set of operator name strings (e.g. {"sin", "Add", "Pow"}).
    """
    ops: set[str] = set()

    def _walk(e: sp.Basic) -> None:
        func = type(e)
        if func in _OPERATOR_NAMES:
            ops.add(_OPERATOR_NAMES[func])
        for arg in e.args:
            _walk(arg)

    _walk(expr)
    return ops


def operator_f1(
    pred_expr: sp.Expr, gt_expr: sp.Expr
) -> dict[str, Any]:
    """
    Compute Operator F1 (precision, recall, F1) between predicted and
    ground-truth expressions based on the set of operators each contains.

    Args:
        pred_expr: Predicted SymPy expression.
        gt_expr: Ground-truth SymPy expression.

    Returns:
        Dict with precision, recall, f1, pred_ops, gt_ops.
    """
    pred_ops = extract_operators(pred_expr)
    gt_ops = extract_operators(gt_expr)

    if not pred_ops and not gt_ops:
        return {
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "pred_ops": [], "gt_ops": [],
        }

    tp = len(pred_ops & gt_ops)
    precision = tp / len(pred_ops) if pred_ops else 0.0
    recall = tp / len(gt_ops) if gt_ops else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_ops": sorted(pred_ops),
        "gt_ops": sorted(gt_ops),
    }


def _tokenize_math(text: str) -> list[str]:
    """Tokenize a math string by splitting on whitespace and operators."""
    # Insert spaces around mathematical operators so they become tokens
    text = re.sub(r"([+\-*/^()=,])", r" \1 ", text)
    return text.split()


def bleu_score(pred_str: str, gt_str: str) -> float:
    """
    Compute BLEU score between predicted and ground-truth solution strings.

    Uses nltk sentence_bleu with smoothing to avoid zero scores on short
    sequences.

    Args:
        pred_str: Predicted solution string.
        gt_str: Ground-truth solution string.

    Returns:
        BLEU score in [0.0, 1.0].
    """
    ref_tokens = _tokenize_math(gt_str)
    hyp_tokens = _tokenize_math(pred_str)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1
    return float(
        sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
    )


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


def _extract_term_coeffs(expr: sp.Expr) -> dict[sp.Expr, float]:
    """Extract numeric coefficients keyed by their base term expression."""
    terms = _split_series_terms(expr)
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


def evaluate_solutions(
    results_path: Path | str,
    mode: str = "both",
    symbolic_tolerance: float = 1e-10,
    numeric_tolerance: float = 1e-6,
    n_test_points: int = 100,
    type_tolerances: Optional[dict[str, float]] = None,
    include_points: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Evaluate solutions from a results file.

    Args:
        results_path: Path to results JSON/JSONL file.
        mode: Evaluation mode (symbolic, numeric, both).
        symbolic_tolerance: Tolerance for symbolic comparison.
        numeric_tolerance: Tolerance for numeric comparison.
        n_test_points: Number of test points for numeric evaluation.
        type_tolerances: Per-solution-type numeric tolerance overrides.
        **kwargs: Additional evaluation parameters.

    Returns:
        Dictionary with evaluation metrics.
    """
    import json

    if type_tolerances is None:
        type_tolerances = {}

    results_path = Path(results_path)
    logger.info(f"Evaluating solutions from {results_path}")

    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return {"error": f"File not found: {results_path}"}

    # Load results
    results: list[dict] = []
    if results_path.suffix == ".jsonl":
        with open(results_path) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    elif results_path.suffix == ".json":
        with open(results_path) as f:
            data = json.load(f)
            results = data if isinstance(data, list) else [data]
    else:
        logger.error(f"Unsupported file format: {results_path.suffix}")
        return {"error": f"Unsupported format: {results_path.suffix}"}

    logger.info(f"Loaded {len(results)} results")

    # Initialize evaluator
    evaluator = SolutionEvaluator(
        symbolic_tolerance=symbolic_tolerance,
        numeric_tolerance=numeric_tolerance,
        n_test_points=n_test_points,
    )

    # Track edge case metrics
    has_solution_correct = 0
    has_solution_total = 0
    solution_type_correct = 0
    solution_type_total = 0
    confusion_matrix: dict[str, int] = {}
    evaluated_count = 0
    errors: list[str] = []

    # None-type detection: TP/FP/FN for precision/recall/F1
    none_tp = 0  # GT=none AND pred says no solution
    none_fp = 0  # GT!=none AND pred says no solution
    none_fn = 0  # GT=none AND pred does NOT say no solution

    # Residual verification results (when equation components available)
    residual_results: list[dict[str, Any]] = []

    for i, result in enumerate(results):
        # Evaluate edge case metrics
        gt_has_solution = result.get("ground_truth_has_solution")
        pred_has_solution = result.get("has_solution")
        if gt_has_solution is not None and pred_has_solution is not None:
            has_solution_total += 1
            if gt_has_solution == pred_has_solution:
                has_solution_correct += 1

        gt_solution_type = result.get("ground_truth_solution_type")
        pred_solution_type = result.get("solution_type")
        if gt_solution_type and pred_solution_type:
            solution_type_total += 1
            if gt_solution_type == pred_solution_type:
                solution_type_correct += 1
            else:
                key = f"{gt_solution_type}_predicted_as_{pred_solution_type}"
                confusion_matrix[key] = confusion_matrix.get(key, 0) + 1

        # None-type detection tracking
        if gt_solution_type == "none":
            if pred_has_solution is False:
                none_tp += 1
            else:
                none_fn += 1
        elif gt_solution_type is not None and pred_has_solution is False:
            none_fp += 1

        # Extract domain from metadata
        domain = tuple(result.get("ground_truth_domain") or [0, 1])
        eval_points = result.get("evaluation_points") or result.get("metadata", {}).get(
            "evaluation_points"
        )

        # Branch on solution type: "none" type
        if gt_solution_type == "none":
            evaluator.evaluate_none_type(pred_has_solution)
            evaluated_count += 1
            continue

        # Evaluate solution accuracy
        ground_truth_str = result.get("ground_truth")
        solution_str = result.get("solution_str")

        if not ground_truth_str or not solution_str:
            continue

        try:
            gt_expr = parse_latex_to_sympy(ground_truth_str)
            pred_expr = parse_latex_to_sympy(solution_str)

            # Branch on solution type: "family" type
            if gt_solution_type == "family":
                evaluator.evaluate_family(
                    pred_expr,
                    gt_expr,
                    domain=domain,
                    evaluation_points=eval_points,
                    include_points=include_points,
                )
                evaluated_count += 1
                continue

            # Standard evaluation with per-type tolerance
            tol_override = (
                type_tolerances.get(gt_solution_type) if gt_solution_type else None
            )
            evaluator.evaluate(
                pred_expr,
                gt_expr,
                domain=domain,
                solution_type=gt_solution_type,
                numeric_tolerance_override=tol_override,
                evaluation_points=eval_points,
                include_points=include_points,
                pred_str=solution_str,
                gt_str=ground_truth_str,
            )
            evaluated_count += 1

            # Residual verification (when equation components available)
            kernel_str = result.get("ground_truth_kernel")
            f_str = result.get("ground_truth_f")
            lambda_val = result.get("ground_truth_lambda")
            if kernel_str and f_str and lambda_val is not None:
                try:
                    kernel_expr = parse_latex_to_sympy(kernel_str)
                    f_expr = parse_latex_to_sympy(f_str)
                    residual = verify_solution(
                        pred_expr, kernel_expr, f_expr,
                        float(lambda_val), domain=domain,
                        x_values=(eval_points or {}).get("x_values"),
                    )
                    residual_results.append(residual)
                except Exception as e_res:
                    logger.debug(
                        f"Residual verification failed for result {i}: {e_res}"
                    )

        except Exception as e:
            errors.append(f"Result {result.get('equation_id', i)}: {str(e)}")
            logger.debug(f"Failed to evaluate result {i}: {e}")

    # Get summary
    summary = evaluator.summary()

    # Build metrics
    metrics = {
        "mode": mode,
        **summary,
        "evaluated_count": evaluated_count,
        "total_results": len(results),
        "parse_errors": len(errors),
    }

    if confusion_matrix:
        metrics["confusion_matrix"] = confusion_matrix

    if has_solution_total > 0:
        metrics["has_solution_accuracy"] = has_solution_correct / has_solution_total
        metrics["has_solution_total"] = has_solution_total

    if solution_type_total > 0:
        metrics["solution_type_accuracy"] = solution_type_correct / solution_type_total
        metrics["solution_type_total"] = solution_type_total

    # None-type detection precision / recall / F1
    if none_tp + none_fp + none_fn > 0:
        none_prec = none_tp / (none_tp + none_fp) if (none_tp + none_fp) > 0 else 0.0
        none_rec = none_tp / (none_tp + none_fn) if (none_tp + none_fn) > 0 else 0.0
        none_f1 = (
            2 * none_prec * none_rec / (none_prec + none_rec)
            if (none_prec + none_rec) > 0 else 0.0
        )
        metrics["none_detection"] = {
            "precision": none_prec,
            "recall": none_rec,
            "f1": none_f1,
            "tp": none_tp,
            "fp": none_fp,
            "fn": none_fn,
        }

    # Aggregate residual verification
    residuals_valid = [r for r in residual_results if "error" not in r]
    if residuals_valid:
        verified = sum(1 for r in residuals_valid if r.get("verified", False))
        metrics["residual_verification"] = {
            "verified_count": verified,
            "total": len(residuals_valid),
            "verified_rate": verified / len(residuals_valid),
            "mean_residual_max": float(np.mean(
                [r["residual_max"] for r in residuals_valid]
            )),
            "mean_residual_mean": float(np.mean(
                [r["residual_mean"] for r in residuals_valid]
            )),
            "mean_residual_mae": float(np.mean(
                [r["residual_mae"] for r in residuals_valid]
            )),
            "mean_residual_rmse": float(np.mean(
                [r["residual_rmse"] for r in residuals_valid]
            )),
        }

    logger.info(f"Evaluation complete: {evaluated_count}/{len(results)} evaluated, accuracy: {summary.get('accuracy', 0):.2%}")

    return metrics


def symbolic_compare(
    solution: sp.Expr,
    ground_truth: sp.Expr,
    tolerance: float = 1e-10,
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


def verify_solution(
    solution: sp.Expr,
    kernel: sp.Expr,
    f: sp.Expr,
    lambda_val: float,
    domain: tuple[float, float] = (0, 1),
    x_values: Optional[list[float]] = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """
    Verify that a solution satisfies the Fredholm equation.

    Checks: u(x) - λ ∫_a^b K(x,t) u(t) dt = f(x)

    Args:
        solution: Proposed solution u(x).
        kernel: Kernel function K(x, t).
        f: Right-hand side f(x).
        lambda_val: Lambda parameter.
        domain: Integration domain (a, b).
        x_values: Optional list of x sample points for residual checks.
        tolerance: Tolerance for verification.

    Returns:
        Dictionary with verification results.
    """
    result = {
        "verified": False,
        "residual_max": float("inf"),
        "residual_mean": float("inf"),
        "residual_mae": float("inf"),
        "residual_rmse": float("inf"),
    }

    x, t = sp.symbols("x t")
    a, b = domain

    try:
        # Substitute u(t) = solution(t)
        u_of_t = solution.subs(x, t)

        # Compute the integral symbolically if possible
        integrand = kernel * u_of_t

        try:
            # Try symbolic integration
            integral = sp.integrate(integrand, (t, a, b))

            # Compute residual: u(x) - λ*integral - f(x)
            residual = solution - lambda_val * integral - f
            residual_simplified = sp.simplify(residual)

            if residual_simplified == 0:
                result["verified"] = True
                result["residual_max"] = 0.0
                result["residual_mean"] = 0.0
                result["residual_mae"] = 0.0
                result["residual_rmse"] = 0.0
                return result

        except Exception:
            logger.debug("Symbolic integration failed, using numeric verification")

        # Numeric verification
        n_points = 50
        test_x = None
        if x_values:
            test_x = np.array(x_values, dtype=float)
            if test_x.size == 0:
                test_x = None
        if test_x is None:
            rng = np.random.default_rng(0)
            test_x = rng.uniform(a, b, n_points)

        # Create numeric functions
        f_solution = sp.lambdify(x, solution, modules=["numpy"])
        f_f = sp.lambdify(x, f, modules=["numpy"])

        # Create kernel function of two variables
        f_kernel = sp.lambdify((x, t), kernel, modules=["numpy"])
        f_u_t = sp.lambdify(t, solution.subs(x, t), modules=["numpy"])

        residuals = []
        for xi in test_x:
            # Compute integral numerically
            def integrand_func(t_val):
                return f_kernel(xi, t_val) * f_u_t(t_val)

            integral_val, _ = integrate.quad(integrand_func, a, b)

            # Compute residual at this point
            lhs = f_solution(xi) - lambda_val * integral_val
            rhs = f_f(xi)
            residuals.append(abs(lhs - rhs))

        residuals = np.array(residuals)
        result["residual_max"] = float(np.max(residuals))
        result["residual_mean"] = float(np.mean(residuals))
        result["residual_mae"] = result["residual_mean"]
        result["residual_rmse"] = float(np.sqrt(np.mean(residuals**2)))
        result["verified"] = result["residual_max"] < tolerance

    except Exception as e:
        logger.warning(f"Solution verification failed: {e}")
        result["error"] = str(e)

    return result


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
                    sample_result["y_pred"] = y_solution.tolist()
                    sample_result["y_true"] = y_truth_arr.tolist()
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


def evaluate_discrete_points(
    pred_points: list[tuple[float, float]],
    gt_points: list[tuple[float, float]],
    x_tolerance: float = 1e-3,
    y_tolerance: float = 1e-3,
) -> dict[str, Any]:
    """
    Compare discrete point predictions with ground truth.

    Matches x-coordinates within tolerance, compares y-values at matched points.

    Args:
        pred_points: List of (x, y) tuples from LLM prediction.
        gt_points: List of (x, y) tuples from ground truth.
        x_tolerance: Tolerance for matching x-coordinates.
        y_tolerance: Tolerance for considering y-values as matching.

    Returns:
        Dictionary with point-wise comparison metrics.
    """
    if not pred_points or not gt_points:
        logger.warning(
            f"Empty discrete_points: pred={len(pred_points)}, gt={len(gt_points)}"
        )
        return {
            "match": False,
            "matched_points": 0,
            "total_points_pred": len(pred_points),
            "total_points_gt": len(gt_points),
            "accuracy": 0.0,
            "max_error": float("inf"),
            "mean_error": float("inf"),
            "mae": float("inf"),
            "rmse": float("inf"),
        }

    matched = 0
    errors = []
    y_differences = []

    for x_pred, y_pred in pred_points:
        # Find closest x-coordinate in ground truth
        if not gt_points:
            break

        closest_gt = min(gt_points, key=lambda p: abs(p[0] - x_pred))
        x_gt, y_gt = closest_gt

        # Check if x-coordinates match within tolerance
        x_diff = abs(x_pred - x_gt)
        if x_diff < x_tolerance:
            # Match found - compare y-values
            y_diff = abs(y_pred - y_gt)
            y_differences.append(y_diff)
            errors.append(y_diff)

            if y_diff < y_tolerance:
                matched += 1

    # Compute metrics
    result = {
        "match": matched > 0 and matched >= len(pred_points) * 0.8,  # 80% threshold
        "matched_points": matched,
        "total_points_pred": len(pred_points),
        "total_points_gt": len(gt_points),
        "accuracy": matched / len(pred_points) if pred_points else 0.0,
        "max_error": float(np.max(errors)) if errors else float("inf"),
        "mean_error": float(np.mean(errors)) if errors else float("inf"),
        "mae": float(np.mean(errors)) if errors else float("inf"),
        "rmse": float(np.sqrt(np.mean(np.array(errors) ** 2)))
        if errors
        else float("inf"),
    }

    return result


class SolutionEvaluator:
    """Evaluator class for batch solution evaluation."""

    def __init__(
        self,
        symbolic_tolerance: float = 1e-10,
        numeric_tolerance: float = 1e-6,
        n_test_points: int = 100,
        compute_bleu: bool = True,
    ) -> None:
        """Initialize the evaluator."""
        self.symbolic_tolerance = symbolic_tolerance
        self.numeric_tolerance = numeric_tolerance
        self.n_test_points = n_test_points
        self.compute_bleu = compute_bleu

        self.results: list[dict[str, Any]] = []

    def evaluate(
        self,
        solution: sp.Expr,
        ground_truth: sp.Expr,
        domain: tuple[float, float] = (0, 1),
        solution_type: Optional[str] = None,
        numeric_tolerance_override: Optional[float] = None,
        evaluation_points: Optional[dict[str, Any]] = None,
        include_points: bool = False,
        pred_str: Optional[str] = None,
        gt_str: Optional[str] = None,
    ) -> dict[str, Any]:
        """Evaluate a single solution.

        Args:
            solution: Predicted solution expression.
            ground_truth: Ground truth expression.
            domain: Evaluation domain.
            solution_type: Type of solution for per-type tracking.
            numeric_tolerance_override: Override numeric tolerance for this evaluation.
            pred_str: Raw predicted solution string (for BLEU).
            gt_str: Raw ground truth string (for BLEU).
        """
        tol = (
            numeric_tolerance_override
            if numeric_tolerance_override is not None
            else self.numeric_tolerance
        )
        symbolic = symbolic_compare(solution, ground_truth, self.symbolic_tolerance)
        if evaluation_points is not None:
            numeric = numeric_compare(
                solution,
                {"evaluation_points": evaluation_points, "u": str(ground_truth)},
                domain,
                self.n_test_points,
                tol,
                include_points=include_points,
            )
        else:
            numeric = numeric_compare(
                solution,
                ground_truth,
                domain,
                self.n_test_points,
                tol,
                include_points=include_points,
            )

        # Operator F1
        op_f1 = operator_f1(solution, ground_truth)

        result: dict[str, Any] = {
            "symbolic": symbolic,
            "numeric": numeric,
            "symbolic_match": symbolic.get("equivalent", False),
            "numeric_match": numeric.get("match", False),
            "correct": symbolic["equivalent"] or numeric["match"],
            "solution_type": solution_type,
            "operator_f1": op_f1,
        }

        if solution_type == "series":
            term_count = count_series_terms(solution)
            result["series_term_count"] = term_count
            result["series_term_target"] = 4
            result["series_term_match"] = term_count == 4
            result["series_term_eval"] = evaluate_series_terms(
                solution,
                ground_truth,
                domain=domain,
                n_points=self.n_test_points,
                tolerance=tol,
            )

        if solution_type == "approx_coef":
            result["approx_coef_eval"] = evaluate_approx_coeffs(
                solution,
                ground_truth,
                tolerance=tol,
                relative_tolerance=0.1,
            )

        # BLEU (only when raw strings are provided)
        if self.compute_bleu and pred_str and gt_str:
            result["bleu"] = bleu_score(pred_str, gt_str)

        self.results.append(result)
        return result

    def evaluate_none_type(self, pred_has_solution: Optional[bool]) -> dict[str, Any]:
        """Evaluate a 'none' type equation (no solution exists).

        Correct iff the LLM predicted has_solution=False.

        Args:
            pred_has_solution: The LLM's prediction of whether a solution exists.

        Returns:
            Evaluation result dict.
        """
        correct = pred_has_solution is False
        result = {
            "symbolic": {
                "equivalent": correct,
                "difference": None,
                "simplified_match": correct,
            },
            "numeric": {
                "match": correct,
                "max_error": 0.0 if correct else float("inf"),
                "mean_error": 0.0 if correct else float("inf"),
                "rmse": 0.0 if correct else float("inf"),
            },
            "symbolic_match": correct,
            "numeric_match": correct,
            "correct": correct,
            "solution_type": "none",
        }
        self.results.append(result)
        return result

    def evaluate_family(
        self,
        solution: sp.Expr,
        ground_truth: sp.Expr,
        domain: tuple[float, float] = (0, 1),
        evaluation_points: Optional[dict[str, Any]] = None,
        include_points: bool = False,
    ) -> dict[str, Any]:
        """Evaluate a 'family' type equation (non-unique solution).

        Uses family_compare first, falls back to standard symbolic/numeric.
        Correct if any method succeeds.

        Args:
            solution: Predicted solution expression.
            ground_truth: Ground truth family expression with free constants.
            domain: Evaluation domain.

        Returns:
            Evaluation result dict.
        """
        # Try family-specific comparison first
        family_match = family_compare(solution, ground_truth)

        # For numeric/symbolic fallback, substitute free constants with 1
        x = sp.Symbol("x")
        standard_vars = {"x", "t"}
        free_constants = [
            s for s in ground_truth.free_symbols if s.name not in standard_vars
        ]
        gt_concrete = ground_truth
        for c in free_constants:
            gt_concrete = gt_concrete.subs(c, 1)

        sol_concrete = _substitute_family_constants(solution, value=1.0)

        # Also try standard comparison as fallback
        symbolic = symbolic_compare(sol_concrete, gt_concrete, self.symbolic_tolerance)
        numeric = _family_numeric_compare_samples(
            solution,
            ground_truth,
            domain,
            self.n_test_points,
            self.numeric_tolerance,
            constant_samples=[-1.0, 1.0, 2.0],
            evaluation_points=evaluation_points,
            include_points=include_points,
        )
        term_eval = evaluate_series_terms(
            sol_concrete,
            gt_concrete,
            domain=domain,
            n_points=self.n_test_points,
            tolerance=self.numeric_tolerance,
        )
        param_eval = _family_param_metadata(solution, ground_truth)

        correct = family_match or symbolic["equivalent"] or numeric["match"]
        result = {
            "symbolic": symbolic,
            "numeric": numeric,
            "symbolic_match": symbolic.get("equivalent", False),
            "numeric_match": numeric.get("match", False),
            "family_match": family_match,
            "family_term_eval": term_eval,
            "family_param_eval": param_eval,
            "correct": correct,
            "solution_type": "family",
        }
        self.results.append(result)
        return result

    def evaluate_discrete_points_type(
        self,
        pred_points: list[tuple[float, float]],
        gt_points: list[tuple[float, float]],
        solution_type: str = "discrete_points",
    ) -> dict[str, Any]:
        """
        Evaluate a 'discrete_points' type equation.

        Compares predicted points with ground truth points using point-wise metrics.

        Args:
            pred_points: Predicted (x, y) points extracted from LLM response.
            gt_points: Ground truth (x, y) points.
            solution_type: Type label for tracking.

        Returns:
            Evaluation result dict with point-wise comparison metrics.
        """
        result_dict = evaluate_discrete_points(
            pred_points, gt_points, x_tolerance=1e-3, y_tolerance=self.numeric_tolerance
        )
        result_dict["pred_points"] = pred_points
        result_dict["gt_points"] = gt_points

        result = {
            "symbolic": {"equivalent": False},  # Not applicable for discrete points
            "numeric": result_dict,
            "symbolic_match": False,
            "numeric_match": result_dict.get("match", False),
            "correct": result_dict["match"],
            "solution_type": solution_type,
        }
        self.results.append(result)
        return result

    def summary(self) -> dict[str, Any]:
        """Get summary statistics including per-type breakdown."""
        if not self.results:
            return {"total": 0}

        total = len(self.results)
        correct = sum(1 for r in self.results if r["correct"])
        symbolic_correct = sum(
            1 for r in self.results if r.get("symbolic", {}).get("equivalent", False)
        )
        numeric_correct = sum(
            1 for r in self.results if r.get("numeric", {}).get("match", False)
        )

        summary: dict[str, Any] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
            "symbolic_accuracy": symbolic_correct / total,
            "numeric_accuracy": numeric_correct / total,
        }

        series_counts = [
            r.get("series_term_count")
            for r in self.results
            if r.get("solution_type") == "series"
            and r.get("series_term_count") is not None
        ]
        if series_counts:
            series_matches = sum(
                1
                for r in self.results
                if r.get("solution_type") == "series" and r.get("series_term_match")
            )
            series_total = len(series_counts)
            summary["series_term_stats"] = {
                "total": series_total,
                "match": series_matches,
                "match_rate": series_matches / series_total,
                "min": int(min(series_counts)),
                "max": int(max(series_counts)),
                "mean": float(np.mean(series_counts)),
                "target": 4,
            }

        approx_eval = [
            r.get("approx_coef_eval")
            for r in self.results
            if r.get("solution_type") == "approx_coef" and r.get("approx_coef_eval")
        ]
        if approx_eval:
            match_rates = [e.get("match_rate", 0.0) for e in approx_eval]
            mean_abs = [e.get("mean_abs_error", float("inf")) for e in approx_eval]
            mean_rel = [e.get("mean_rel_error", float("inf")) for e in approx_eval]
            summary["approx_coef_stats"] = {
                "total": len(approx_eval),
                "mean_match_rate": float(np.mean(match_rates)),
                "mean_abs_error": float(np.mean(mean_abs)),
                "mean_rel_error": float(np.mean(mean_rel)),
            }

        discrete_eval = [
            r.get("numeric")
            for r in self.results
            if r.get("solution_type") == "discrete_points" and r.get("numeric")
        ]
        if discrete_eval:
            matched = sum(e.get("matched_points", 0) for e in discrete_eval)
            total_gt = sum(e.get("total_points_gt", 0) for e in discrete_eval)
            accuracies = [e.get("accuracy", 0.0) for e in discrete_eval]
            rmses = [e.get("rmse", float("inf")) for e in discrete_eval]
            summary["discrete_points_stats"] = {
                "total": len(discrete_eval),
                "matched_point_rate": matched / total_gt if total_gt else 0.0,
                "mean_accuracy": float(np.mean(accuracies)),
                "mean_rmse": float(np.mean(rmses)),
            }

        family_params = [
            r.get("family_param_eval")
            for r in self.results
            if r.get("solution_type") == "family" and r.get("family_param_eval")
        ]
        if family_params:
            naming_rates = [
                1.0 if e.get("param_naming_valid") else 0.0 for e in family_params
            ]
            count_match = [
                1.0 if e.get("param_count_match") else 0.0 for e in family_params
            ]
            summary["family_param_stats"] = {
                "total": len(family_params),
                "naming_convention_rate": float(np.mean(naming_rates)),
                "param_count_match_rate": float(np.mean(count_match)),
            }

        # Aggregate Operator F1
        op_f1_results = [r["operator_f1"] for r in self.results if "operator_f1" in r]
        if op_f1_results:
            summary["mean_operator_precision"] = sum(
                r["precision"] for r in op_f1_results
            ) / len(op_f1_results)
            summary["mean_operator_recall"] = sum(
                r["recall"] for r in op_f1_results
            ) / len(op_f1_results)
            summary["mean_operator_f1"] = sum(
                r["f1"] for r in op_f1_results
            ) / len(op_f1_results)

        # Aggregate relative L2 error
        rel_l2_values = [
            r["numeric"]["rel_l2"] for r in self.results
            if r.get("numeric", {}).get("rel_l2") is not None
            and r["numeric"]["rel_l2"] != float("inf")
        ]
        if rel_l2_values:
            summary["mean_rel_l2"] = sum(rel_l2_values) / len(rel_l2_values)

        # Aggregate BLEU
        bleu_results = [r["bleu"] for r in self.results if "bleu" in r]
        if bleu_results:
            summary["mean_bleu"] = sum(bleu_results) / len(bleu_results)

        # Per-type breakdown
        per_type: dict[str, dict[str, Any]] = {}
        for r in self.results:
            st = r.get("solution_type") or "unknown"
            if st not in per_type:
                per_type[st] = {"total": 0, "correct": 0}
            per_type[st]["total"] += 1
            if r["correct"]:
                per_type[st]["correct"] += 1

        for _st, counts in per_type.items():
            counts["accuracy"] = (
                counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
            )

        summary["per_type"] = per_type
        return summary
