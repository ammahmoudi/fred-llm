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


def evaluate_solutions(
    results_path: Path | str,
    mode: str = "both",
    symbolic_tolerance: float = 1e-10,
    numeric_tolerance: float = 1e-6,
    n_test_points: int = 100,
    type_tolerances: Optional[dict[str, float]] = None,
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
                evaluator.evaluate_family(pred_expr, gt_expr, domain=domain)
                evaluated_count += 1
                continue

            # Standard evaluation with per-type tolerance
            tol_override = type_tolerances.get(gt_solution_type) if gt_solution_type else None
            evaluator.evaluate(
                pred_expr, gt_expr, domain=domain,
                solution_type=gt_solution_type,
                numeric_tolerance_override=tol_override,
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
    ground_truth: sp.Expr,
    domain: tuple[float, float] = (0, 1),
    n_points: int = 100,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """
    Compare two expressions numerically over a domain.

    Args:
        solution: Generated solution as SymPy expression.
        ground_truth: Expected solution as SymPy expression.
        domain: Integration domain (a, b).
        n_points: Number of test points.
        tolerance: Tolerance for numeric comparison.

    Returns:
        Dictionary with numeric comparison results.
    """
    result = {
        "match": False,
        "max_error": float("inf"),
        "mean_error": float("inf"),
        "rmse": float("inf"),
        "rel_l2": float("inf"),
    }

    try:
        x = sp.Symbol("x")

        # Evaluate any unevaluated Integral objects before lambdify
        if solution.has(sp.Integral):
            solution = solution.doit()
        if ground_truth.has(sp.Integral):
            ground_truth = ground_truth.doit()

        # Check that expressions only depend on x (no other free symbols)
        extra_sol = solution.free_symbols - {x}
        extra_gt = ground_truth.free_symbols - {x}
        if extra_sol or extra_gt:
            extra = extra_sol | extra_gt
            result["error"] = f"Expressions contain non-numeric symbols: {extra}"
            logger.debug(f"Numeric comparison skipped: non-numeric symbols {extra}")
            return result

        # Convert to numeric functions
        f_solution = sp.lambdify(x, solution, modules=["numpy"])
        f_truth = sp.lambdify(x, ground_truth, modules=["numpy"])

        # Generate test points
        a, b = domain
        test_points = np.linspace(a, b, n_points)

        # Evaluate
        y_solution = np.array([f_solution(xi) for xi in test_points])
        y_truth = np.array([f_truth(xi) for xi in test_points])

        # Compute errors
        errors = np.abs(y_solution - y_truth)
        result["max_error"] = float(np.max(errors))
        result["mean_error"] = float(np.mean(errors))
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
        tolerance: Tolerance for verification.

    Returns:
        Dictionary with verification results.
    """
    result = {
        "verified": False,
        "residual_max": float("inf"),
        "residual_mean": float("inf"),
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
                return result

        except Exception:
            logger.debug("Symbolic integration failed, using numeric verification")

        # Numeric verification
        n_points = 50
        test_x = np.linspace(a, b, n_points)

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
        s for s in ground_truth.free_symbols
        if s.name not in standard_vars
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
        tol = numeric_tolerance_override if numeric_tolerance_override is not None else self.numeric_tolerance
        symbolic = symbolic_compare(solution, ground_truth, self.symbolic_tolerance)
        numeric = numeric_compare(
            solution, ground_truth, domain, self.n_test_points, tol
        )

        # Operator F1
        op_f1 = operator_f1(solution, ground_truth)

        result: dict[str, Any] = {
            "symbolic": symbolic,
            "numeric": numeric,
            "correct": symbolic["equivalent"] or numeric["match"],
            "solution_type": solution_type,
            "operator_f1": op_f1,
        }

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
            "symbolic": {"equivalent": correct, "difference": None, "simplified_match": correct},
            "numeric": {"match": correct, "max_error": 0.0 if correct else float("inf"),
                        "mean_error": 0.0 if correct else float("inf"),
                        "rmse": 0.0 if correct else float("inf")},
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
        # so expressions can be evaluated numerically
        x = sp.Symbol("x")
        standard_vars = {"x", "t"}
        free_constants = [
            s for s in ground_truth.free_symbols
            if s.name not in standard_vars
        ]
        gt_concrete = ground_truth
        for c in free_constants:
            gt_concrete = gt_concrete.subs(c, 1)

        # Also try standard comparison as fallback
        symbolic = symbolic_compare(solution, gt_concrete, self.symbolic_tolerance)
        numeric = numeric_compare(
            solution, gt_concrete, domain, self.n_test_points, self.numeric_tolerance
        )

        correct = family_match or symbolic["equivalent"] or numeric["match"]
        result = {
            "symbolic": symbolic,
            "numeric": numeric,
            "family_match": family_match,
            "correct": correct,
            "solution_type": "family",
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
            1 for r in self.results
            if r.get("symbolic", {}).get("equivalent", False)
        )
        numeric_correct = sum(
            1 for r in self.results
            if r.get("numeric", {}).get("match", False)
        )

        summary: dict[str, Any] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
            "symbolic_accuracy": symbolic_correct / total,
            "numeric_accuracy": numeric_correct / total,
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
            counts["accuracy"] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0

        summary["per_type"] = per_type
        return summary
