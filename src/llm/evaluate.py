"""
Evaluation utilities for Fredholm equation solutions.

Provides both symbolic and numeric evaluation metrics.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import sympy as sp
from scipy import integrate

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_solutions(
    results_path: Path | str,
    mode: str = "both",
    symbolic_tolerance: float = 1e-10,
    numeric_tolerance: float = 1e-6,
    n_test_points: int = 100,
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
        **kwargs: Additional evaluation parameters.

    Returns:
        Dictionary with evaluation metrics.
    """
    import json
    from sympy.parsing.sympy_parser import (
        implicit_multiplication,
        parse_expr,
        standard_transformations,
    )

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

    # Define symbols for parsing
    x, t = sp.symbols("x t")
    local_dict = {"x": x, "t": t, "e": sp.E, "pi": sp.pi}
    transformations = standard_transformations + (implicit_multiplication,)

    # Track edge case metrics
    has_solution_correct = 0
    has_solution_total = 0
    solution_type_correct = 0
    solution_type_total = 0
    evaluated_count = 0
    errors: list[str] = []

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

        # Evaluate solution accuracy
        ground_truth_str = result.get("ground_truth")
        solution_str = result.get("solution_str")

        if not ground_truth_str or not solution_str:
            continue

        try:
            # Parse ground truth
            gt_expr = parse_expr(
                ground_truth_str,
                local_dict=local_dict,
                transformations=transformations,
            )

            # Parse predicted solution
            pred_expr = parse_expr(
                solution_str,
                local_dict=local_dict,
                transformations=transformations,
            )

            # Run evaluation based on mode
            if mode in ("symbolic", "both"):
                evaluator.evaluate(pred_expr, gt_expr, domain=(0, 1))
                evaluated_count += 1
            elif mode == "numeric":
                # Numeric only
                evaluator.evaluate(pred_expr, gt_expr, domain=(0, 1))
                evaluated_count += 1

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
    }

    try:
        x = sp.Symbol("x")

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


class SolutionEvaluator:
    """Evaluator class for batch solution evaluation."""

    def __init__(
        self,
        symbolic_tolerance: float = 1e-10,
        numeric_tolerance: float = 1e-6,
        n_test_points: int = 100,
    ) -> None:
        """Initialize the evaluator."""
        self.symbolic_tolerance = symbolic_tolerance
        self.numeric_tolerance = numeric_tolerance
        self.n_test_points = n_test_points

        self.results: list[dict[str, Any]] = []

    def evaluate(
        self,
        solution: sp.Expr,
        ground_truth: sp.Expr,
        domain: tuple[float, float] = (0, 1),
    ) -> dict[str, Any]:
        """Evaluate a single solution."""
        symbolic = symbolic_compare(solution, ground_truth, self.symbolic_tolerance)
        numeric = numeric_compare(
            solution, ground_truth, domain, self.n_test_points, self.numeric_tolerance
        )

        result = {
            "symbolic": symbolic,
            "numeric": numeric,
            "correct": symbolic["equivalent"] or numeric["match"],
        }
        self.results.append(result)
        return result

    def summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        if not self.results:
            return {"total": 0}

        total = len(self.results)
        correct = sum(1 for r in self.results if r["correct"])
        symbolic_correct = sum(1 for r in self.results if r["symbolic"]["equivalent"])
        numeric_correct = sum(1 for r in self.results if r["numeric"]["match"])

        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
            "symbolic_accuracy": symbolic_correct / total,
            "numeric_accuracy": numeric_correct / total,
        }
