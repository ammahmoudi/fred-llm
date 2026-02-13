"""Core evaluation orchestration for Fredholm solutions."""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import sympy as sp

from src.evaluation.metrics import (
    bleu_score,
    numeric_compare,
    operator_f1,
    symbolic_compare,
)
from src.evaluation.types import (
    evaluate_approx_coeffs,
    evaluate_discrete_points,
    evaluate_series_terms,
    family_compare,
    verify_solution,
)
from src.evaluation.types.family import (
    _family_numeric_compare_samples,
    _family_param_metadata,
    _substitute_family_constants,
)
from src.evaluation.types.series import count_series_terms
from src.llm.math_verify_adapter import parse_latex_to_sympy
from src.utils.logging_utils import get_logger

from src.evaluation.types.family import (
    _family_numeric_compare_samples,
    _family_param_metadata,
    _substitute_family_constants,
)


def evaluate_solutions(
    results_path: Path | str,
    mode: str = "both",
    symbolic_tolerance: float = 1e-10,
    numeric_tolerance: float = 1e-6,
    n_test_points: int = 100,
    use_math_verify: bool = True,
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
        use_math_verify: Whether to use Math-Verify when available.
        type_tolerances: Per-solution-type numeric tolerance overrides.
        **kwargs: Additional evaluation parameters.

    Returns:
        Dictionary with evaluation metrics.
    """
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
        use_math_verify=use_math_verify,
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
            gt_expr = parse_latex_to_sympy(
                ground_truth_str, use_math_verify=use_math_verify
            )
            pred_expr = parse_latex_to_sympy(
                solution_str, use_math_verify=use_math_verify
            )

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
                    kernel_expr = parse_latex_to_sympy(
                        kernel_str, use_math_verify=use_math_verify
                    )
                    f_expr = parse_latex_to_sympy(
                        f_str, use_math_verify=use_math_verify
                    )
                    residual = verify_solution(
                        pred_expr,
                        kernel_expr,
                        f_expr,
                        float(lambda_val),
                        domain=domain,
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
            if (none_prec + none_rec) > 0
            else 0.0
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
            "mean_residual_max": float(
                np.mean([r["residual_max"] for r in residuals_valid])
            ),
            "mean_residual_mean": float(
                np.mean([r["residual_mean"] for r in residuals_valid])
            ),
            "mean_residual_mae": float(
                np.mean([r["residual_mae"] for r in residuals_valid])
            ),
            "mean_residual_rmse": float(
                np.mean([r["residual_rmse"] for r in residuals_valid])
            ),
        }

    logger.info(
        f"Evaluation complete: {evaluated_count}/{len(results)} evaluated, accuracy: {summary.get('accuracy', 0):.2%}"
    )

    return metrics


class SolutionEvaluator:
    """Evaluator class for batch solution evaluation."""

    def __init__(
        self,
        symbolic_tolerance: float = 1e-10,
        numeric_tolerance: float = 1e-6,
        n_test_points: int = 100,
        compute_bleu: bool = True,
        use_math_verify: bool = True,
    ) -> None:
        """Initialize the evaluator."""
        self.symbolic_tolerance = symbolic_tolerance
        self.numeric_tolerance = numeric_tolerance
        self.n_test_points = n_test_points
        self.compute_bleu = compute_bleu
        self.use_math_verify = use_math_verify

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
        symbolic = symbolic_compare(
            solution,
            ground_truth,
            self.symbolic_tolerance,
            use_math_verify=self.use_math_verify,
        )
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
        symbolic = symbolic_compare(
            sol_concrete,
            gt_concrete,
            self.symbolic_tolerance,
            use_math_verify=self.use_math_verify,
        )
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
            summary["mean_operator_f1"] = sum(r["f1"] for r in op_f1_results) / len(
                op_f1_results
            )

        # Aggregate relative L2 error
        rel_l2_values = [
            r["numeric"]["rel_l2"]
            for r in self.results
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
