"""
Tests for evaluation module.
"""

import json
import tempfile
from pathlib import Path

import pytest
import sympy as sp

from src.evaluation import SolutionEvaluator, evaluate_solutions
from src.evaluation.metrics import (
    bleu_score,
    numeric_compare,
    operator_f1,
    symbolic_compare,
)
from src.evaluation.metrics.operator_f1 import extract_operators
from src.evaluation.types import verify_solution


class TestSymbolicCompare:
    """Tests for symbolic comparison."""

    def test_identical_expressions(self) -> None:
        """Test comparison of identical expressions."""
        x = sp.Symbol("x")
        expr = x**2 + 2 * x + 1

        result = symbolic_compare(expr, expr)

        assert result["equivalent"] is True
        assert result["simplified_match"] is True

    def test_equivalent_expressions(self) -> None:
        """Test comparison of algebraically equivalent expressions."""
        x = sp.Symbol("x")
        expr1 = (x + 1) ** 2
        expr2 = x**2 + 2 * x + 1

        result = symbolic_compare(expr1, expr2)

        assert result["equivalent"] is True

    def test_different_expressions(self) -> None:
        """Test comparison of different expressions."""
        x = sp.Symbol("x")
        expr1 = x**2
        expr2 = x**3

        result = symbolic_compare(expr1, expr2)

        assert result["equivalent"] is False
        assert result["difference"] is not None

    def test_trigonometric_identity(self) -> None:
        """Test comparison using trigonometric identity."""
        x = sp.Symbol("x")
        expr1 = sp.sin(x) ** 2 + sp.cos(x) ** 2
        expr2 = sp.Integer(1)

        result = symbolic_compare(expr1, expr2)

        assert result["equivalent"] is True


class TestNumericCompare:
    """Tests for numeric comparison."""

    def test_identical_functions(self) -> None:
        """Test numeric comparison of identical functions."""
        x = sp.Symbol("x")
        expr = x**2

        result = numeric_compare(expr, expr, domain=(0, 1))

        assert result["match"] is True
        assert result["max_error"] < 1e-10
        assert result["mean_error"] < 1e-10

    def test_close_functions(self) -> None:
        """Test numeric comparison of close functions."""
        x = sp.Symbol("x")
        expr1 = x**2
        expr2 = x**2 + 1e-8

        result = numeric_compare(expr1, expr2, domain=(0, 1), tolerance=1e-6)

        assert result["match"] is True
        assert result["max_error"] < 1e-6

    def test_different_functions(self) -> None:
        """Test numeric comparison of different functions."""
        x = sp.Symbol("x")
        expr1 = x
        expr2 = x**2

        result = numeric_compare(expr1, expr2, domain=(0, 1))

        assert result["match"] is False
        assert result["max_error"] > 0

    def test_custom_domain(self) -> None:
        """Test numeric comparison with custom domain."""
        x = sp.Symbol("x")
        expr = x

        result = numeric_compare(expr, expr, domain=(-1, 1), n_points=50)

        assert result["match"] is True


class TestVerifySolution:
    """Tests for solution verification."""

    def test_simple_valid_solution(self) -> None:
        """Test verification of a simple valid solution."""
        x, t = sp.symbols("x t")

        # u(x) - ∫_0^1 x*t * u(t) dt = x
        # Solution: u(x) = 3x/2
        solution = sp.Rational(3, 2) * x
        kernel = x * t
        f = x
        lambda_val = 1.0

        result = verify_solution(solution, kernel, f, lambda_val, domain=(0, 1))

        # Should verify (residual close to 0)
        assert result["residual_max"] < 0.1 or result["verified"]

    def test_handles_errors_gracefully(self) -> None:
        """Test that verification handles errors gracefully."""
        x, t = sp.symbols("x t")

        # Invalid expressions that might cause issues
        solution = sp.Integer(1) / x  # Could have issues at x=0
        kernel = x * t
        f = x
        lambda_val = 1.0

        result = verify_solution(solution, kernel, f, lambda_val, domain=(0.1, 1))

        # Should return a result without crashing
        assert "verified" in result or "error" in result


class TestSolutionEvaluator:
    """Tests for SolutionEvaluator class."""

    def test_evaluate_single_solution(self) -> None:
        """Test evaluating a single solution."""
        x = sp.Symbol("x")

        evaluator = SolutionEvaluator()
        result = evaluator.evaluate(x**2, x**2, domain=(0, 1))

        assert result["correct"] is True
        assert result["symbolic"]["equivalent"] is True

    def test_summary_with_no_results(self) -> None:
        """Test summary with no results."""
        evaluator = SolutionEvaluator()
        summary = evaluator.summary()

        assert summary["total"] == 0

    def test_summary_with_results(self) -> None:
        """Test summary after evaluating multiple solutions."""
        x = sp.Symbol("x")
        evaluator = SolutionEvaluator()

        # Add correct and incorrect solutions
        evaluator.evaluate(x**2, x**2, domain=(0, 1))  # Correct
        evaluator.evaluate(x, x, domain=(0, 1))  # Correct
        evaluator.evaluate(x, x**2, domain=(0, 1))  # Incorrect

        summary = evaluator.summary()

        assert summary["total"] == 3
        assert summary["correct"] == 2
        assert summary["accuracy"] == pytest.approx(2 / 3)

    def test_series_term_count_metadata(self) -> None:
        """Test series term count metadata is recorded."""
        x = sp.Symbol("x")
        evaluator = SolutionEvaluator()

        expr = x + x**2 + x**3 + x**4
        result = evaluator.evaluate(expr, expr, domain=(0, 1), solution_type="series")

        assert result["series_term_count"] == 4
        assert result["series_term_target"] == 4
        assert result["series_term_match"] is True
        assert result["series_term_eval"]["match"] is True
        assert result["series_term_eval"]["terms_compared"] == 4

    def test_series_term_stats_in_summary(self) -> None:
        """Test series term stats are included in summary."""
        x = sp.Symbol("x")
        evaluator = SolutionEvaluator()

        expr = x + x**2 + x**3 + x**4
        evaluator.evaluate(expr, expr, domain=(0, 1), solution_type="series")
        evaluator.evaluate(expr, expr, domain=(0, 1), solution_type="series")

        summary = evaluator.summary()

        assert "series_term_stats" in summary
        stats = summary["series_term_stats"]
        assert stats["total"] == 2
        assert stats["match"] == 2
        assert stats["target"] == 4

    def test_approx_coef_eval_metadata(self) -> None:
        """Test approx_coef evaluation metadata is recorded."""
        x = sp.Symbol("x")
        evaluator = SolutionEvaluator()

        expr = 2 * x + 3 * sp.sin(x)
        result = evaluator.evaluate(
            expr, expr, domain=(0, 1), solution_type="approx_coef"
        )

        assert result["approx_coef_eval"]["match"] is True
        assert result["approx_coef_eval"]["terms_compared"] == 2

    def test_approx_coef_stats_in_summary(self) -> None:
        """Test approx_coef stats are included in summary."""
        x = sp.Symbol("x")
        evaluator = SolutionEvaluator()

        expr = 2 * x + 3 * sp.sin(x)
        evaluator.evaluate(expr, expr, domain=(0, 1), solution_type="approx_coef")

        summary = evaluator.summary()

        assert "approx_coef_stats" in summary
        assert summary["approx_coef_stats"]["total"] == 1

    def test_family_term_eval_metadata(self) -> None:
        """Test family term evaluation metadata is recorded."""
        x = sp.Symbol("x")
        C = sp.Symbol("C")
        evaluator = SolutionEvaluator()

        expr = C * x
        result = evaluator.evaluate_family(expr, expr, domain=(0, 1))

        assert result["family_term_eval"]["match"] is True
        assert result["family_term_eval"]["terms_compared"] == 1
        assert result["family_param_eval"]["param_count_pred"] == 1
        assert result["family_param_eval"]["param_count_gt"] == 1
        assert result["family_param_eval"]["param_count_match"] is True
        assert result["family_param_eval"]["param_naming_valid"] is True


class TestEvaluateSolutions:
    """Tests for evaluate_solutions function."""

    def test_file_not_found(self) -> None:
        """Test handling of missing file."""
        result = evaluate_solutions("/nonexistent/path.jsonl")

        assert "error" in result
        assert "not found" in result["error"]

    def test_unsupported_format(self) -> None:
        """Test handling of unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            result = evaluate_solutions(temp_path)
            assert "error" in result
        finally:
            Path(temp_path).unlink()

    def test_evaluate_jsonl_file(self) -> None:
        """Test evaluating results from JSONL file."""
        # Create test JSONL file
        results = [
            {
                "equation_id": "eq_1",
                "solution_str": "x**2",
                "ground_truth": "x**2",
                "has_solution": True,
                "ground_truth_has_solution": True,
                "solution_type": "exact_symbolic",
                "ground_truth_solution_type": "exact_symbolic",
            },
            {
                "equation_id": "eq_2",
                "solution_str": "x",
                "ground_truth": "x",
                "has_solution": True,
                "ground_truth_has_solution": True,
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
            temp_path = f.name

        try:
            metrics = evaluate_solutions(temp_path)

            assert metrics["total_results"] == 2
            assert metrics["evaluated_count"] == 2
            assert metrics["accuracy"] == 1.0
            assert metrics["has_solution_accuracy"] == 1.0
        finally:
            Path(temp_path).unlink()

    def test_evaluate_json_file(self) -> None:
        """Test evaluating results from JSON file."""
        results = [
            {
                "equation_id": "eq_1",
                "solution_str": "x",
                "ground_truth": "x",
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f)
            temp_path = f.name

        try:
            metrics = evaluate_solutions(temp_path)

            assert metrics["total_results"] == 1
            assert metrics["evaluated_count"] == 1
        finally:
            Path(temp_path).unlink()

    def test_handles_parse_errors(self) -> None:
        """Test that parse errors are counted but don't crash."""
        results = [
            {
                "equation_id": "eq_1",
                "solution_str": "invalid math $$$",
                "ground_truth": "x",
            },
            {
                "equation_id": "eq_2",
                "solution_str": "x**2",
                "ground_truth": "x**2",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
            temp_path = f.name

        try:
            metrics = evaluate_solutions(temp_path)

            assert metrics["total_results"] == 2
            # One should fail to parse, one should succeed
            assert metrics["evaluated_count"] >= 1
        finally:
            Path(temp_path).unlink()


class TestExtractOperators:
    """Tests for extract_operators."""

    def test_simple_polynomial(self) -> None:
        """Polynomial x**2 + x contains Add and Pow."""
        x = sp.Symbol("x")
        ops = extract_operators(x**2 + x)
        assert "Add" in ops
        assert "Pow" in ops

    def test_trig_expression(self) -> None:
        """sin(x) + cos(x) contains sin, cos, Add."""
        x = sp.Symbol("x")
        ops = extract_operators(sp.sin(x) + sp.cos(x))
        assert ops == {"sin", "cos", "Add"}

    def test_nested_expression(self) -> None:
        """exp(sin(x)) contains exp and sin."""
        x = sp.Symbol("x")
        ops = extract_operators(sp.exp(sp.sin(x)))
        assert "exp" in ops
        assert "sin" in ops

    def test_constant_expression(self) -> None:
        """A plain number has no operators."""
        ops = extract_operators(sp.Integer(5))
        assert ops == set()

    def test_single_symbol(self) -> None:
        """A single symbol has no operators."""
        x = sp.Symbol("x")
        ops = extract_operators(x)
        assert ops == set()

    def test_integral(self) -> None:
        """An Integral node is detected."""
        x, t = sp.symbols("x t")
        expr = sp.Integral(sp.sin(t), (t, 0, x))
        ops = extract_operators(expr)
        assert "Integral" in ops
        assert "sin" in ops


class TestOperatorF1:
    """Tests for operator_f1."""

    def test_identical_operators(self) -> None:
        """Same expression → perfect scores."""
        x = sp.Symbol("x")
        expr = sp.sin(x) + sp.cos(x)
        result = operator_f1(expr, expr)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_disjoint_operators(self) -> None:
        """Completely different operators → zero F1."""
        x = sp.Symbol("x")
        pred = sp.sin(x)
        gt = sp.exp(x)
        result = operator_f1(pred, gt)
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"] == pytest.approx(0.0)
        assert result["f1"] == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Overlapping operator sets → partial F1."""
        x = sp.Symbol("x")
        pred = sp.sin(x) + sp.exp(x)  # {sin, exp, Add}
        gt = sp.sin(x) + sp.cos(x)  # {sin, cos, Add}
        result = operator_f1(pred, gt)
        # intersection = {sin, Add} → 2
        # precision = 2/3, recall = 2/3, f1 = 2/3
        assert result["precision"] == pytest.approx(2 / 3)
        assert result["recall"] == pytest.approx(2 / 3)
        assert result["f1"] == pytest.approx(2 / 3)

    def test_both_empty(self) -> None:
        """Two constants with no operators → perfect F1."""
        result = operator_f1(sp.Integer(1), sp.Integer(2))
        assert result["f1"] == pytest.approx(1.0)

    def test_pred_superset(self) -> None:
        """Prediction has extra operators → recall=1, precision<1."""
        x = sp.Symbol("x")
        pred = sp.sin(x) + sp.cos(x)  # {sin, cos, Add}
        gt = sp.sin(x)  # {sin}
        result = operator_f1(pred, gt)
        assert result["recall"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1 / 3)


class TestBleuScore:
    """Tests for bleu_score."""

    def test_identical_strings(self) -> None:
        """Identical strings → BLEU = 1.0."""
        score = bleu_score("x**2 + 1", "x**2 + 1")
        assert score == pytest.approx(1.0)

    def test_completely_different(self) -> None:
        """Completely different strings → low BLEU."""
        score = bleu_score("sin(x)", "a*b*c*d*e*f")
        assert score < 0.3

    def test_partial_match(self) -> None:
        """Partially matching strings → moderate BLEU."""
        score = bleu_score("x**2 + 2*x + 1", "x**2 + 3*x + 1")
        assert 0.3 < score < 1.0

    def test_empty_pred(self) -> None:
        """Empty prediction → 0.0."""
        score = bleu_score("", "x**2")
        assert score == pytest.approx(0.0)

    def test_empty_gt(self) -> None:
        """Empty ground truth → 0.0."""
        score = bleu_score("x**2", "")
        assert score == pytest.approx(0.0)

    def test_returns_float(self) -> None:
        """Score is always a float in [0, 1]."""
        score = bleu_score("x + 1", "x + 2")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestRelativeL2:
    """Tests for relative L2 error in numeric_compare."""

    def test_identical_has_zero_rel_l2(self) -> None:
        """Identical expressions have rel_l2 = 0."""
        x = sp.Symbol("x")
        result = numeric_compare(x**2, x**2, domain=(0, 1))
        assert result["rel_l2"] < 1e-10

    def test_rel_l2_present_in_result(self) -> None:
        """rel_l2 key is always present."""
        x = sp.Symbol("x")
        result = numeric_compare(x, x**2, domain=(0, 1))
        assert "rel_l2" in result

    def test_rel_l2_scale_invariance(self) -> None:
        """rel_l2 is the same whether GT is large or small (scale-invariant)."""
        x = sp.Symbol("x")
        # pred = gt + 0.1*gt = 1.1*gt  →  rel_l2 ≈ 0.1 regardless of scale
        small_gt = x
        small_pred = sp.Rational(11, 10) * x
        big_gt = 1000 * x
        big_pred = 1100 * x

        r_small = numeric_compare(small_pred, small_gt, domain=(0.1, 1))
        r_big = numeric_compare(big_pred, big_gt, domain=(0.1, 1))

        assert r_small["rel_l2"] == pytest.approx(r_big["rel_l2"], rel=1e-6)

    def test_rel_l2_zero_gt(self) -> None:
        """When GT is identically zero, rel_l2 is inf for nonzero pred."""
        x = sp.Symbol("x")
        result = numeric_compare(x, sp.Integer(0), domain=(0, 1))
        assert result["rel_l2"] == float("inf")

    def test_rel_l2_both_zero(self) -> None:
        """When both are zero, rel_l2 is 0."""
        result = numeric_compare(sp.Integer(0), sp.Integer(0), domain=(0, 1))
        assert result["rel_l2"] == pytest.approx(0.0)

    def test_summary_includes_mean_rel_l2(self) -> None:
        """SolutionEvaluator summary includes mean_rel_l2."""
        x = sp.Symbol("x")
        evaluator = SolutionEvaluator()
        evaluator.evaluate(x**2, x**2, domain=(0, 1))
        evaluator.evaluate(x, x, domain=(0, 1))

        summary = evaluator.summary()
        assert "mean_rel_l2" in summary
        assert summary["mean_rel_l2"] < 1e-10


class TestNoneDetectionPRF1:
    """Tests for none-type precision/recall/F1 in evaluate_solutions."""

    def _write_jsonl(self, results: list[dict]) -> str:
        """Helper to write results to a temp JSONL file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
            return f.name

    def test_perfect_none_detection(self) -> None:
        """All none-types correctly identified, no false positives."""
        results = [
            {
                "equation_id": "eq_1",
                "ground_truth_solution_type": "none",
                "has_solution": False,
                "ground_truth_has_solution": False,
            },
            {
                "equation_id": "eq_2",
                "solution_str": "x",
                "ground_truth": "x",
                "ground_truth_solution_type": "exact_symbolic",
                "has_solution": True,
                "ground_truth_has_solution": True,
            },
        ]
        path = self._write_jsonl(results)
        try:
            metrics = evaluate_solutions(path)
            nd = metrics["none_detection"]
            assert nd["precision"] == pytest.approx(1.0)
            assert nd["recall"] == pytest.approx(1.0)
            assert nd["f1"] == pytest.approx(1.0)
            assert nd["tp"] == 1
            assert nd["fp"] == 0
            assert nd["fn"] == 0
        finally:
            Path(path).unlink()

    def test_missed_none_type(self) -> None:
        """Model fails to detect a none-type (FN)."""
        results = [
            {
                "equation_id": "eq_1",
                "ground_truth_solution_type": "none",
                "has_solution": True,  # Wrong: should be False
                "ground_truth_has_solution": False,
            },
        ]
        path = self._write_jsonl(results)
        try:
            metrics = evaluate_solutions(path)
            nd = metrics["none_detection"]
            assert nd["tp"] == 0
            assert nd["fn"] == 1
            assert nd["recall"] == pytest.approx(0.0)
        finally:
            Path(path).unlink()

    def test_false_positive_none(self) -> None:
        """Model claims no solution when one exists (FP)."""
        results = [
            {
                "equation_id": "eq_1",
                "solution_str": "x",
                "ground_truth": "x",
                "ground_truth_solution_type": "exact_symbolic",
                "has_solution": False,  # Wrong: solution does exist
                "ground_truth_has_solution": True,
            },
        ]
        path = self._write_jsonl(results)
        try:
            metrics = evaluate_solutions(path)
            nd = metrics["none_detection"]
            assert nd["fp"] == 1
            assert nd["tp"] == 0
            assert nd["precision"] == pytest.approx(0.0)
        finally:
            Path(path).unlink()

    def test_mixed_scenario(self) -> None:
        """Mixed: 1 TP, 1 FP, 1 FN → precision=0.5, recall=0.5, F1=0.5."""
        results = [
            {  # TP: none correctly detected
                "equation_id": "eq_1",
                "ground_truth_solution_type": "none",
                "has_solution": False,
                "ground_truth_has_solution": False,
            },
            {  # FN: none missed
                "equation_id": "eq_2",
                "ground_truth_solution_type": "none",
                "has_solution": True,
                "ground_truth_has_solution": False,
            },
            {  # FP: falsely said no solution
                "equation_id": "eq_3",
                "solution_str": "x",
                "ground_truth": "x",
                "ground_truth_solution_type": "exact_symbolic",
                "has_solution": False,
                "ground_truth_has_solution": True,
            },
        ]
        path = self._write_jsonl(results)
        try:
            metrics = evaluate_solutions(path)
            nd = metrics["none_detection"]
            assert nd["tp"] == 1
            assert nd["fp"] == 1
            assert nd["fn"] == 1
            assert nd["precision"] == pytest.approx(0.5)
            assert nd["recall"] == pytest.approx(0.5)
            assert nd["f1"] == pytest.approx(0.5)
        finally:
            Path(path).unlink()

    def test_no_none_types_means_no_key(self) -> None:
        """When there are no none-type equations, none_detection is absent."""
        results = [
            {
                "equation_id": "eq_1",
                "solution_str": "x",
                "ground_truth": "x",
                "ground_truth_solution_type": "exact_symbolic",
                "has_solution": True,
                "ground_truth_has_solution": True,
            },
        ]
        path = self._write_jsonl(results)
        try:
            metrics = evaluate_solutions(path)
            assert "none_detection" not in metrics
        finally:
            Path(path).unlink()


class TestResidualVerificationWiring:
    """Tests that verify_solution is called when equation components are present."""

    def _write_jsonl(self, results: list[dict]) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
            return f.name

    def test_residual_computed_when_components_present(self) -> None:
        """verify_solution is called when kernel/f/lambda are in the result."""
        results = [
            {
                "equation_id": "eq_1",
                "solution_str": "3*x/2",
                "ground_truth": "3*x/2",
                "ground_truth_solution_type": "exact_symbolic",
                "ground_truth_kernel": "x*t",
                "ground_truth_f": "x",
                "ground_truth_lambda": 1.0,
                "ground_truth_domain": [0, 1],
            },
        ]
        path = self._write_jsonl(results)
        try:
            metrics = evaluate_solutions(path)
            assert "residual_verification" in metrics
            rv = metrics["residual_verification"]
            assert rv["total"] == 1
            assert rv["verified_rate"] > 0
        finally:
            Path(path).unlink()

    def test_no_residual_without_components(self) -> None:
        """No residual verification when kernel/f/lambda missing."""
        results = [
            {
                "equation_id": "eq_1",
                "solution_str": "x",
                "ground_truth": "x",
                "ground_truth_solution_type": "exact_symbolic",
            },
        ]
        path = self._write_jsonl(results)
        try:
            metrics = evaluate_solutions(path)
            assert "residual_verification" not in metrics
        finally:
            Path(path).unlink()
