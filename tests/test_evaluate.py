"""
Tests for evaluation module.
"""

import json
import tempfile
from pathlib import Path

import pytest
import sympy as sp

from src.llm.evaluate import (
    SolutionEvaluator,
    evaluate_solutions,
    numeric_compare,
    symbolic_compare,
    verify_solution,
)


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

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
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

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
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

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
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
