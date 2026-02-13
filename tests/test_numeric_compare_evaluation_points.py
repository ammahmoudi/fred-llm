"""
Unit tests for numeric_compare with evaluation_points support.

Tests that numeric_compare correctly uses pre-computed evaluation points
from ground_truth dictionaries for consistent numeric evaluation.
"""

import numpy as np
import pytest
import sympy as sp

from src.evaluation.metrics import numeric_compare


class TestNumericCompareWithEvaluationPoints:
    """Test suite for numeric_compare with evaluation_points."""

    def test_numeric_compare_with_evaluation_points(self):
        """Test numeric_compare uses evaluation_points when available."""
        # Solution: x^2
        solution = sp.sympify("x**2")

        # Ground truth with evaluation_points
        ground_truth = {
            "u": "x**2",
            "evaluation_points": {
                "x_values": [0.0, 0.25, 0.5, 0.75, 1.0],
                "u_values": [0.0, 0.0625, 0.25, 0.5625, 1.0],
                "n_points": 5,
            },
        }

        result = numeric_compare(solution, ground_truth, domain=(0, 1))

        # Should match perfectly
        assert result["match"] is True
        assert result["max_error"] < 1e-10
        assert result["rmse"] < 1e-10

    def test_numeric_compare_with_evaluation_points_mismatch(self):
        """Test numeric_compare detects errors when solution doesn't match."""
        # Solution: x^2
        solution = sp.sympify("x**2")

        # Ground truth: x^3 (different function)
        x_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        u_values = [x**3 for x in x_values]  # x^3 values

        ground_truth = {
            "u": "x**3",
            "evaluation_points": {
                "x_values": x_values,
                "u_values": u_values,
                "n_points": len(x_values),
            },
        }

        result = numeric_compare(solution, ground_truth, domain=(0, 1), tolerance=1e-6)

        # Should not match
        assert result["match"] is False
        assert result["max_error"] > 0.01  # Significant error at x=1

    def test_numeric_compare_fallback_to_u_field(self):
        """Test numeric_compare falls back to u field if no evaluation_points."""
        solution = sp.sympify("sin(x)")

        # Ground truth dict without evaluation_points
        ground_truth = {
            "u": "sin(x)",
            "a": "0",
            "b": "1",
        }

        result = numeric_compare(solution, ground_truth, domain=(0, 1), n_points=50)

        # Should still work using u field
        assert result["match"] is True
        assert result["max_error"] < 1e-6

    def test_numeric_compare_traditional_sympy_expr(self):
        """Test numeric_compare backward compatibility with SymPy expressions."""
        solution = sp.sympify("exp(x)")
        ground_truth = sp.sympify("exp(x)")

        result = numeric_compare(solution, ground_truth, domain=(0, 1), n_points=100)

        # Should work as before
        assert result["match"] is True
        assert result["max_error"] < 1e-6

    def test_numeric_compare_evaluation_points_different_domain(self):
        """Test evaluation_points work correctly for different domains."""
        solution = sp.sympify("x**2 + 2*x + 1")

        # Evaluation points for domain [1, 3]
        x_values = np.linspace(1, 3, 20)
        u_values = [x**2 + 2 * x + 1 for x in x_values]

        ground_truth = {
            "u": "x**2 + 2*x + 1",
            "evaluation_points": {
                "x_values": x_values.tolist(),
                "u_values": u_values,
                "n_points": len(x_values),
            },
        }

        result = numeric_compare(solution, ground_truth, domain=(1, 3))

        assert result["match"] is True
        assert result["max_error"] < 1e-9

    def test_numeric_compare_empty_ground_truth_dict(self):
        """Test numeric_compare handles empty ground_truth dict gracefully."""
        solution = sp.sympify("x")
        ground_truth = {}  # Empty dict

        result = numeric_compare(solution, ground_truth, domain=(0, 1))

        # Should return error
        assert "error" in result
        assert result["match"] is False

    def test_numeric_compare_evaluation_points_consistency(self):
        """Test that evaluation_points provide consistent results across runs."""
        solution = sp.sympify("sin(pi*x)")

        # Fixed evaluation points
        x_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        u_values = [np.sin(np.pi * x) for x in x_values]

        ground_truth = {
            "u": "sin(pi*x)",
            "evaluation_points": {
                "x_values": x_values,
                "u_values": u_values,
                "n_points": len(x_values),
            },
        }

        # Run multiple times
        results = [
            numeric_compare(solution, ground_truth, domain=(0, 1)) for _ in range(5)
        ]

        # All results should be identical
        for i in range(1, len(results)):
            assert results[i]["max_error"] == results[0]["max_error"]
            assert results[i]["mean_error"] == results[0]["mean_error"]
            assert results[i]["rmse"] == results[0]["rmse"]

    def test_numeric_compare_complex_expression_evaluation_points(self):
        """Test evaluation_points work with complex expressions."""
        solution = sp.sympify("exp(x) * cos(2*x) + sin(x)")

        # Generate evaluation points
        x_values = np.linspace(-1, 1, 30)
        u_values = [np.exp(x) * np.cos(2 * x) + np.sin(x) for x in x_values]

        ground_truth = {
            "u": "exp(x) * cos(2*x) + sin(x)",
            "evaluation_points": {
                "x_values": x_values.tolist(),
                "u_values": u_values,
                "n_points": len(x_values),
            },
        }

        result = numeric_compare(solution, ground_truth, domain=(-1, 1))

        assert result["match"] is True
        assert result["max_error"] < 1e-9

    def test_numeric_compare_evaluation_points_with_tolerance(self):
        """Test tolerance parameter works correctly with evaluation_points."""
        # Solution with small error
        solution = sp.sympify("x**2 + 0.0001")

        # Ground truth: x^2
        x_values = [0.0, 0.5, 1.0]
        u_values = [0.0, 0.25, 1.0]

        ground_truth = {
            "u": "x**2",
            "evaluation_points": {
                "x_values": x_values,
                "u_values": u_values,
                "n_points": len(x_values),
            },
        }

        # With strict tolerance - should not match
        result_strict = numeric_compare(
            solution, ground_truth, domain=(0, 1), tolerance=1e-6
        )
        assert result_strict["match"] is False

        # With relaxed tolerance - should match
        result_relaxed = numeric_compare(
            solution, ground_truth, domain=(0, 1), tolerance=1e-3
        )
        assert result_relaxed["match"] is True
