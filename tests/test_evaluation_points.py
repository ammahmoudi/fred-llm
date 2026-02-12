"""
Unit tests for evaluation points generation in augmentation strategies.

Tests that the _generate_evaluation_points method in BaseAugmentation
correctly generates fixed evaluation points for consistent numeric evaluation.
"""

import numpy as np
import pytest
import sympy as sp

from src.data.augmentations.base import BaseAugmentation


class MockAugmentation(BaseAugmentation):
    """Mock augmentation for testing base class functionality."""

    def augment(self, item):
        return []

    @property
    def strategy_name(self):
        return "mock"


class TestEvaluationPointsGeneration:
    """Test suite for evaluation points generation."""

    @pytest.fixture
    def mock_augmentation(self):
        """Create a mock augmentation instance."""
        return MockAugmentation()

    def test_generate_evaluation_points_simple_polynomial(self, mock_augmentation):
        """Test evaluation points generation for simple polynomial."""
        u_expr = "x**2"
        a, b = 0.0, 1.0
        n_points = 50

        result = mock_augmentation._generate_evaluation_points(u_expr, a, b, n_points)

        # Check structure
        assert "x_values" in result
        assert "u_values" in result
        assert "n_points" in result

        # Check sizes
        assert len(result["x_values"]) == result["n_points"]
        assert len(result["u_values"]) == result["n_points"]
        assert result["n_points"] >= n_points  # May have extra critical points

        # Check values are correct
        x_values = np.array(result["x_values"])
        u_values = np.array(result["u_values"])
        expected_u = x_values**2
        np.testing.assert_allclose(u_values, expected_u, rtol=1e-10)

        # Check domain bounds
        assert np.min(x_values) == a
        assert np.max(x_values) == b

    def test_generate_evaluation_points_trig_function(self, mock_augmentation):
        """Test evaluation points generation for trigonometric function."""
        u_expr = "sin(x)"
        a, b = 0.0, np.pi
        n_points = 30

        result = mock_augmentation._generate_evaluation_points(u_expr, a, b, n_points)

        # Verify sine values
        x_values = np.array(result["x_values"])
        u_values = np.array(result["u_values"])
        expected_u = np.sin(x_values)
        np.testing.assert_allclose(u_values, expected_u, rtol=1e-10)

    def test_generate_evaluation_points_complex_expression(self, mock_augmentation):
        """Test evaluation points generation for complex expression."""
        u_expr = "exp(x) * cos(2*x) + x**3"
        a, b = -1.0, 1.0
        n_points = 50

        result = mock_augmentation._generate_evaluation_points(u_expr, a, b, n_points)

        # Verify values
        x_values = np.array(result["x_values"])
        u_values = np.array(result["u_values"])
        expected_u = np.exp(x_values) * np.cos(2 * x_values) + x_values**3
        np.testing.assert_allclose(u_values, expected_u, rtol=1e-10)

    def test_generate_evaluation_points_includes_boundaries(self, mock_augmentation):
        """Test that evaluation points include domain boundaries."""
        u_expr = "x"
        a, b = 0.5, 2.5
        n_points = 20

        result = mock_augmentation._generate_evaluation_points(u_expr, a, b, n_points)

        x_values = result["x_values"]
        # Check boundaries are included
        assert a in x_values
        assert b in x_values

        # Check midpoint is included (from critical points)
        midpoint = (a + b) / 2
        assert midpoint in x_values

    def test_generate_evaluation_points_sorted(self, mock_augmentation):
        """Test that x_values are sorted in ascending order."""
        u_expr = "x**2 + 1"
        a, b = -2.0, 2.0
        n_points = 40

        result = mock_augmentation._generate_evaluation_points(u_expr, a, b, n_points)

        x_values = result["x_values"]
        # Check sorted
        assert x_values == sorted(x_values)

    def test_generate_evaluation_points_no_duplicates(self, mock_augmentation):
        """Test that x_values contain no duplicates."""
        u_expr = "sin(x)"
        a, b = 0.0, 1.0
        n_points = 50

        result = mock_augmentation._generate_evaluation_points(u_expr, a, b, n_points)

        x_values = result["x_values"]
        # Check no duplicates
        assert len(x_values) == len(set(x_values))

    def test_generate_evaluation_points_invalid_expression(self, mock_augmentation):
        """Test that invalid expression raises exception."""
        u_expr = "invalid_function_xyz(x)"
        a, b = 0.0, 1.0
        n_points = 50

        with pytest.raises(Exception):
            mock_augmentation._generate_evaluation_points(u_expr, a, b, n_points)

    def test_generate_evaluation_points_empty_expression(self, mock_augmentation):
        """Test that empty expression raises exception."""
        u_expr = ""
        a, b = 0.0, 1.0
        n_points = 50

        with pytest.raises(Exception):
            mock_augmentation._generate_evaluation_points(u_expr, a, b, n_points)

    def test_generate_evaluation_points_different_domains(self, mock_augmentation):
        """Test evaluation points generation for different domains."""
        u_expr = "x**2"

        # Test various domains
        domains = [(0, 1), (-1, 1), (5, 10), (-5, -1)]

        for a, b in domains:
            result = mock_augmentation._generate_evaluation_points(u_expr, a, b, 30)
            x_values = result["x_values"]

            # Check domain respected
            assert np.min(x_values) == a
            assert np.max(x_values) == b
            assert all(a <= x <= b for x in x_values)

    def test_generate_evaluation_points_consistent_results(self, mock_augmentation):
        """Test that multiple calls produce identical results."""
        u_expr = "exp(x) * sin(x)"
        a, b = 0.0, np.pi
        n_points = 50

        result1 = mock_augmentation._generate_evaluation_points(u_expr, a, b, n_points)
        result2 = mock_augmentation._generate_evaluation_points(u_expr, a, b, n_points)

        # Should be identical (deterministic)
        np.testing.assert_array_equal(result1["x_values"], result2["x_values"])
        np.testing.assert_array_equal(result1["u_values"], result2["u_values"])
        assert result1["n_points"] == result2["n_points"]

    def test_generate_evaluation_points_filters_overflow(self, mock_augmentation):
        """Test overflow values are filtered out for extreme expressions."""
        u_expr = "exp(100*x)"
        a, b = 0.0, 1.0

        result = mock_augmentation._generate_evaluation_points(u_expr, a, b, 50)

        # All returned values should be finite
        u_values = np.array(result["u_values"])
        assert np.all(np.isfinite(u_values))
        assert result["n_points"] > 0

    def test_generate_evaluation_points_substitutes_free_constants(
        self, mock_augmentation
    ):
        """Test that free constants are substituted for numeric evaluation."""
        u_expr = "C*x + 2"
        a, b = 0.0, 1.0

        result = mock_augmentation._generate_evaluation_points(u_expr, a, b, 10)

        x_values = np.array(result["x_values"])
        u_values = np.array(result["u_values"])
        expected_u = x_values + 2  # C is substituted with 1
        np.testing.assert_allclose(u_values, expected_u, rtol=1e-10)

        assert result["constant_samples"] == [-1.0, 1.0, 2.0]
        assert len(result["u_values_samples"]) == 3
