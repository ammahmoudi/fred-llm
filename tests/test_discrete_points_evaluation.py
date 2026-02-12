"""Tests for discrete_points evaluation functionality."""

import pytest

from src.llm.evaluate import SolutionEvaluator, evaluate_discrete_points


class TestEvaluateDiscretePoints:
    """Test the evaluate_discrete_points function."""

    def test_exact_match(self):
        """Test evaluation when predicted points match ground truth exactly."""
        pred_points = [(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)]
        gt_points = [(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)]

        result = evaluate_discrete_points(pred_points, gt_points)

        assert result["match"] is True
        assert result["matched_points"] == 3
        assert result["accuracy"] == 1.0
        assert result["max_error"] == 0.0
        assert result["mean_error"] == 0.0

    def test_partial_match(self):
        """Test evaluation with some mismatched points."""
        pred_points = [(0.0, 1.0), (0.5, 2.5), (1.0, 3.0)]  # Middle point off
        gt_points = [(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)]

        result = evaluate_discrete_points(pred_points, gt_points, y_tolerance=1e-3)

        assert result["matched_points"] == 2
        assert result["accuracy"] == 2 / 3
        assert result["max_error"] == pytest.approx(0.5)

    def test_x_coordinate_tolerance(self):
        """Test that x-coordinates are matched within tolerance."""
        pred_points = [(0.0, 1.0), (0.501, 2.0), (1.0, 3.0)]  # x off by 0.001
        gt_points = [(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)]

        result = evaluate_discrete_points(pred_points, gt_points, x_tolerance=1e-2)

        assert result["matched_points"] == 3
        assert result["match"] is True

    def test_x_coordinate_tolerance_exceeded(self):
        """Test that x-coordinates outside tolerance are not matched."""
        pred_points = [(0.0, 1.0), (0.52, 2.0), (1.0, 3.0)]  # x off by 0.02
        gt_points = [(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)]

        result = evaluate_discrete_points(pred_points, gt_points, x_tolerance=1e-2)

        # Only 2 points should match (the ones at x=0.0 and x=1.0)
        assert result["matched_points"] == 2

    def test_negative_values(self):
        """Test evaluation with negative point values."""
        pred_points = [(-1.0, -2.5), (0.0, -1.0), (1.0, 0.5)]
        gt_points = [(-1.0, -2.5), (0.0, -1.0), (1.0, 0.5)]

        result = evaluate_discrete_points(pred_points, gt_points)

        assert result["match"] is True
        assert result["matched_points"] == 3

    def test_scientific_notation(self):
        """Test evaluation with scientific notation values."""
        pred_points = [(0.0, 1.23e-2), (0.5, 3.45e1), (1.0, 2.1e0)]
        gt_points = [(0.0, 0.0123), (0.5, 34.5), (1.0, 2.1)]

        result = evaluate_discrete_points(pred_points, gt_points)

        assert result["match"] is True
        assert result["matched_points"] == 3

    def test_empty_predicted_points(self):
        """Test handling of empty predicted points."""
        pred_points = []
        gt_points = [(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)]

        result = evaluate_discrete_points(pred_points, gt_points)

        assert result["match"] is False
        assert result["matched_points"] == 0
        assert result["accuracy"] == 0.0

    def test_empty_ground_truth_points(self):
        """Test handling of empty ground truth points."""
        pred_points = [(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)]
        gt_points = []

        result = evaluate_discrete_points(pred_points, gt_points)

        assert result["match"] is False
        assert result["matched_points"] == 0

    def test_80_percent_threshold(self):
        """Test that match requires at least 80% accuracy."""
        pred_points = [(0.0, 1.0), (0.5, 2.5), (1.0, 3.5)]  # All off by 0.5
        gt_points = [(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)]

        result = evaluate_discrete_points(pred_points, gt_points, y_tolerance=1e-3)

        # 0 out of 3 points match, less than 80%
        assert result["match"] is False


class TestSolutionEvaluatorDiscretePoints:
    """Test discrete_points evaluation in SolutionEvaluator."""

    def test_evaluate_discrete_points_type_success(self):
        """Test successful discrete_points evaluation."""
        evaluator = SolutionEvaluator()

        pred_points = [(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)]
        gt_points = [(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)]

        result = evaluator.evaluate_discrete_points_type(pred_points, gt_points)

        assert result["correct"] is True
        assert result["solution_type"] == "discrete_points"
        assert result["numeric"]["matched_points"] == 3
        assert result["numeric"]["accuracy"] == 1.0

    def test_evaluate_discrete_points_type_failure(self):
        """Test failed discrete_points evaluation."""
        evaluator = SolutionEvaluator()

        pred_points = [(0.0, 5.0), (0.5, 5.0), (1.0, 5.0)]  # All wrong
        gt_points = [(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)]

        result = evaluator.evaluate_discrete_points_type(pred_points, gt_points)

        assert result["correct"] is False
        assert result["numeric"]["matched_points"] == 0

    def test_evaluator_tracks_results(self):
        """Test that evaluator tracks multiple discrete_points evaluations."""
        evaluator = SolutionEvaluator()

        # First evaluation
        pred_points_1 = [(0.0, 1.0), (0.5, 2.0)]
        gt_points_1 = [(0.0, 1.0), (0.5, 2.0)]
        evaluator.evaluate_discrete_points_type(pred_points_1, gt_points_1)

        # Second evaluation
        pred_points_2 = [(0.0, 1.0), (0.5, 3.0)]
        gt_points_2 = [(0.0, 1.0), (0.5, 2.0)]
        evaluator.evaluate_discrete_points_type(pred_points_2, gt_points_2)

        # Check that both are tracked
        assert len(evaluator.results) == 2
        assert evaluator.results[0]["correct"] is True
        assert evaluator.results[1]["correct"] is False

    def test_evaluator_summary_with_discrete_points(self):
        """Test summary statistics include discrete_points."""
        evaluator = SolutionEvaluator()

        # Add some discrete_points evaluations
        pred_points = [(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)]
        gt_points = [(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)]
        evaluator.evaluate_discrete_points_type(pred_points, gt_points)

        summary = evaluator.summary()

        assert summary["total"] == 1
        assert summary["correct"] == 1
        assert summary["accuracy"] == 1.0
        assert "discrete_points" in summary["per_type"]
        assert summary["per_type"]["discrete_points"]["accuracy"] == 1.0
