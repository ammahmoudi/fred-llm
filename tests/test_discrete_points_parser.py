"""Tests for discrete_points parser in postprocess.py."""

import pytest

from src.llm.postprocess import extract_discrete_points, parse_llm_output


class TestExtractDiscretePoints:
    """Test the extract_discrete_points function."""

    def test_standard_format(self):
        """Test extraction from standard format."""
        response = "SOLUTION: [(0.0, 1.234), (0.25, 2.456), (0.5, 3.789), (0.75, 1.123), (1.0, 0.567)]"
        result = extract_discrete_points(response)

        assert result is not None
        assert len(result) == 5
        assert result[0] == (0.0, 1.234)
        assert result[-1] == (1.0, 0.567)

    def test_with_whitespace(self):
        """Test extraction with extra whitespace."""
        response = "SOLUTION: [ ( 0.0 , 1.2 ) , ( 0.5 , 3.4 ) , ( 1.0 , 2.1 ) ]"
        result = extract_discrete_points(response)

        assert result is not None
        assert len(result) == 3
        assert result[0] == (0.0, 1.2)

    def test_scientific_notation(self):
        """Test extraction with scientific notation."""
        response = "SOLUTION: [(0.0, 1.23e-2), (0.5, 3.45e1), (1.0, 2.1e0)]"
        result = extract_discrete_points(response)

        assert result is not None
        assert len(result) == 3
        assert abs(result[0][1] - 0.0123) < 1e-10

    def test_negative_values(self):
        """Test extraction with negative values."""
        response = "SOLUTION: [(-1.0, -2.5), (0.0, 0.0), (1.0, 2.5)]"
        result = extract_discrete_points(response)

        assert result is not None
        assert len(result) == 3
        assert result[0] == (-1.0, -2.5)

    def test_in_context(self):
        """Test extraction from response with surrounding text."""
        response = """The equation has discrete point solutions only.
SOLUTION: [(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)]
HAS_SOLUTION: yes
SOLUTION_TYPE: discrete_points"""

        result = extract_discrete_points(response)

        assert result is not None
        assert len(result) == 3
        assert result[0] == (0.0, 1.0)

    def test_too_few_points(self):
        """Test that single point is rejected."""
        response = "SOLUTION: [(0.0, 1.0)]"
        result = extract_discrete_points(response)

        assert result is None

    def test_no_points(self):
        """Test that missing points returns None."""
        response = "SOLUTION: No discrete points available"
        result = extract_discrete_points(response)

        assert result is None

    def test_invalid_format(self):
        """Test that invalid format returns None."""
        response = "SOLUTION: [0.0, 1.0, 0.5, 2.0]"  # Missing tuples
        result = extract_discrete_points(response)

        assert result is None


class TestParseDiscretePointsIntegration:
    """Test integration of discrete_points extraction with parse_llm_output."""

    def test_full_discrete_points_response(self):
        """Test full parsing of discrete_points response."""
        response = """I'll solve this Fredholm equation numerically.

SOLUTION: [(0.0, 1.234), (0.25, 2.456), (0.5, 3.789), (0.75, 1.123), (1.0, 0.567)]
HAS_SOLUTION: yes
SOLUTION_TYPE: discrete_points"""

        result = parse_llm_output(response, extract_solution=True, validate=True)

        assert result["has_solution"] is True
        assert result["solution_type"] == "discrete_points"
        assert result["discrete_points"] is not None
        assert len(result["discrete_points"]) == 5
        assert result["confidence"] == 0.8
        assert result["solution_str"] is not None

    def test_discrete_points_without_type_marker(self):
        """Test that points without SOLUTION_TYPE marker are not extracted specially."""
        response = """SOLUTION: [(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)]
HAS_SOLUTION: yes"""

        result = parse_llm_output(response, extract_solution=True, validate=True)

        # Should not extract as discrete_points without SOLUTION_TYPE marker
        assert result["solution_type"] is None
        assert result["discrete_points"] is None

    def test_discrete_points_type_but_missing_points(self):
        """Test handling of discrete_points type without actual points."""
        response = """SOLUTION: No points could be computed
HAS_SOLUTION: no
SOLUTION_TYPE: discrete_points"""

        result = parse_llm_output(response, extract_solution=True, validate=True)

        assert result["solution_type"] == "discrete_points"
        assert result["discrete_points"] is None
        assert result["confidence"] == 0.3
