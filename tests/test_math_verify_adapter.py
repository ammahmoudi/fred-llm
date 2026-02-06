"""
Tests for the Math-Verify adapter module.
"""

from unittest.mock import patch

import pytest
import sympy as sp

from src.llm.math_verify_adapter import (
    FREDHOLM_LOCAL_DICT,
    HAS_MATH_VERIFY,
    extract_answer_from_response,
    extract_solution_from_response,
    math_verify_compare,
    parse_latex_to_sympy,
)


class TestFredholmLocalDict:
    """Tests for the consolidated symbol dictionary."""

    def test_contains_standard_variables(self) -> None:
        assert "x" in FREDHOLM_LOCAL_DICT
        assert "t" in FREDHOLM_LOCAL_DICT

    def test_contains_constants(self) -> None:
        assert FREDHOLM_LOCAL_DICT["e"] is sp.E
        assert FREDHOLM_LOCAL_DICT["pi"] is sp.pi
        assert FREDHOLM_LOCAL_DICT["oo"] is sp.oo

    def test_contains_free_constants(self) -> None:
        for name in ("C", "c_1", "c_2", "C_1", "C_2"):
            assert name in FREDHOLM_LOCAL_DICT

    def test_contains_integral(self) -> None:
        assert FREDHOLM_LOCAL_DICT["Integral"] is sp.Integral


class TestParseLatexToSympy:
    """Tests for parse_latex_to_sympy."""

    def test_simple_polynomial(self) -> None:
        expr = parse_latex_to_sympy("x**2 + 2*x + 1")
        x = FREDHOLM_LOCAL_DICT["x"]
        expected = x**2 + 2 * x + 1
        assert expr.equals(expected)

    def test_trig_expression(self) -> None:
        expr = parse_latex_to_sympy("\\sin(x)")
        x = FREDHOLM_LOCAL_DICT["x"]
        assert expr.equals(sp.sin(x))

    def test_exponential(self) -> None:
        expr = parse_latex_to_sympy("\\exp(x)")
        x = FREDHOLM_LOCAL_DICT["x"]
        assert expr.equals(sp.exp(x))

    def test_fraction_latex(self) -> None:
        expr = parse_latex_to_sympy("\\frac{3}{2}*x")
        x = FREDHOLM_LOCAL_DICT["x"]
        assert expr.equals(sp.Rational(3, 2) * x)

    def test_raises_on_garbage(self) -> None:
        from src.llm.postprocess import ParseError

        with pytest.raises(ParseError):
            parse_latex_to_sympy("@@@totally_not_math!!!")

    def test_fallback_without_math_verify(self) -> None:
        """Parsing still works when Math-Verify is disabled."""
        with patch("src.llm.math_verify_adapter.HAS_MATH_VERIFY", False):
            expr = parse_latex_to_sympy("x**2 + 1")
            x = FREDHOLM_LOCAL_DICT["x"]
            assert expr.equals(x**2 + 1)

    def test_latex_sqrt(self) -> None:
        expr = parse_latex_to_sympy("\\sqrt{x}")
        x = FREDHOLM_LOCAL_DICT["x"]
        assert expr.equals(sp.sqrt(x))

    def test_latex_e_power(self) -> None:
        expr = parse_latex_to_sympy("e^{2*x}")
        x = FREDHOLM_LOCAL_DICT["x"]
        assert expr.equals(sp.exp(2 * x))


class TestMathVerifyCompare:
    """Tests for the math_verify_compare fast-path."""

    def test_equivalent_expressions(self) -> None:
        x = sp.Symbol("x")
        result = math_verify_compare(
            (x + 1) ** 2,
            x**2 + 2 * x + 1,
        )
        if HAS_MATH_VERIFY:
            assert result is True
        else:
            assert result is None

    def test_different_expressions(self) -> None:
        x = sp.Symbol("x")
        result = math_verify_compare(x**2, x**3)
        if HAS_MATH_VERIFY:
            assert result is False
        else:
            assert result is None

    def test_returns_none_without_math_verify(self) -> None:
        x = sp.Symbol("x")
        with patch("src.llm.math_verify_adapter.HAS_MATH_VERIFY", False):
            result = math_verify_compare(x, x)
            assert result is None

    def test_trig_identity(self) -> None:
        x = sp.Symbol("x")
        result = math_verify_compare(
            sp.sin(x) ** 2 + sp.cos(x) ** 2,
            sp.Integer(1),
        )
        if HAS_MATH_VERIFY:
            assert result is True
        else:
            assert result is None


class TestExtractAnswerFromResponse:
    """Tests for extract_answer_from_response."""

    def test_extracts_equation_rhs(self) -> None:
        """Should extract RHS from u(x) = expr in LLM text."""
        response = (
            "After solving the Fredholm equation, we get:\n"
            "$$u(x) = x^2 + 1$$\n"
            "This satisfies the original equation."
        )
        result = extract_answer_from_response(response)
        if HAS_MATH_VERIFY:
            assert result is not None
            expr, raw = result
            x = FREDHOLM_LOCAL_DICT["x"]
            assert expr.equals(x**2 + 1)
        else:
            assert result is None

    def test_rejects_scrambled_text(self) -> None:
        """Should reject parses that are products of single letters."""
        response = "No solution exists for this equation."
        result = extract_answer_from_response(response)
        # Should be None (either no parse or scrambled text detection)
        if result is not None:
            _, raw = result
            # If something was extracted, it shouldn't be scrambled
            assert "n*o" not in str(result[0])

    def test_returns_none_without_math_verify(self) -> None:
        with patch("src.llm.math_verify_adapter.HAS_MATH_VERIFY", False):
            result = extract_answer_from_response("u(x) = x^2")
            assert result is None

    def test_rejects_trivial_number(self) -> None:
        """Should reject results that are just numbers (lost structure)."""
        response = "The answer is approximately 42."
        result = extract_answer_from_response(response)
        # Should be None since 42 is just a number
        assert result is None

    def test_parse_llm_output_uses_mv_fallback(self) -> None:
        """parse_llm_output should recover via MV when regex extraction fails."""
        from src.llm.postprocess import parse_llm_output

        # A response where regex extraction will grab garbage but MV can find the math
        response = (
            "SOLUTION: The solution to this Fredholm equation is\n"
            "$$u(x) = \\sin(x) + x^2$$\n"
            "which can be verified by substitution."
        )
        result = parse_llm_output(response)
        # Should have a sympy expression one way or another
        assert result["solution_sympy"] is not None


class TestExtractSolutionFromResponse:
    """Tests for the multi-strategy extract_solution_from_response."""

    def test_targeted_ux_strategy(self) -> None:
        """Should extract RHS from the last u(x) = ... line."""
        response = (
            "Step 1: We substitute into the equation.\n"
            "Step 2: After simplification, we get\n"
            "u(x) = x^2 + 3x + 1\n"
            "which satisfies the original equation."
        )
        if HAS_MATH_VERIFY:
            result = extract_solution_from_response(response)
            assert result is not None
            expr, raw = result
            x = FREDHOLM_LOCAL_DICT["x"]
            assert expr.equals(x**2 + 3 * x + 1)
        else:
            assert extract_solution_from_response(response) is None

    def test_structured_solution_marker(self) -> None:
        """Should extract from SOLUTION: marker line."""
        response = (
            "HAS_SOLUTION: yes\n"
            "SOLUTION_TYPE: exact_symbolic\n"
            "SOLUTION: u(x) = \\sin(x) + x\n"
            "REASONING: Direct substitution."
        )
        if HAS_MATH_VERIFY:
            result = extract_solution_from_response(response)
            assert result is not None
            expr, _ = result
            x = FREDHOLM_LOCAL_DICT["x"]
            assert expr.equals(sp.sin(x) + x)
        else:
            assert extract_solution_from_response(response) is None

    def test_full_response_eq_unwrap(self) -> None:
        """Should unwrap Eq(u(x), rhs) from full response parse."""
        response = (
            "After solving the Fredholm equation:\n"
            "$$u(x) = e^{x}$$\n"
        )
        if HAS_MATH_VERIFY:
            result = extract_solution_from_response(response)
            assert result is not None
            expr, _ = result
            x = FREDHOLM_LOCAL_DICT["x"]
            assert expr.equals(sp.exp(x))
        else:
            assert extract_solution_from_response(response) is None

    def test_returns_none_for_no_solution_text(self) -> None:
        """Should return None for 'no solution' text."""
        response = "No solution exists for this equation."
        result = extract_solution_from_response(response)
        assert result is None

    def test_returns_none_without_math_verify(self) -> None:
        with patch("src.llm.math_verify_adapter.HAS_MATH_VERIFY", False):
            result = extract_solution_from_response("u(x) = x^2")
            assert result is None

    def test_parse_llm_output_uses_mv_primary(self) -> None:
        """parse_llm_output should use MV as primary extraction path."""
        from src.llm.postprocess import parse_llm_output

        response = (
            "The solution is:\n"
            "$$u(x) = \\frac{x^2}{2} + 1$$\n"
        )
        result = parse_llm_output(response)
        if HAS_MATH_VERIFY:
            assert result["solution_sympy"] is not None
            x = FREDHOLM_LOCAL_DICT["x"]
            assert result["solution_sympy"].equals(x**2 / 2 + 1)
            assert result["confidence"] == 0.8


class TestIntegrationWithExistingCode:
    """Verify the adapter integrates correctly with postprocess and evaluate."""

    def test_postprocess_parse_to_sympy_uses_adapter(self) -> None:
        """_parse_to_sympy should delegate to the adapter."""
        from src.llm.postprocess import _parse_to_sympy

        expr = _parse_to_sympy("x**2 + 1")
        x = sp.Symbol("x")
        assert sp.simplify(expr - (x**2 + 1)) == 0

    def test_symbolic_compare_with_math_verify(self) -> None:
        """symbolic_compare should use the fast-path when available."""
        from src.llm.evaluate import symbolic_compare

        x = sp.Symbol("x")
        result = symbolic_compare((x + 1) ** 2, x**2 + 2 * x + 1)
        assert result["equivalent"] is True
