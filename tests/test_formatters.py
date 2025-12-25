"""Tests for format conversion between different mathematical notations."""

from pathlib import Path

import pytest
import sympy as sp

from src.data.format_converter import FormatConverter
from src.data.fredholm_loader import FredholmDatasetLoader


@pytest.fixture
def sample_dataset_path() -> Path:
    """Return path to sample dataset."""
    return Path("data/raw/Fredholm_Dataset_Sample.csv")


@pytest.fixture
def fredholm_loader(sample_dataset_path: Path) -> FredholmDatasetLoader:
    """Create FredholmDatasetLoader instance."""
    return FredholmDatasetLoader(data_path=sample_dataset_path, auto_download=False)


@pytest.fixture
def format_converter() -> FormatConverter:
    """Create FormatConverter instance."""
    return FormatConverter()


class TestFormatConversion:
    """Test format conversion between Infix, LaTeX, and RPN."""

    def test_infix_to_latex(self, format_converter: FormatConverter) -> None:
        """Test converting infix expressions to LaTeX format."""
        test_cases = [
            ("x**2", "x^{2}"),
            ("sin(x)", r"\sin{\left(x \right)}"),
            ("x + 2*y", "x + 2 y"),
        ]

        for infix, expected_latex in test_cases:
            latex = format_converter.convert(
                infix, source_format="infix", target_format="latex"
            )
            assert expected_latex in latex or latex in expected_latex

    def test_infix_to_rpn(self, format_converter: FormatConverter) -> None:
        """Test converting infix expressions to RPN format."""
        test_cases = [
            ("x**2", ["x", "2", "^"]),
            ("x + 2", ["x", "2", "+"]),
            ("2*x", ["2", "x", "*"]),
        ]

        for infix, expected_tokens in test_cases:
            rpn = format_converter.convert(
                infix, source_format="infix", target_format="rpn"
            )
            rpn_tokens = rpn.split()
            for token in expected_tokens:
                assert token in rpn_tokens

    def test_rpn_roundtrip(self, format_converter: FormatConverter) -> None:
        """Test converting to RPN and back to infix."""
        test_expr = "x**2 + 2*x + 1"
        rpn = format_converter.convert(
            test_expr, source_format="infix", target_format="rpn"
        )
        back_to_infix = format_converter.convert(
            rpn, source_format="rpn", target_format="infix"
        )
        # Normalize both expressions
        assert sp.simplify(sp.sympify(test_expr) - sp.sympify(back_to_infix)) == 0

    def test_trig_functions_conversion(self, format_converter: FormatConverter) -> None:
        """Test conversion of trigonometric functions."""
        trig_funcs = ["sin(x)", "cos(x)", "tan(x)"]

        for func in trig_funcs:
            rpn = format_converter.convert(
                func, source_format="infix", target_format="rpn"
            )
            assert func[:3] in rpn  # sin, cos, tan should be in RPN

    def test_hyperbolic_functions_conversion(
        self, format_converter: FormatConverter
    ) -> None:
        """Test conversion of hyperbolic functions."""
        hyp_funcs = ["sinh(x)", "cosh(x)", "tanh(x)"]

        for func in hyp_funcs:
            rpn = format_converter.convert(
                func, source_format="infix", target_format="rpn"
            )
            assert func[:4] in rpn  # sinh, cosh, tanh should be in RPN

    @pytest.mark.skipif(
        not Path("data/raw/Fredholm_Dataset_Sample.csv").exists(),
        reason="Sample dataset not found",
    )
    def test_bulk_conversion(
        self, sample_dataset_path: Path, fredholm_loader: FredholmDatasetLoader
    ) -> None:
        """Test converting entire dataset to RPN format."""
        # Load small sample
        fredholm_loader.max_samples = 100
        equations = fredholm_loader.load()
        assert len(equations) <= 100

        converter = FormatConverter()
        converted_count = 0

        for eq in equations[:100]:  # Test first 100
            try:
                rpn_u = converter.convert(
                    eq.u, source_format="infix", target_format="rpn"
                )
                if rpn_u:
                    converted_count += 1
            except Exception:
                pass  # Skip problematic equations

        # Should convert at least 80%
        assert converted_count >= 80

    def test_supported_formats(self, format_converter: FormatConverter) -> None:
        """Test that all supported formats are available."""
        formats = format_converter.supported_formats
        assert "latex" in formats
        assert "rpn" in formats
        assert "infix" in formats
        assert "python" in formats
        assert "sympy" in formats

    def test_python_format(self, format_converter: FormatConverter) -> None:
        """Test Python code format conversion."""
        expr = "x**2 + 1"
        python_code = format_converter.convert(
            expr, source_format="infix", target_format="python"
        )
        assert "x**2" in python_code or "x ** 2" in python_code

    def test_fredholm_equation_formatter_infix(self) -> None:
        """Test FredholmEquationFormatter with infix format."""
        from src.data.formatters.fredholm_formatter import FredholmEquationFormatter
        
        formatter = FredholmEquationFormatter(expression_format="infix")
        
        # Test format_equation
        equation = formatter.format_equation(
            u="u(x)",
            f="x**2 + 2*x",
            kernel="x*t",
            lambda_val="0.5",
            a="0",
            b="1"
        )
        
        assert "u(x)" in equation
        assert "∫" in equation or "integral" in equation.lower()
        assert "x**2" in equation or "x^2" in equation
        assert "0.5" in equation

    def test_fredholm_equation_formatter_latex(self) -> None:
        """Test FredholmEquationFormatter with LaTeX format."""
        from src.data.formatters.fredholm_formatter import FredholmEquationFormatter
        
        formatter = FredholmEquationFormatter(expression_format="latex")
        
        equation = formatter.format_equation(
            u="u(x)",
            f="x**2",
            kernel="x*t",
            lambda_val="1",
            a="0",
            b="1"
        )
        
        assert "u(x)" in equation
        assert "\\int" in equation
        assert "x^{2}" in equation or "x**2" in equation

    def test_fredholm_equation_formatter_rpn(self) -> None:
        """Test FredholmEquationFormatter with RPN format."""
        from src.data.formatters.fredholm_formatter import FredholmEquationFormatter
        
        formatter = FredholmEquationFormatter(expression_format="rpn")
        
        equation = formatter.format_equation(
            u="u(x)",
            f="x + 1",
            kernel="x*t",
            lambda_val="1",
            a="0",
            b="1"
        )
        
        # RPN format should be present
        assert isinstance(equation, str)
        assert len(equation) > 0

    def test_tokenized_equation_formatter(self) -> None:
        """Test TokenizedEquationFormatter with special tokens."""
        from src.data.formatters.fredholm_formatter import TokenizedEquationFormatter
        
        formatter = TokenizedEquationFormatter()
        
        equation = formatter.format_equation(
            u="u(x)",
            f="x**2",
            kernel="x*t",
            lambda_val="0.5",
            a="0",
            b="1"
        )
        
        # Check for special tokens
        assert "<LAMBDA>" in equation
        assert "<INT>" in equation
        assert "<LOWER>" in equation
        assert "<UPPER>" in equation
        assert "<SEP>" in equation

    def test_simplify_parameter(self) -> None:
        """Test that simplify parameter canonicalizes expressions."""
        from src.data.formatters.infix_formatter import InfixFormatter
        
        formatter = InfixFormatter()
        
        # Create a non-simplified expression
        expr = sp.sympify("x + x + 1")
        
        # Without simplify
        result1 = formatter.from_sympy(expr, simplify=False)
        
        # With simplify
        result2 = formatter.from_sympy(expr, simplify=True)
        
        # Simplified version should be cleaner
        assert "2*x" in result2 or "x + x" in result1

    def test_series_formatter_taylor(self) -> None:
        """Test SeriesFormatter with Taylor series."""
        from src.data.formatters.series_formatter import SeriesFormatter
        
        formatter = SeriesFormatter(order=5, x_var="x", x0=0)
        
        # Test with sin(x)
        expr = sp.sin(sp.Symbol("x"))
        series = formatter.from_sympy(expr)
        
        # Should contain series terms
        assert "x" in series
        assert "O(x" in series or "..." in series
        
    def test_series_formatter_polynomial(self) -> None:
        """Test SeriesFormatter polynomial approximation."""
        from src.data.formatters.series_formatter import SeriesFormatter
        
        formatter = SeriesFormatter(order=3)
        
        # Test polynomial approximation
        poly = formatter.format_polynomial_approximation("exp(x)", degree=3)
        
        # Should be polynomial without O() term
        assert "O(" not in poly
        assert "x" in poly

    def test_series_formatter_roundtrip(self) -> None:
        """Test SeriesFormatter roundtrip conversion."""
        from src.data.formatters.series_formatter import SeriesFormatter
        
        formatter = SeriesFormatter(order=5)
        
        # Convert to series and back
        expr = sp.sympify("x**2 + 2*x + 1")
        series_str = formatter.from_sympy(expr)
        back_to_expr = formatter.to_sympy(series_str)
        
        # Should be equivalent (series expansion of polynomial is itself)
        assert sp.simplify(expr - back_to_expr) == 0

    def test_neumann_series_formatter(self) -> None:
        """Test NeumannSeriesFormatter for Fredholm equations."""
        from src.data.formatters.series_formatter import NeumannSeriesFormatter
        
        formatter = NeumannSeriesFormatter(n_terms=3)
        
        # Format Neumann series
        series = formatter.format_neumann_series(
            f="x**2",
            kernel="x*t",
            lambda_val=0.5,
            bounds=(0, 1)
        )
        
        # Should contain series terms
        assert "u(x)" in series
        assert "λ" in series or "lambda" in series.lower()
        assert "K" in series
        
    def test_neumann_series_truncated(self) -> None:
        """Test NeumannSeriesFormatter truncated solution."""
        from src.data.formatters.series_formatter import NeumannSeriesFormatter
        
        formatter = NeumannSeriesFormatter(n_terms=4)
        
        # Format truncated series
        truncated = formatter.format_truncated_solution(
            f="x",
            kernel="x*t",
            lambda_val=1.0,
            n_terms=3
        )
        
        # Should contain approximation
        assert "u(x)" in truncated
        assert "≈" in truncated
        assert "λ" in truncated or "lambda" in truncated.lower()
