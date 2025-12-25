"""
Example demonstrating all formatters in the Fred-LLM project.

This script shows how to use each formatter to convert mathematical
expressions between different formats, including CSV export.
"""

from pathlib import Path

import sympy as sp

from src.data.format_converter import FormatConverter
from src.data.formatters import (
    FredholmEquationFormatter,
    InfixFormatter,
    LaTeXFormatter,
    NeumannSeriesFormatter,
    PythonFormatter,
    RPNFormatter,
    SeriesFormatter,
    TokenizedEquationFormatter,
    TokenizedFormatter,
)


def example_basic_formatters():
    """Demonstrate basic expression formatters."""
    print("=" * 60)
    print("BASIC EXPRESSION FORMATTERS")
    print("=" * 60)

    expr = "x**2 + 2*x + 1"
    print(f"\nOriginal expression: {expr}\n")

    # Infix
    infix = InfixFormatter()
    print(f"Infix:      {expr}")

    # LaTeX
    latex = LaTeXFormatter()
    sympy_expr = sp.sympify(expr)
    print(f"LaTeX:      {latex.from_sympy(sympy_expr)}")

    # RPN
    rpn = RPNFormatter()
    print(f"RPN:        {rpn.from_sympy(sympy_expr)}")

    # Python
    python = PythonFormatter()
    print(f"Python:     {python.from_sympy(sympy_expr)}")

    # Tokenized
    tokenized = TokenizedFormatter()
    print(f"Tokenized:  {tokenized.from_sympy(sympy_expr)}")


def example_fredholm_formatters():
    """Demonstrate Fredholm equation formatters."""
    print("\n" + "=" * 60)
    print("FREDHOLM EQUATION FORMATTERS")
    print("=" * 60)

    print("\n1. Standard Fredholm Formatter (Infix):")
    formatter_infix = FredholmEquationFormatter(expression_format="infix")
    eq_infix = formatter_infix.format_equation(
        u="u(x)", f="x**2 + 2*x", kernel="x*t", lambda_val="0.5", a="0", b="1"
    )
    print(f"   {eq_infix}")

    print("\n2. Fredholm Formatter (LaTeX):")
    formatter_latex = FredholmEquationFormatter(expression_format="latex")
    eq_latex = formatter_latex.format_equation(
        u="u(x)", f="x**2", kernel="x*t", lambda_val="1", a="0", b="1"
    )
    print(f"   {eq_latex}")

    print("\n3. Tokenized Equation Formatter (with special tokens):")
    tokenized = TokenizedEquationFormatter()
    eq_tokenized = tokenized.format_equation(
        u="u(x)", f="x**2", kernel="x*t", lambda_val="0.5", a="0", b="1"
    )
    print(f"   {eq_tokenized}")


def example_series_formatters():
    """Demonstrate series expansion formatters."""
    print("\n" + "=" * 60)
    print("SERIES EXPANSION FORMATTERS")
    print("=" * 60)

    # Taylor series
    print("\n1. Taylor Series Formatter:")
    series_fmt = SeriesFormatter(order=5, x_var="x", x0=0)

    expr = sp.sin(sp.Symbol("x"))
    print(f"   sin(x) Taylor series (5 terms):")
    print(f"   {series_fmt.from_sympy(expr)}")

    expr = sp.exp(sp.Symbol("x"))
    poly = series_fmt.format_polynomial_approximation("exp(x)", degree=3)
    print(f"\n   exp(x) polynomial approximation (degree 3):")
    print(f"   {poly}")

    # Neumann series
    print("\n2. Neumann Series Formatter:")
    neumann_fmt = NeumannSeriesFormatter(n_terms=3, include_symbolic=True)

    series = neumann_fmt.format_neumann_series(
        f="x**2", kernel="x*t", lambda_val=0.5, bounds=(0, 1)
    )
    print(f"   Complete Neumann series:")
    print(f"   {series}")

    truncated = neumann_fmt.format_truncated_solution(
        f="x", kernel="x*t", lambda_val=1.0, n_terms=3
    )
    print(f"\n   Truncated Neumann series:")
    print(f"   {truncated}")


def example_canonicalization():
    """Demonstrate expression canonicalization."""
    print("\n" + "=" * 60)
    print("EXPRESSION CANONICALIZATION")
    print("=" * 60)

    formatter = InfixFormatter()
    expr = sp.sympify("x + x + 1 + x")

    print("\nOriginal expression: x + x + 1 + x")
    print(f"Without simplify: {formatter.from_sympy(expr, simplify=False)}")
    print(f"With simplify:    {formatter.from_sympy(expr, simplify=True)}")


def example_roundtrip():
    """Demonstrate roundtrip conversion."""
    print("\n" + "=" * 60)
    print("ROUNDTRIP CONVERSION")
    print("=" * 60)

    original = "x**2 + 2*x + 1"
    print(f"\nOriginal: {original}")

    # Infix -> RPN -> Infix (more reliable than LaTeX parsing)
    infix_fmt = InfixFormatter()
    rpn_fmt = RPNFormatter()

    sympy_expr = infix_fmt.to_sympy(original)
    rpn_str = rpn_fmt.from_sympy(sympy_expr)
    print(f"RPN:      {rpn_str}")

    back_to_sympy = rpn_fmt.to_sympy(rpn_str)
    back_to_infix = infix_fmt.from_sympy(back_to_sympy)
    print(f"Back:     {back_to_infix}")

    # Verify equivalence
    is_equal = sp.simplify(sympy_expr - back_to_sympy) == 0
    print(f"Equal:    {is_equal}")


def example_csv_export():
    """Demonstrate CSV export functionality."""
    print("\n" + "=" * 60)
    print("CSV EXPORT")
    print("=" * 60)

    # Sample equations
    equations = [
        {
            "u": "x**2",
            "f": "x**2 - x**4/3",
            "kernel": "x*t",
            "lambda": "0.5",
            "a": "0",
            "b": "1",
        },
        {
            "u": "sin(x)",
            "f": "sin(x) - 0.3*cos(x)",
            "kernel": "cos(x-t)",
            "lambda": "0.3",
            "a": "0",
            "b": "3.14159",
        },
    ]

    converter = FormatConverter()
    output_path = Path("data/examples/formatted_equations.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export in different formats
    for fmt in ["infix", "latex", "rpn", "tokenized"]:
        output_file = Path(f"data/examples/equations_{fmt}.csv")
        converter.convert_to_csv(equations, output_file, format=fmt)
        print(
            f"✓ Exported {len(equations)} equations to {fmt.upper()} CSV: {output_file}"
        )

    print("\nCSV files preserve the original dataset structure")
    print("Columns: u, f, kernel, lambda, a, b")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("FRED-LLM FORMATTERS EXAMPLE")
    print("=" * 60)

    example_basic_formatters()
    example_fredholm_formatters()
    example_series_formatters()
    example_canonicalization()
    example_roundtrip()
    example_csv_export()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
