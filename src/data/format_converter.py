"""
Format converter for mathematical expressions.

Unified interface for converting between different mathematical formats.
Uses individual formatter classes for each format.
"""

from pathlib import Path
from typing import Any

import sympy as sp

from src.data.formatters import (
    InfixFormatter,
    LaTeXFormatter,
    PythonFormatter,
    RPNFormatter,
    TokenizedFormatter,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FormatConverter:
    """
    Converter between mathematical expression formats.

    Uses formatter classes to convert between formats via SymPy as intermediate.
    """

    def __init__(self) -> None:
        """Initialize formatters."""
        self._formatters = {
            "latex": LaTeXFormatter(),
            "rpn": RPNFormatter(),
            "infix": InfixFormatter(),
            "python": PythonFormatter(),
            "tokenized": TokenizedFormatter(),
        }

    @property
    def supported_formats(self) -> list[str]:
        """Return list of supported format names."""
        return list(self._formatters.keys()) + ["sympy"]

    def convert(
        self,
        expression: str | sp.Expr,
        source_format: str,
        target_format: str,
    ) -> str | sp.Expr:
        """
        Convert expression between formats.

        Args:
            expression: Input expression (string or SymPy).
            source_format: Source format name.
            target_format: Target format name.

        Returns:
            Converted expression.

        Raises:
            ValueError: If format is not supported.
        """
        if source_format == target_format:
            return expression

        # Convert to SymPy (canonical form)
        sympy_expr = self._to_sympy(expression, source_format)

        # Convert to target format
        return self._from_sympy(sympy_expr, target_format)

    def _to_sympy(self, expression: str | sp.Expr, format: str) -> sp.Expr:
        """Convert any format to SymPy expression."""
        if isinstance(expression, sp.Expr):
            return expression

        if format == "sympy":
            return sp.sympify(expression)

        if format not in self._formatters:
            raise ValueError(
                f"Unknown format: {format}. Supported: {self.supported_formats}"
            )

        return self._formatters[format].to_sympy(expression)

    def _from_sympy(self, expr: sp.Expr, format: str) -> str | sp.Expr:
        """Convert SymPy expression to target format."""
        if format == "sympy":
            return expr

        if format not in self._formatters:
            raise ValueError(
                f"Unknown format: {format}. Supported: {self.supported_formats}"
            )

        return self._formatters[format].from_sympy(expr)

    def convert_to_csv(
        self,
        equations: list[dict[str, Any]],
        output_path: Path | str,
        format: str = "infix",
    ) -> None:
        """
        Export equations to CSV format with formatted expressions.

        Args:
            equations: List of equation dictionaries.
            output_path: Output CSV file path.
            format: Format for expressions (infix, latex, rpn, tokenized).
        """
        import pandas as pd

        formatted_data = []
        for eq in equations:
            formatted_eq = eq.copy()

            # Convert expressions if format specified
            if format != "infix" and format in self._formatters:
                for field in ["u", "f", "kernel"]:
                    if field in eq:
                        try:
                            sympy_expr = self._to_sympy(eq[field], "infix")
                            formatted_eq[field] = self._from_sympy(sympy_expr, format)
                        except Exception as e:
                            logger.warning(f"Failed to convert {field}: {e}")

            formatted_data.append(formatted_eq)

        df = pd.DataFrame(formatted_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(formatted_data)} equations to CSV: {output_path}")


def convert_format(
    input_file: Path | str,
    target_format: str = "rpn",
    source_format: str | None = None,
) -> list[dict[str, Any]]:
    """
    Convert a file of equations to a target format.

    Args:
        input_file: Path to input file.
        target_format: Target format for expressions.
        source_format: Source format (auto-detected if None).

    Returns:
        List of converted equations.
    """
    from src.data.fredholm_loader import FredholmDatasetLoader

    logger.info(f"Converting {input_file} to {target_format} format")

    # Load the dataset
    loader = FredholmDatasetLoader(data_path=input_file)
    equations = loader.load()

    # Convert each equation
    converter = FormatConverter()
    converted = []

    for eq in equations:
        try:
            # Convert u, f, and kernel
            converted_eq = {
                "original_u": eq.u,
                "original_f": eq.f,
                "original_kernel": eq.kernel,
                "lambda": eq.lambda_val,
                "a": eq.a,
                "b": eq.b,
            }

            # Parse and convert expressions
            if source_format is None:
                source_format = "infix"  # Assume infix by default

            u_sympy = converter._to_sympy(eq.u, source_format)
            f_sympy = converter._to_sympy(eq.f, source_format)
            kernel_sympy = converter._to_sympy(eq.kernel, source_format)

            converted_eq[f"u_{target_format}"] = converter._from_sympy(
                u_sympy, target_format
            )
            converted_eq[f"f_{target_format}"] = converter._from_sympy(
                f_sympy, target_format
            )
            converted_eq[f"kernel_{target_format}"] = converter._from_sympy(
                kernel_sympy, target_format
            )

            converted.append(converted_eq)

        except Exception as e:
            logger.warning(f"Failed to convert equation: {e}")
            continue

    logger.info(
        f"Converted {len(converted)}/{len(equations)} equations to {target_format}"
    )
    return converted


# Convenience functions
def latex_to_sympy(latex_str: str) -> sp.Expr:
    """Convert LaTeX to SymPy."""
    return LaTeXFormatter().to_sympy(latex_str)


def sympy_to_latex(expr: sp.Expr) -> str:
    """Convert SymPy to LaTeX."""
    return LaTeXFormatter().from_sympy(expr)


def sympy_to_rpn(expr: sp.Expr) -> str:
    """Convert SymPy to RPN."""
    return RPNFormatter().from_sympy(expr)


def rpn_to_sympy(rpn_str: str) -> sp.Expr:
    """Convert RPN to SymPy."""
    return RPNFormatter().to_sympy(rpn_str)
