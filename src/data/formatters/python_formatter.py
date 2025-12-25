"""
Python code formatter.

Converts between Python code representation and SymPy expressions.
"""

import sympy as sp
from sympy.printing.pycode import pycode

from src.data.formatters.base import BaseFormatter


class PythonFormatter(BaseFormatter):
    """Formatter for Python code representation."""

    @property
    def format_name(self) -> str:
        """Return format name."""
        return "python"

    def to_sympy(self, expression: str) -> sp.Expr:
        """
        Convert Python code string to SymPy expression.

        Args:
            expression: Python code string.

        Returns:
            SymPy expression.
        """
        return sp.sympify(expression)

    def from_sympy(self, expr: sp.Expr, simplify: bool = False) -> str:
        """
        Convert SymPy expression to Python code string.

        Args:
            expr: SymPy expression.
            simplify: Whether to canonicalize the expression first.

        Returns:
            Python code string.
        """
        if simplify:
            expr = self.canonicalize(expr)
        # Use strict=False to allow partial printing for unsupported functions
        return pycode(expr, strict=False)
