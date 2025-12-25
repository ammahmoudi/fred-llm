"""
Infix formatter.

Converts between standard infix notation and SymPy expressions.
Infix is the standard mathematical notation (e.g., "x + 2").
"""

import sympy as sp

from src.data.formatters.base import BaseFormatter


class InfixFormatter(BaseFormatter):
    """Formatter for standard infix mathematical notation."""

    @property
    def format_name(self) -> str:
        """Return format name."""
        return "infix"

    def to_sympy(self, expression: str) -> sp.Expr:
        """
        Convert infix string to SymPy expression.

        Args:
            expression: Infix string (e.g., "x**2 + 2*x + 1").

        Returns:
            SymPy expression.
        """
        return sp.sympify(expression)

    def from_sympy(self, expr: sp.Expr, simplify: bool = False) -> str:
        """
        Convert SymPy expression to infix string.

        Args:
            expr: SymPy expression.
            simplify: Whether to canonicalize the expression first.

        Returns:
            Infix string.
        """
        if simplify:
            expr = self.canonicalize(expr)
        return str(expr)
