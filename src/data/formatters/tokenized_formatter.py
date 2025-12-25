"""
Tokenized formatter.

Converts between space-separated token format and SymPy expressions.
"""

import sympy as sp

from src.data.formatters.base import BaseFormatter


class TokenizedFormatter(BaseFormatter):
    """Formatter for tokenized representation (space-separated tokens)."""

    @property
    def format_name(self) -> str:
        """Return format name."""
        return "tokenized"

    def to_sympy(self, expression: str) -> sp.Expr:
        """
        Convert tokenized string to SymPy expression.

        Args:
            expression: Tokenized string (e.g., "x ** 2 + 2 * x + 1").

        Returns:
            SymPy expression.

        Note:
            Currently just removes spaces and parses as infix.
            Can be extended for more sophisticated tokenization.
        """
        # Remove spaces and parse
        cleaned = expression.replace(" ", "")
        return sp.sympify(cleaned)

    def from_sympy(self, expr: sp.Expr, simplify: bool = False) -> str:
        """
        Convert SymPy expression to tokenized string.

        Args:
            expr: SymPy expression.
            simplify: Whether to canonicalize the expression first.

        Returns:
            Tokenized string with spaces around operators.
        """
        if simplify:
            expr = self.canonicalize(expr)
        expr_str = str(expr)
        # Add spaces around operators
        for op in ["+", "-", "*", "/", "^", "(", ")", "**"]:
            expr_str = expr_str.replace(op, f" {op} ")
        # Clean up multiple spaces
        return " ".join(expr_str.split())
