"""
LaTeX formatter.

Converts between LaTeX mathematical notation and SymPy expressions.
"""

import sympy as sp
from sympy.parsing.latex import parse_latex

from src.data.formatters.base import BaseFormatter
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LaTeXFormatter(BaseFormatter):
    """Formatter for LaTeX mathematical notation."""

    @property
    def format_name(self) -> str:
        """Return format name."""
        return "latex"

    def to_sympy(self, expression: str) -> sp.Expr:
        """
        Convert LaTeX string to SymPy expression.

        Args:
            expression: LaTeX string (e.g., "x^2 + 2x + 1").

        Returns:
            SymPy expression.
        """
        try:
            return parse_latex(expression)
        except Exception as e:
            logger.warning(f"LaTeX parsing failed: {e}, trying fallback")
            # Fallback: simple cleaning
            cleaned = (
                expression.replace("\\", "")
                .replace("{", "(")
                .replace("}", ")")
                .replace("\\frac", "")
            )
            return sp.sympify(cleaned)

    def from_sympy(self, expr: sp.Expr, simplify: bool = False) -> str:
        """
        Convert SymPy expression to LaTeX string.

        Args:
            expr: SymPy expression.
            simplify: Whether to canonicalize the expression first.

        Returns:
            LaTeX string.
        """
        if simplify:
            expr = self.canonicalize(expr)
        return sp.latex(expr)
