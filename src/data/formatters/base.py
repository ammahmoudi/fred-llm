"""
Base formatter interface.

All formatters inherit from this class and implement to_sympy and from_sympy.
"""

from abc import ABC, abstractmethod

import sympy as sp


class BaseFormatter(ABC):
    """Base class for all formatters."""

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the name of this format."""
        pass

    @abstractmethod
    def to_sympy(self, expression: str) -> sp.Expr:
        """
        Convert expression from this format to SymPy.

        Args:
            expression: String in this formatter's format.

        Returns:
            SymPy expression.
        """
        pass

    @abstractmethod
    def from_sympy(self, expr: sp.Expr, simplify: bool = False) -> str:
        """
        Convert SymPy expression to this format.

        Args:
            expr: SymPy expression.
            simplify: Whether to simplify/canonicalize the expression first.

        Returns:
            String in this formatter's format.
        """
        pass

    def canonicalize(self, expr: sp.Expr) -> sp.Expr:
        """
        Canonicalize expression for consistent formatting.
        
        Args:
            expr: SymPy expression to canonicalize.
        
        Returns:
            Simplified and sorted expression.
        """
        # Simplify and expand
        expr = sp.simplify(expr)
        # Sort terms for consistency
        if expr.is_Add:
            expr = sp.Add(*sorted(expr.args, key=str))
        elif expr.is_Mul:
            expr = sp.Mul(*sorted(expr.args, key=str))
        return expr
