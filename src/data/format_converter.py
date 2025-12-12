"""
Format converter for mathematical expressions.

Converts between LaTeX, RPN, tokenized, and SymPy formats.
"""

from pathlib import Path
from typing import Any

import sympy as sp
from sympy.parsing.latex import parse_latex

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FormatConverter:
    """Converter between mathematical expression formats."""

    FORMATS = ["latex", "rpn", "sympy", "tokenized", "python", "infix"]

    def __init__(self) -> None:
        """Initialize the format converter."""
        self._cache: dict[str, Any] = {}

    def convert(
        self,
        expression: str | sp.Expr,
        source_format: str,
        target_format: str,
    ) -> str | sp.Expr:
        """
        Convert expression between formats.

        Args:
            expression: Input expression.
            source_format: Source format (latex, rpn, sympy, etc.).
            target_format: Target format.

        Returns:
            Converted expression.
        """
        if source_format not in self.FORMATS:
            raise ValueError(f"Unknown source format: {source_format}")
        if target_format not in self.FORMATS:
            raise ValueError(f"Unknown target format: {target_format}")

        # First convert to SymPy (canonical form)
        sympy_expr = self._to_sympy(expression, source_format)

        # Then convert to target format
        return self._from_sympy(sympy_expr, target_format)

    def _to_sympy(self, expression: str | sp.Expr, format: str) -> sp.Expr:
        """Convert any format to SymPy expression."""
        if isinstance(expression, sp.Expr):
            return expression

        if format == "sympy":
            return sp.sympify(expression)
        elif format == "latex":
            return self._latex_to_sympy(expression)
        elif format == "rpn":
            return self._rpn_to_sympy(expression)
        elif format == "infix":
            return sp.sympify(expression)
        elif format == "tokenized":
            return self._tokenized_to_sympy(expression)
        elif format == "python":
            return sp.sympify(expression)
        else:
            raise ValueError(f"Cannot convert from {format} to sympy")

    def _from_sympy(self, expr: sp.Expr, format: str) -> str | sp.Expr:
        """Convert SymPy expression to target format."""
        if format == "sympy":
            return expr
        elif format == "latex":
            return sp.latex(expr)
        elif format == "rpn":
            return self._sympy_to_rpn(expr)
        elif format == "infix":
            return str(expr)
        elif format == "tokenized":
            return self._sympy_to_tokenized(expr)
        elif format == "python":
            from sympy.printing.pycode import pycode
            return pycode(expr)
        else:
            raise ValueError(f"Cannot convert from sympy to {format}")

    def _latex_to_sympy(self, latex_str: str) -> sp.Expr:
        """Convert LaTeX string to SymPy expression."""
        try:
            return parse_latex(latex_str)
        except Exception as e:
            logger.warning(f"LaTeX parsing failed: {e}")
            # Fallback: try basic sympify with substitutions
            cleaned = latex_str.replace("\\", "").replace("{", "(").replace("}", ")")
            return sp.sympify(cleaned)

    def _rpn_to_sympy(self, rpn_str: str) -> sp.Expr:
        """Convert RPN (Reverse Polish Notation) to SymPy expression."""
        tokens = rpn_str.split()
        stack: list[sp.Expr] = []

        operators = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b,
            "^": lambda a, b: a ** b,
            "**": lambda a, b: a ** b,
        }

        functions = {
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
        }

        x, t = sp.symbols("x t")
        symbols = {"x": x, "t": t}

        for token in tokens:
            if token in operators:
                b = stack.pop()
                a = stack.pop()
                stack.append(operators[token](a, b))
            elif token in functions:
                a = stack.pop()
                stack.append(functions[token](a))
            elif token in symbols:
                stack.append(symbols[token])
            else:
                try:
                    stack.append(sp.Rational(token))
                except Exception:
                    stack.append(sp.Symbol(token))

        return stack[0] if stack else sp.Integer(0)

    def _sympy_to_rpn(self, expr: sp.Expr) -> str:
        """Convert SymPy expression to RPN string."""
        # TODO: Implement full RPN conversion
        # This is a simplified version
        tokens = []
        self._expr_to_rpn_tokens(expr, tokens)
        return " ".join(tokens)

    def _expr_to_rpn_tokens(self, expr: sp.Expr, tokens: list[str]) -> None:
        """Recursively convert expression to RPN tokens."""
        if expr.is_number:
            tokens.append(str(expr))
        elif expr.is_symbol:
            tokens.append(str(expr))
        elif expr.is_Add:
            args = list(expr.args)
            self._expr_to_rpn_tokens(args[0], tokens)
            for arg in args[1:]:
                self._expr_to_rpn_tokens(arg, tokens)
                tokens.append("+")
        elif expr.is_Mul:
            args = list(expr.args)
            self._expr_to_rpn_tokens(args[0], tokens)
            for arg in args[1:]:
                self._expr_to_rpn_tokens(arg, tokens)
                tokens.append("*")
        elif expr.is_Pow:
            base, exp = expr.as_base_exp()
            self._expr_to_rpn_tokens(base, tokens)
            self._expr_to_rpn_tokens(exp, tokens)
            tokens.append("^")
        else:
            # For functions like sin, cos, etc.
            func_name = expr.func.__name__
            for arg in expr.args:
                self._expr_to_rpn_tokens(arg, tokens)
            tokens.append(func_name)

    def _tokenized_to_sympy(self, tokenized: str) -> sp.Expr:
        """Convert tokenized format to SymPy expression."""
        # TODO: Implement tokenized parsing
        # Tokenized format: space-separated tokens
        return sp.sympify(tokenized.replace(" ", ""))

    def _sympy_to_tokenized(self, expr: sp.Expr) -> str:
        """Convert SymPy expression to tokenized format."""
        # TODO: Implement full tokenization
        expr_str = str(expr)
        # Add spaces around operators
        for op in ["+", "-", "*", "/", "^", "(", ")"]:
            expr_str = expr_str.replace(op, f" {op} ")
        return " ".join(expr_str.split())


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
    # TODO: Implement file-based conversion
    logger.info(f"Converting {input_file} to {target_format} format")
    return []


# Convenience functions
def latex_to_sympy(latex_str: str) -> sp.Expr:
    """Convert LaTeX to SymPy."""
    converter = FormatConverter()
    return converter._latex_to_sympy(latex_str)


def sympy_to_latex(expr: sp.Expr) -> str:
    """Convert SymPy to LaTeX."""
    return sp.latex(expr)


def sympy_to_rpn(expr: sp.Expr) -> str:
    """Convert SymPy to RPN."""
    converter = FormatConverter()
    return converter._sympy_to_rpn(expr)


def rpn_to_sympy(rpn_str: str) -> sp.Expr:
    """Convert RPN to SymPy."""
    converter = FormatConverter()
    return converter._rpn_to_sympy(rpn_str)
