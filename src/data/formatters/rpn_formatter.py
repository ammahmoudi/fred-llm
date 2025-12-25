"""
RPN (Reverse Polish Notation) formatter.

Converts between RPN and SymPy expressions.
RPN is a postfix notation where operators follow operands.
Example: "x 2 +" represents "x + 2"
"""

import sympy as sp

from src.data.formatters.base import BaseFormatter
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RPNFormatter(BaseFormatter):
    """Formatter for Reverse Polish Notation."""

    @property
    def format_name(self) -> str:
        """Return format name."""
        return "rpn"

    def to_sympy(self, expression: str) -> sp.Expr:
        """
        Convert RPN string to SymPy expression.

        Args:
            expression: RPN string (e.g., "x 2 +").

        Returns:
            SymPy expression.
        """
        tokens = expression.split()
        stack: list[sp.Expr] = []

        operators = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b,
            "^": lambda a, b: a**b,
            "**": lambda a, b: a**b,
        }

        functions = {
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "sinh": sp.sinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "neg": lambda a: -a,
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

    def from_sympy(self, expr: sp.Expr, simplify: bool = False) -> str:
        """
        Convert SymPy expression to RPN string.

        Args:
            expr: SymPy expression.
            simplify: Whether to canonicalize the expression first.

        Returns:
            RPN string.
        """
        if simplify:
            expr = self.canonicalize(expr)
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
            if len(args) > 0 and args[0].is_number and args[0].is_negative:
                # Handle negative multiplication
                self._expr_to_rpn_tokens(-args[0], tokens)
                for arg in args[1:]:
                    self._expr_to_rpn_tokens(arg, tokens)
                    tokens.append("*")
                tokens.append("neg")
            else:
                self._expr_to_rpn_tokens(args[0], tokens)
                for arg in args[1:]:
                    self._expr_to_rpn_tokens(arg, tokens)
                    tokens.append("*")
        elif expr.is_Pow:
            base, exp = expr.as_base_exp()
            self._expr_to_rpn_tokens(base, tokens)
            self._expr_to_rpn_tokens(exp, tokens)
            tokens.append("^")
        elif expr.func == sp.sin:
            self._expr_to_rpn_tokens(expr.args[0], tokens)
            tokens.append("sin")
        elif expr.func == sp.cos:
            self._expr_to_rpn_tokens(expr.args[0], tokens)
            tokens.append("cos")
        elif expr.func == sp.tan:
            self._expr_to_rpn_tokens(expr.args[0], tokens)
            tokens.append("tan")
        elif expr.func == sp.exp:
            self._expr_to_rpn_tokens(expr.args[0], tokens)
            tokens.append("exp")
        elif expr.func == sp.log:
            self._expr_to_rpn_tokens(expr.args[0], tokens)
            tokens.append("log")
        elif expr.func == sp.sqrt:
            self._expr_to_rpn_tokens(expr.args[0], tokens)
            tokens.append("sqrt")
        elif expr.func in (sp.sinh, sp.cosh, sp.tanh):
            self._expr_to_rpn_tokens(expr.args[0], tokens)
            tokens.append(expr.func.__name__)
        else:
            # Generic function handling
            func_name = expr.func.__name__.lower()
            for arg in expr.args:
                self._expr_to_rpn_tokens(arg, tokens)
            tokens.append(func_name)
