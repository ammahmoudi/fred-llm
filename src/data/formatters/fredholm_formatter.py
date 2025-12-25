"""
Fredholm Equation Formatter.

Formats complete Fredholm integral equations with all components.
Converts structured equation components to formatted strings.
"""

import sympy as sp

from src.data.formatters.base import BaseFormatter
from src.data.formatters.infix_formatter import InfixFormatter
from src.data.formatters.latex_formatter import LaTeXFormatter
from src.data.formatters.rpn_formatter import RPNFormatter


class FredholmEquationFormatter:
    """
    Formatter for complete Fredholm integral equations.

    Combines all equation components (u, f, kernel, lambda, bounds) into
    formatted strings suitable for LLM input/output.
    """

    def __init__(self, expression_format: str = "infix", simplify: bool = False):
        """
        Initialize formatter.

        Args:
            expression_format: Format for expressions ("infix", "latex", "rpn").
            simplify: Whether to canonicalize/simplify expressions.
        """
        self.format = expression_format
        self.simplify = simplify

        # Get appropriate expression formatter
        if expression_format == "latex":
            self._expr_formatter = LaTeXFormatter()
        elif expression_format == "rpn":
            self._expr_formatter = RPNFormatter()
        else:
            self._expr_formatter = InfixFormatter()

    def format_equation(
        self,
        u: str | sp.Expr | None = None,
        f: str | sp.Expr | None = None,
        kernel: str | sp.Expr | None = None,
        lambda_val: str | float | None = None,
        a: str | float | None = None,
        b: str | float | None = None,
        equation_dict: dict | None = None,
    ) -> str:
        """
        Format a complete Fredholm equation.

        Args:
            u: Solution function u(x).
            f: Right-hand side f(x).
            kernel: Kernel function K(x,t).
            lambda_val: Lambda parameter.
            a: Lower integration bound.
            b: Upper integration bound.
            equation_dict: Alternative input as dictionary.

        Returns:
            Formatted equation string.

        Example:
            >>> formatter = FredholmEquationFormatter("infix")
            >>> eq = formatter.format_equation(
            ...     u="x**2", f="x**2 + 2*x", kernel="x*t",
            ...     lambda_val="0.5", a="0", b="1"
            ... )
            >>> # Returns: "u(x) - 0.5 * ∫[0,1] (x*t) u(t) dt = x**2 + 2*x"
        """
        # Handle dict input
        if equation_dict:
            u = equation_dict.get("u", u)
            f = equation_dict.get("f", f)
            kernel = equation_dict.get("kernel", kernel)
            lambda_val = equation_dict.get(
                "lambda", equation_dict.get("lambda_val", lambda_val)
            )
            a = equation_dict.get("a", a)
            b = equation_dict.get("b", b)

        # Convert to SymPy and format
        u_expr = self._prepare_expr(u) if u else "u(x)"
        f_expr = self._prepare_expr(f) if f else "f(x)"
        k_expr = self._prepare_expr(kernel) if kernel else "K(x,t)"

        lambda_str = str(lambda_val) if lambda_val else "λ"
        a_str = str(a) if a else "a"
        b_str = str(b) if b else "b"

        # Format based on chosen style
        if self.format == "latex":
            return self._format_latex(u_expr, f_expr, k_expr, lambda_str, a_str, b_str)
        elif self.format == "rpn":
            return self._format_rpn(u_expr, f_expr, k_expr, lambda_str, a_str, b_str)
        else:
            return self._format_infix(u_expr, f_expr, k_expr, lambda_str, a_str, b_str)

    def format_components(self, equation_dict: dict) -> dict[str, str]:
        """
        Format individual equation components.

        Args:
            equation_dict: Dictionary with u, f, kernel, lambda, a, b.

        Returns:
            Dictionary with formatted components.
        """
        formatted = {}

        for key in ["u", "f", "kernel"]:
            if key in equation_dict:
                expr = self._prepare_expr(equation_dict[key])
                formatted[key] = expr

        for key in ["lambda", "lambda_val", "a", "b"]:
            if key in equation_dict:
                formatted[key] = str(equation_dict[key])

        return formatted

    def _prepare_expr(self, expr: str | sp.Expr) -> str:
        """Convert expression to formatted string."""
        if isinstance(expr, str):
            expr = sp.sympify(expr)

        if self.simplify:
            expr = sp.simplify(expr)

        return self._expr_formatter.from_sympy(expr)

    def _format_infix(self, u: str, f: str, k: str, lam: str, a: str, b: str) -> str:
        """Format as infix notation."""
        return f"u(x) - {lam} * ∫[{a},{b}] ({k}) u(t) dt = {f}"

    def _format_latex(self, u: str, f: str, k: str, lam: str, a: str, b: str) -> str:
        """Format as LaTeX."""
        return f"u(x) - {lam} \\int_{{{a}}}^{{{b}}} {k} \\, u(t) \\, dt = {f}"

    def _format_rpn(self, u: str, f: str, k: str, lam: str, a: str, b: str) -> str:
        """Format as RPN-style tokens."""
        return f"u x {f} = {k} u t * {lam} * {a} {b} ∫ -"


class TokenizedEquationFormatter(FredholmEquationFormatter):
    """
    Tokenized formatter with special tokens for equation components.

    Adds special markers like <INT>, <LAMBDA>, <SEP> for better LLM parsing.
    """

    def __init__(
        self,
        simplify: bool = False,
        use_special_tokens: bool = True,
    ):
        """
        Initialize tokenized formatter.

        Args:
            simplify: Whether to canonicalize expressions.
            use_special_tokens: Whether to use special tokens.
        """
        super().__init__(expression_format="infix", simplify=simplify)
        self.use_special_tokens = use_special_tokens

    def format_equation(
        self,
        u: str | sp.Expr | None = None,
        f: str | sp.Expr | None = None,
        kernel: str | sp.Expr | None = None,
        lambda_val: str | float | None = None,
        a: str | float | None = None,
        b: str | float | None = None,
        equation_dict: dict | None = None,
    ) -> str:
        """Format equation with space-separated tokens and special markers."""
        # Handle dict input
        if equation_dict:
            u = equation_dict.get("u", u)
            f = equation_dict.get("f", f)
            kernel = equation_dict.get("kernel", kernel)
            lambda_val = equation_dict.get(
                "lambda", equation_dict.get("lambda_val", lambda_val)
            )
            a = equation_dict.get("a", a)
            b = equation_dict.get("b", b)

        # Format components
        u_expr = self._tokenize_expr(self._prepare_expr(u)) if u else "u ( x )"
        f_expr = self._tokenize_expr(self._prepare_expr(f)) if f else "f ( x )"
        k_expr = (
            self._tokenize_expr(self._prepare_expr(kernel)) if kernel else "K ( x , t )"
        )

        lambda_str = str(lambda_val) if lambda_val else "λ"
        a_str = str(a) if a else "a"
        b_str = str(b) if b else "b"

        if self.use_special_tokens:
            return (
                f"u ( x ) - <LAMBDA> {lambda_str} <INT> <LOWER> {a_str} <UPPER> {b_str} "
                f"{k_expr} u ( t ) dt <SEP> {f_expr}"
            )
        else:
            return f"u ( x ) - {lambda_str} * ∫ {a_str} {b_str} {k_expr} u ( t ) dt = {f_expr}"

    def _tokenize_expr(self, expr_str: str) -> str:
        """Add spaces around all operators."""
        for op in ["**", "+", "-", "*", "/", "(", ")", ",", "^"]:
            expr_str = expr_str.replace(op, f" {op} ")
        return " ".join(expr_str.split())
