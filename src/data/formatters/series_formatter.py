"""
Series Expansion Formatter.

Formats expressions as series expansions (Taylor, Fourier, Neumann).
Useful for approximation-based solutions and iterative methods.
"""

import sympy as sp

from src.data.formatters.base import BaseFormatter


class SeriesFormatter(BaseFormatter):
    """
    Formatter for series expansions.

    Converts expressions to series expansions around a point.
    Supports Taylor series, Laurent series, and custom expansion orders.
    """

    def __init__(self, order: int = 5, x_var: str = "x", x0: float = 0):
        """
        Initialize series formatter.

        Args:
            order: Number of terms in the series expansion.
            x_var: Variable to expand around.
            x0: Point to expand around (default: 0 for Taylor/Maclaurin).
        """
        self.order = order
        self.x_var = x_var
        self.x0 = x0

    @property
    def format_name(self) -> str:
        """Return the name of this format."""
        return "series"

    def to_sympy(self, expression: str) -> sp.Expr:
        """
        Convert series expansion string back to SymPy expression.

        Args:
            expression: Series expansion string (e.g., "1 + x + x**2/2 + O(x**3)").

        Returns:
            SymPy expression (removes O() term).
        """
        # Parse the expression
        expr = sp.sympify(expression)

        # Remove the O() term if present
        if expr.has(sp.O):
            expr = expr.removeO()

        return expr

    def from_sympy(self, expr: sp.Expr, simplify: bool = False) -> str:
        """
        Convert SymPy expression to series expansion.

        Args:
            expr: SymPy expression.
            simplify: Whether to simplify the series terms.

        Returns:
            Series expansion string.
        """
        if simplify:
            expr = self.canonicalize(expr)

        # Get the variable
        x = sp.Symbol(self.x_var)

        # Compute Taylor series expansion
        series = expr.series(x, self.x0, self.order)

        return str(series)

    def format_taylor_series(
        self, expr: sp.Expr | str, n_terms: int | None = None
    ) -> str:
        """
        Format expression as Taylor series.

        Args:
            expr: Expression to expand (SymPy or string).
            n_terms: Number of terms (uses self.order if None).

        Returns:
            Taylor series string.
        """
        if isinstance(expr, str):
            expr = sp.sympify(expr)

        n = n_terms if n_terms is not None else self.order
        x = sp.Symbol(self.x_var)

        series = expr.series(x, self.x0, n)
        return str(series)

    def format_neumann_series(
        self,
        kernel: sp.Expr | str,
        f: sp.Expr | str,
        lambda_val: float = 1.0,
        n_terms: int | None = None,
    ) -> list[str]:
        """
        Format Neumann series for Fredholm equation.

        The Neumann series solution is: u = Σ(λ^n * K^n * f) for n=0 to ∞
        where K^n represents n-fold application of the integral operator.

        Args:
            kernel: Integral kernel K(x,t).
            f: Right-hand side function.
            lambda_val: Lambda coefficient.
            n_terms: Number of terms (uses self.order if None).

        Returns:
            List of Neumann series term strings.
        """
        if isinstance(kernel, str):
            kernel = sp.sympify(kernel)
        if isinstance(f, str):
            f = sp.sympify(f)

        n = n_terms if n_terms is not None else self.order
        x = sp.Symbol("x")
        t = sp.Symbol("t")

        terms = []

        # First term: f(x)
        terms.append(f"Term 0: {f}")

        # Subsequent terms: λ^n * ∫K^n(x,t)f(t)dt
        current_kernel = kernel
        for i in range(1, n):
            term_str = f"Term {i}: λ^{i} * ∫ {current_kernel} * f(t) dt"
            terms.append(term_str)

            # Compose kernel for next iteration (simplified representation)
            # In practice, this would involve nested integrals
            current_kernel = kernel * sp.Symbol(f"K_{i}")

        return terms

    def format_polynomial_approximation(
        self, expr: sp.Expr | str, degree: int | None = None
    ) -> str:
        """
        Format expression as polynomial approximation.

        Args:
            expr: Expression to approximate.
            degree: Polynomial degree (uses self.order-1 if None).

        Returns:
            Polynomial approximation string.
        """
        if isinstance(expr, str):
            expr = sp.sympify(expr)

        deg = degree if degree is not None else (self.order - 1)
        x = sp.Symbol(self.x_var)

        # Get series and remove O() term
        series = expr.series(x, self.x0, deg + 1)
        poly = series.removeO()

        return str(poly)


class NeumannSeriesFormatter(BaseFormatter):
    """
    Specialized formatter for Neumann series solutions to Fredholm equations.

    Formats the iterative Neumann series: u = f + λKf + λ²K²f + ...
    """

    def __init__(self, n_terms: int = 5, include_symbolic: bool = True):
        """
        Initialize Neumann series formatter.

        Args:
            n_terms: Number of terms in the series.
            include_symbolic: Whether to include symbolic K^n notation.
        """
        self.n_terms = n_terms
        self.include_symbolic = include_symbolic

    @property
    def format_name(self) -> str:
        """Return the name of this format."""
        return "neumann"

    def to_sympy(self, expression: str) -> sp.Expr:
        """
        Convert Neumann series string to SymPy expression.

        Args:
            expression: Neumann series representation.

        Returns:
            SymPy expression (approximation).
        """
        # Parse individual terms and sum them
        # This is a simplified version
        return sp.sympify(expression)

    def from_sympy(self, expr: sp.Expr, simplify: bool = False) -> str:
        """
        Not directly applicable for Neumann series.
        Use format_neumann_series instead.
        """
        if simplify:
            expr = self.canonicalize(expr)
        return str(expr)

    def format_neumann_series(
        self,
        f: str | sp.Expr,
        kernel: str | sp.Expr,
        lambda_val: float | str = 1.0,
        bounds: tuple[float | str, float | str] = (0, 1),
    ) -> str:
        """
        Format complete Neumann series for Fredholm equation.

        Args:
            f: Right-hand side function f(x).
            kernel: Integral kernel K(x,t).
            lambda_val: Lambda coefficient.
            bounds: Integration bounds (a, b).

        Returns:
            Neumann series representation string.
        """
        if isinstance(f, str):
            f_expr = sp.sympify(f)
        else:
            f_expr = f

        if isinstance(kernel, str):
            k_expr = sp.sympify(kernel)
        else:
            k_expr = kernel

        a, b = bounds

        # Build the series representation
        terms = [f"u(x) = {f_expr}"]

        for n in range(1, self.n_terms):
            if self.include_symbolic:
                term = f"+ ({lambda_val})^{n} * ∫[{a},{b}] K^{n}(x,t) f(t) dt"
            else:
                term = f"+ ({lambda_val})^{n} * (∫[{a},{b}] {k_expr} ...)^{n}"
            terms.append(term)

        terms.append(f"+ O(λ^{self.n_terms})")

        return " ".join(terms)

    def format_truncated_solution(
        self,
        f: str | sp.Expr,
        kernel: str | sp.Expr,
        lambda_val: float = 1.0,
        n_terms: int | None = None,
    ) -> str:
        """
        Format truncated Neumann series (without O() notation).

        Args:
            f: Right-hand side function.
            kernel: Integral kernel.
            lambda_val: Lambda coefficient.
            n_terms: Number of terms (uses self.n_terms if None).

        Returns:
            Truncated series string.
        """
        n = n_terms if n_terms is not None else self.n_terms

        result = f"u(x) ≈ {f}"
        for i in range(1, n):
            result += f" + λ^{i} * K^{i}(f)"

        return result
