"""
Mathematical utilities for Fredholm integral equations.

Provides numerical integration, evaluation, and norm computation.
"""

import re
from typing import Callable

import numpy as np
import sympy as sp
from scipy import integrate

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Compile regex pattern once for performance
_IMPLICIT_MULT_PATTERN = re.compile(r'(\d+\.?\d*(?:[eE][+-]?\d+)?)([a-zA-Z])')


def fix_implicit_multiplication(expression: str) -> str:
    """
    Fix implicit multiplication in mathematical expressions.
    
    Converts expressions like '2x' to '2*x', '3.14x**2' to '3.14*x**2', etc.
    This is needed because SymPy's sympify doesn't handle implicit multiplication.
    
    Args:
        expression: String expression with potential implicit multiplication.
        
    Returns:
        Expression with explicit multiplication operators.
        
    Examples:
        >>> fix_implicit_multiplication("2x + 3y")
        '2*x + 3*y'
        >>> fix_implicit_multiplication("3.14x**2")
        '3.14*x**2'
        >>> fix_implicit_multiplication("2.5x*y")
        '2.5*x*y'
    """
    # Use pre-compiled pattern for better performance on large datasets
    return _IMPLICIT_MULT_PATTERN.sub(r'\1*\2', expression)


def integrate_kernel(
    kernel: sp.Expr | Callable,
    u: sp.Expr | Callable,
    a: float,
    b: float,
    x_val: float | None = None,
    method: str = "quad",
) -> float | sp.Expr:
    """
    Compute ∫_a^b K(x, t) u(t) dt.

    Args:
        kernel: Kernel function K(x, t).
        u: Function u(t).
        a: Lower integration bound.
        b: Upper integration bound.
        x_val: Value of x (for numeric evaluation).
        method: Integration method (quad, trapz, simpson, symbolic).

    Returns:
        Integral value or symbolic expression.
    """
    x, t = sp.symbols("x t")

    if method == "symbolic":
        # Symbolic integration
        if isinstance(kernel, sp.Expr) and isinstance(u, sp.Expr):
            u_of_t = u.subs(x, t)
            integrand = kernel * u_of_t
            result = sp.integrate(integrand, (t, a, b))
            return result
        else:
            raise ValueError("Symbolic integration requires SymPy expressions")

    # Numeric integration
    if isinstance(kernel, sp.Expr):
        kernel_func = sp.lambdify((x, t), kernel, modules=["numpy"])
    else:
        kernel_func = kernel

    if isinstance(u, sp.Expr):
        u_func = sp.lambdify(t, u.subs(x, t), modules=["numpy"])
    else:
        u_func = u

    def integrand(t_val: float) -> float:
        return kernel_func(x_val, t_val) * u_func(t_val)

    if method == "quad":
        result, error = integrate.quad(integrand, a, b)
        return result
    elif method == "trapz":
        t_points = np.linspace(a, b, 100)
        y_points = [integrand(ti) for ti in t_points]
        return float(np.trapz(y_points, t_points))
    elif method == "simpson":
        t_points = np.linspace(a, b, 101)  # Odd number for Simpson
        y_points = [integrand(ti) for ti in t_points]
        return float(integrate.simpson(y_points, x=t_points))
    else:
        raise ValueError(f"Unknown integration method: {method}")


def evaluate_at_points(
    expr: sp.Expr | Callable,
    points: np.ndarray | list[float],
    var: sp.Symbol | None = None,
) -> np.ndarray:
    """
    Evaluate an expression at multiple points.

    Args:
        expr: Expression or function to evaluate.
        points: Points at which to evaluate.
        var: Variable symbol (default: x).

    Returns:
        Array of evaluated values.
    """
    points = np.asarray(points)

    if callable(expr) and not isinstance(expr, sp.Expr):
        return np.array([expr(p) for p in points])

    if var is None:
        var = sp.Symbol("x")

    func = sp.lambdify(var, expr, modules=["numpy"])
    return np.array([func(p) for p in points])


def compute_norm(
    f: sp.Expr | Callable | np.ndarray,
    g: sp.Expr | Callable | np.ndarray | None = None,
    a: float = 0,
    b: float = 1,
    n_points: int = 100,
    norm_type: str = "L2",
) -> float:
    """
    Compute the norm of f or the distance ||f - g||.

    Args:
        f: First function or array.
        g: Second function or array (optional).
        a: Lower bound for integration.
        b: Upper bound for integration.
        n_points: Number of points for numerical integration.
        norm_type: Type of norm (L2, L1, Linf).

    Returns:
        Norm value.
    """
    x_points = np.linspace(a, b, n_points)

    # Evaluate functions
    if isinstance(f, np.ndarray):
        f_vals = f
    else:
        f_vals = evaluate_at_points(f, x_points)

    if g is not None:
        if isinstance(g, np.ndarray):
            g_vals = g
        else:
            g_vals = evaluate_at_points(g, x_points)
        diff = f_vals - g_vals
    else:
        diff = f_vals

    if norm_type == "L2":
        # ||f||_2 = sqrt(∫|f|^2 dx)
        return float(np.sqrt(np.trapz(np.abs(diff) ** 2, x_points)))
    elif norm_type == "L1":
        # ||f||_1 = ∫|f| dx
        return float(np.trapz(np.abs(diff), x_points))
    elif norm_type == "Linf":
        # ||f||_∞ = max|f|
        return float(np.max(np.abs(diff)))
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def verify_fredholm_solution(
    u: sp.Expr,
    kernel: sp.Expr,
    f: sp.Expr,
    lambda_val: float,
    a: float = 0,
    b: float = 1,
    n_points: int = 50,
) -> dict[str, float]:
    """
    Verify a solution to the Fredholm equation.

    Checks: u(x) - λ ∫_a^b K(x,t) u(t) dt = f(x)

    Args:
        u: Proposed solution.
        kernel: Kernel function K(x, t).
        f: Right-hand side f(x).
        lambda_val: Lambda parameter.
        a: Lower integration bound.
        b: Upper integration bound.
        n_points: Number of test points.

    Returns:
        Dictionary with verification metrics.
    """
    x = sp.Symbol("x")
    x_points = np.linspace(a, b, n_points)

    u_func = sp.lambdify(x, u, modules=["numpy"])
    f_func = sp.lambdify(x, f, modules=["numpy"])

    residuals = []

    for xi in x_points:
        # Compute integral numerically
        integral_val = integrate_kernel(kernel, u, a, b, x_val=xi, method="quad")

        # Compute residual: u(x) - λ*integral - f(x)
        lhs = u_func(xi) - lambda_val * integral_val
        rhs = f_func(xi)
        residuals.append(abs(lhs - rhs))

    residuals = np.array(residuals)

    return {
        "max_residual": float(np.max(residuals)),
        "mean_residual": float(np.mean(residuals)),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "verified": float(np.max(residuals)) < 1e-6,
    }


def separable_kernel_solve(
    phi: list[sp.Expr],
    psi: list[sp.Expr],
    f: sp.Expr,
    lambda_val: float,
    a: float = 0,
    b: float = 1,
) -> sp.Expr | None:
    """
    Solve Fredholm equation with separable kernel.

    Separable kernel: K(x, t) = Σ φ_i(x) ψ_i(t)

    Args:
        phi: List of φ_i(x) functions.
        psi: List of ψ_i(t) functions.
        f: Right-hand side f(x).
        lambda_val: Lambda parameter.
        a: Lower integration bound.
        b: Upper integration bound.

    Returns:
        Solution u(x) or None if not solvable.
    """
    x, t = sp.symbols("x t")
    n = len(phi)

    if len(psi) != n:
        raise ValueError("phi and psi must have the same length")

    # TODO: Implement separable kernel solution
    # This involves solving a system of linear equations
    logger.info(f"Solving separable kernel with {n} terms")

    return None


def series_solution(
    kernel: sp.Expr,
    f: sp.Expr,
    lambda_val: float,
    n_terms: int = 5,
    a: float = 0,
    b: float = 1,
) -> sp.Expr:
    """
    Compute series solution (Neumann series).

    u = f + λKf + λ²K²f + ...

    Args:
        kernel: Kernel function K(x, t).
        f: Right-hand side f(x).
        lambda_val: Lambda parameter.
        n_terms: Number of terms in series.
        a: Lower integration bound.
        b: Upper integration bound.

    Returns:
        Approximate solution as SymPy expression.
    """
    x, t = sp.symbols("x t")

    # TODO: Implement Neumann series computation
    logger.info(f"Computing Neumann series with {n_terms} terms")

    # Placeholder: return f as zeroth approximation
    return f
