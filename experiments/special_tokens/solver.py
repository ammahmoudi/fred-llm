"""Deterministic sympy solver for separable-kernel Fredholm equations (v2 tool).

Degenerate-kernel method: if K(x,t) = sum_i g_i(x) h_i(t), then
u(x) = f(x) + lam * sum_j c_j g_j(x) where (I - lam*A) c = b,
A_ij = ∫ h_i(t) g_j(t) dt,  b_i = ∫ h_i(t) f(t) dt  over [a, b].

Input parsing cross-validates Math-Verify against sympy's antlr parse_latex
numerically (Math-Verify silently drops terms on some inputs); disagreement
means we skip the item rather than solve the wrong equation.

Requires: `uv pip install "antlr4-python3-runtime==4.11.*"` (sympy parse_latex).
"""

import warnings

import numpy as np
import sympy as sp

from src.evaluation.core import _metric_alarm, _MetricTimeout
from src.llm.math_verify_adapter import parse_latex_to_sympy

warnings.filterwarnings("ignore")

X, T = sp.Symbol("x"), sp.Symbol("t")
_INTEGRATE_TIMEOUT_S = 4.0


def _normalize(expr: sp.Expr) -> sp.Expr:
    """Map any x/t/e symbol variants to plain X, T, E."""
    subs = {}
    for s in expr.free_symbols:
        if s.name == "x":
            subs[s] = X
        elif s.name == "t":
            subs[s] = T
        elif s.name == "e":
            subs[s] = sp.E
        elif s.name == "pi":
            subs[s] = sp.pi
    return expr.subs(subs)


def _agree(e1: sp.Expr, e2: sp.Expr, n: int = 8) -> bool:
    rng = np.random.default_rng(0)
    f1 = sp.lambdify((X, T), e1, "numpy")
    f2 = sp.lambdify((X, T), e2, "numpy")
    for _ in range(n):
        xv, tv = rng.uniform(0.1, 2.0, 2)
        try:
            v1, v2 = complex(f1(xv, tv)), complex(f2(xv, tv))
        except Exception:
            return False
        if not np.isfinite([v1.real, v2.real]).all():
            continue
        if abs(v1 - v2) > 1e-6 * (1 + abs(v1) + abs(v2)):
            return False
    return True


def parse_eq_latex(latex_str: str) -> sp.Expr | None:
    """Parse LaTeX to sympy, trusting antlr but cross-checking Math-Verify."""
    antlr_expr = mv_expr = None
    try:
        from sympy.parsing.latex import parse_latex

        antlr_expr = _normalize(parse_latex(latex_str))
    except Exception:
        pass
    try:
        mv_expr = _normalize(parse_latex_to_sympy(latex_str))
    except Exception:
        pass
    if antlr_expr is None:
        return None  # mv alone is untrusted (known silent term drops)
    if mv_expr is not None and not _agree(antlr_expr, mv_expr):
        return None  # parsers disagree -> don't solve the wrong equation
    return antlr_expr


def decompose_separable(kernel: sp.Expr) -> tuple[list, list] | None:
    """Split K(x,t) into (g_i(x), h_i(t)) or None if any factor mixes x,t."""
    gs, hs = [], []
    for term in sp.Add.make_args(sp.expand(kernel)):
        g, h = sp.S.One, sp.S.One
        for fac in sp.Mul.make_args(term):
            fs = fac.free_symbols & {X, T}
            if fs <= {X}:
                g *= fac
            elif fs == {T}:
                h *= fac
            else:
                return None
        gs.append(g)
        hs.append(h)
    return gs, hs


def _integrate(expr: sp.Expr, a: float, b: float) -> float | None:
    """∫ over t in [a,b] as a float: bounded symbolic, numeric quad fallback.

    Entries are always constants; keeping them symbolic (e.g. unevaluated
    e^{122}·erf(...) forms) makes the downstream determinant explode.
    """
    try:
        with _metric_alarm(_INTEGRATE_TIMEOUT_S):
            val = sp.integrate(expr, (T, a, b))
        if not val.has(sp.Integral):
            v = complex(val.evalf())
            if abs(v.imag) < 1e-9 * (1 + abs(v.real)) and np.isfinite(v.real):
                return float(v.real)
    except (_MetricTimeout, Exception):
        pass
    try:
        from scipy.integrate import quad

        fn = sp.lambdify(T, expr, "numpy")
        val, _ = quad(fn, a, b, limit=200)
        return float(val) if np.isfinite(val) else None
    except Exception:
        return None


def solve_separable(
    kernel: sp.Expr, f: sp.Expr, lam: float, a: float, b: float
) -> sp.Expr | None:
    """Solve u(x) - lam ∫_a^b K(x,t) u(t) dt = f(x); None if not solvable here."""
    parts = decompose_separable(kernel)
    if parts is None:
        return None
    gs, hs = parts
    n = len(gs)
    if n > 6:  # ponytail: rank cap; dataset kernels are rank 1-4
        return None
    f_t = f.subs(X, T)
    A = np.zeros((n, n))
    bvec = np.zeros(n)
    for i in range(n):
        for j in range(n):
            val = _integrate(hs[i] * gs[j].subs(X, T), a, b)
            if val is None:
                return None
            A[i, j] = val
        val = _integrate(hs[i] * f_t, a, b)
        if val is None:
            return None
        bvec[i] = val
    M = np.eye(n) - lam * A
    det = np.linalg.det(M)
    if not np.isfinite(det) or abs(det) < 1e-10 * max(1.0, np.abs(M).max() ** n):
        return None  # resonance: family/none territory, not ours
    c = np.linalg.solve(M, bvec)
    if not np.isfinite(c).all():
        return None
    try:
        with _metric_alarm(_INTEGRATE_TIMEOUT_S):
            u = f + lam * sum(float(c[j]) * gs[j] for j in range(n))
            return sp.expand(u)
    except (_MetricTimeout, Exception):
        return None


if __name__ == "__main__":
    # classic fixture: K=x*t on [0,1], f=x, lam=1 -> u = 3x/2
    u = solve_separable(X * T, X, 1.0, 0.0, 1.0)
    assert u is not None and sp.simplify(u - sp.Rational(3, 2) * X) == 0, u
    # rank-2 with mixed constant term
    u2 = solve_separable(X + T, X, 0.5, 0.0, 1.0)
    assert u2 is not None
    # non-separable must refuse
    assert solve_separable(sp.cos(X * T), X, 1.0, 0.0, 1.0) is None
    assert solve_separable(sp.exp(-sp.Abs(T - X)), X, 1.0, 0.0, 1.0) is None
    # the known Math-Verify drop case -> parsers disagree -> refused
    assert parse_eq_latex(r"t^{3} e^{- 0.6923 t + 16.39 x} + 8.388 t") is None
    # agreeing case parses
    e = parse_eq_latex(r"\sin{\left(x \right)} \cos{\left(t \right)}")
    assert e is not None and e.equals(sp.sin(X) * sp.cos(T))
    print("solver self-check OK:", u, "|", sp.nsimplify(u2))
