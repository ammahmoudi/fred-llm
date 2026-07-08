"""Symbolic comparison metric for solutions."""

import signal
import threading
from typing import Any, Callable

import sympy as sp

from src.llm.math_verify_adapter import math_verify_compare
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _run_with_timeout(seconds: float, func: Callable[[], Any]) -> tuple[bool, Any]:
    """Run ``func`` with a hard SIGALRM timeout.

    SIGALRM only fires on the main thread, so off-thread callers (and platforms
    without ``setitimer``) run untimed — i.e. the original behavior, no
    regression. Returns ``(ok, value)`` where ``ok`` is False on timeout.
    """
    on_main = threading.current_thread() is threading.main_thread()
    if not on_main or not hasattr(signal, "setitimer"):
        return True, func()

    def _handler(signum: int, frame: Any) -> None:
        raise TimeoutError("symbolic comparison timed out")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return True, func()
    except TimeoutError:
        return False, None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def symbolic_compare(
    solution: sp.Expr,
    ground_truth: sp.Expr,
    tolerance: float = 1e-10,
    use_math_verify: bool = True,
    timeout_s: float = 8.0,
) -> dict[str, Any]:
    """
    Compare two symbolic expressions for equivalence.

    Args:
        solution: Generated solution as SymPy expression.
        ground_truth: Expected solution as SymPy expression.
        tolerance: Tolerance for numerical comparison.
        use_math_verify: Try the Math-Verify fast-path first.
        timeout_s: Hard cap on the untimed sympy fallback (simplify/equals/
            trigsimp), which can otherwise run for minutes on the gnarly
            expressions reasoning models emit. On timeout we return
            inconclusive (equivalent=False); the numeric metric still scores it.

    Returns:
        Dictionary with comparison results.
    """
    result = {
        "equivalent": False,
        "difference": None,
        "simplified_match": False,
    }

    try:
        # Math-Verify fast-path: quick boolean check before heavy simplification
        if use_math_verify:
            mv_result = math_verify_compare(solution, ground_truth)
            if mv_result is True:
                result["equivalent"] = True
                result["simplified_match"] = True
                return result

        def _sympy_fallback() -> dict[str, Any]:
            sol, gt = solution, ground_truth
            # Evaluate any unevaluated Integral objects first
            if sol.has(sp.Integral):
                sol = sol.doit()
            if gt.has(sp.Integral):
                gt = gt.doit()

            out = {"equivalent": False, "difference": None, "simplified_match": False}

            # Direct symbolic equality
            if sp.simplify(sol - gt) == 0:
                out["equivalent"] = True
                out["simplified_match"] = True
                return out

            diff = sp.simplify(sol - gt)
            out["difference"] = str(diff)
            if diff.equals(sp.Integer(0)):
                out["equivalent"] = True
                out["simplified_match"] = True
            if sp.expand(sol - gt) == 0:
                out["equivalent"] = True
            if sp.trigsimp(sol - gt) == 0:
                out["equivalent"] = True
            return out

        ok, val = _run_with_timeout(timeout_s, _sympy_fallback)
        if not ok:
            logger.warning(
                "Symbolic comparison timed out after %ss; numeric metric still applies",
                timeout_s,
            )
            result["error"] = "symbolic_timeout"
            return result
        result.update(val)

    except Exception as e:
        logger.warning(f"Symbolic comparison failed: {e}")
        result["error"] = str(e)

    return result
