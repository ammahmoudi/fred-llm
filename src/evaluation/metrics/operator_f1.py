"""Operator F1 metric for comparing solution operators."""

from typing import Any

import sympy as sp

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Operators we track, matching article Appendix C
_TRACKED_OPERATORS: set[type] = {
    sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.sqrt,
    sp.sinh, sp.cosh, sp.tanh, sp.Abs,
    sp.Add, sp.Mul, sp.Pow,
    sp.Integral,
}

_OPERATOR_NAMES: dict[type, str] = {
    sp.sin: "sin", sp.cos: "cos", sp.tan: "tan",
    sp.exp: "exp", sp.log: "log", sp.sqrt: "sqrt",
    sp.sinh: "sinh", sp.cosh: "cosh", sp.tanh: "tanh",
    sp.Abs: "Abs",
    sp.Add: "Add", sp.Mul: "Mul", sp.Pow: "Pow",
    sp.Integral: "Integral",
}


def extract_operators(expr: sp.Expr) -> set[str]:
    """
    Recursively walk a SymPy expression tree and return the set of
    operator/function names found.

    Args:
        expr: A SymPy expression.

    Returns:
        Set of operator name strings (e.g. {"sin", "Add", "Pow"}).
    """
    ops: set[str] = set()

    def _walk(e: sp.Basic) -> None:
        func = type(e)
        if func in _OPERATOR_NAMES:
            ops.add(_OPERATOR_NAMES[func])
        for arg in e.args:
            _walk(arg)

    _walk(expr)
    return ops


def operator_f1(
    pred_expr: sp.Expr, gt_expr: sp.Expr
) -> dict[str, Any]:
    """
    Compute Operator F1 (precision, recall, F1) between predicted and
    ground-truth expressions based on the set of operators each contains.

    Args:
        pred_expr: Predicted SymPy expression.
        gt_expr: Ground-truth SymPy expression.

    Returns:
        Dict with precision, recall, f1, pred_ops, gt_ops.
    """
    pred_ops = extract_operators(pred_expr)
    gt_ops = extract_operators(gt_expr)

    if not pred_ops and not gt_ops:
        return {
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "pred_ops": [], "gt_ops": [],
        }

    tp = len(pred_ops & gt_ops)
    precision = tp / len(pred_ops) if pred_ops else 0.0
    recall = tp / len(gt_ops) if gt_ops else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_ops": sorted(pred_ops),
        "gt_ops": sorted(gt_ops),
    }
