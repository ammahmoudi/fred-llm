"""
Adapter for HuggingFace Math-Verify library.

Provides unified parsing and comparison with graceful fallback
to custom LaTeX-to-SymPy logic when Math-Verify is unavailable.
"""

import re
from typing import Optional

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# --- Optional Math-Verify import ---
try:
    from math_verify import LatexExtractionConfig, parse, verify

    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False

# --- Consolidated Fredholm symbol dictionary ---
# Previously duplicated in postprocess.py, evaluate.py, and adaptive_pipeline.py
_x, _t = sp.symbols("x t")
_C, _c_1, _c_2 = sp.symbols("C c_1 c_2")
_C_1, _C_2 = sp.symbols("C_1 C_2")
_n, _k = sp.symbols("n k")

FREDHOLM_LOCAL_DICT: dict[str, sp.Basic] = {
    "x": _x, "t": _t, "e": sp.E, "pi": sp.pi,
    "C": _C, "c_1": _c_1, "c_2": _c_2,
    "C_1": _C_1, "C_2": _C_2,
    "n": _n, "k": _k,
    "Integral": sp.Integral,
    "oo": sp.oo,
}

TRANSFORMATIONS = standard_transformations + (implicit_multiplication,)


def parse_latex_to_sympy(expr_str: str) -> sp.Expr:
    """
    Parse a LaTeX or infix expression string to a SymPy expression.

    Tries Math-Verify's ``parse()`` first (better LaTeX handling),
    then falls back to the custom ``_latex_to_infix()`` + ``parse_expr()`` path.

    Args:
        expr_str: Mathematical expression (LaTeX or infix).

    Returns:
        Parsed SymPy expression.

    Raises:
        ParseError: If all parsing strategies fail.
    """
    from src.llm.postprocess import ParseError, _latex_to_infix, _preprocess_for_sympy

    # --- Strategy 1: Math-Verify parse ---
    if HAS_MATH_VERIFY:
        mv_expr = _try_math_verify_parse(expr_str)
        if mv_expr is not None:
            return mv_expr

    # --- Strategy 2: Custom _latex_to_infix + parse_expr ---
    infix = _latex_to_infix(expr_str)
    try:
        return parse_expr(
            infix,
            local_dict=FREDHOLM_LOCAL_DICT,
            transformations=TRANSFORMATIONS,
        )
    except Exception:
        pass

    # --- Strategy 3: Additional preprocessing + parse_expr ---
    cleaned = _preprocess_for_sympy(expr_str)
    try:
        return parse_expr(
            cleaned,
            local_dict=FREDHOLM_LOCAL_DICT,
            transformations=TRANSFORMATIONS,
        )
    except Exception as exc:
        raise ParseError(f"Failed to parse expression: {expr_str}. Error: {exc}")


def _try_math_verify_parse(expr_str: str) -> Optional[sp.Expr]:
    """
    Attempt to parse *expr_str* with Math-Verify.

    Math-Verify expects LaTeX wrapped in dollar signs for reliable extraction.
    Returns ``None`` on failure so callers can fall back.
    """
    if not HAS_MATH_VERIFY:
        return None

    # Wrap in dollar signs if not already delimited
    wrapped = expr_str
    if not re.search(r"(?:^|\s)\$", expr_str):
        wrapped = f"${expr_str}$"

    try:
        parsed = parse(wrapped)
        if parsed and len(parsed) >= 1:
            candidate = parsed[0]
            if isinstance(candidate, sp.Basic):
                # Unwrap Eq(lhs, rhs) — happens when expression contains '='
                # (e.g. "expr1 = expr2" from chained equalities). Take lhs.
                if isinstance(candidate, sp.Equality):
                    candidate = candidate.lhs
                # Reject trivial parses that lost structure
                # (e.g. parse returning just '1' for a complex expression)
                if _is_nontrivial(candidate, expr_str):
                    return _normalize_symbols(candidate)
    except Exception as exc:
        logger.debug(f"Math-Verify parse failed for '{expr_str}': {exc}")

    return None


def _normalize_symbols(expr: sp.Basic) -> sp.Basic:
    """
    Replace Math-Verify's assumption-laden symbols with our canonical ones.

    Math-Verify creates symbols with ``real=True`` (and other assumptions).
    Our pipeline uses plain ``Symbol('x')`` with no assumptions.  Mixing the
    two causes ``simplify(a - b)`` to be non-zero even when both represent
    the same expression.
    """
    subs = {}
    for sym in expr.free_symbols:
        name = sym.name
        if name in FREDHOLM_LOCAL_DICT:
            canonical = FREDHOLM_LOCAL_DICT[name]
            if sym is not canonical:
                subs[sym] = canonical
    if subs:
        return expr.subs(subs)
    return expr


def _is_nontrivial(expr: sp.Basic, original: str) -> bool:
    """
    Heuristic check that *expr* is not a degenerate parse of *original*.

    Math-Verify sometimes collapses complex strings to a single number.
    We accept the result only when it looks structurally plausible.
    """
    # If the original contains a variable name and the parsed result is a plain number,
    # the parse likely failed silently.
    has_variable = bool(re.search(r"[a-zA-Z]", re.sub(r"\\[a-zA-Z]+", "", original)))
    if has_variable and expr.is_Number:
        return False
    return True


def extract_answer_from_response(
    response: str,
) -> Optional[tuple[sp.Expr, str]]:
    """
    Extract and parse a mathematical answer directly from a raw LLM response.

    Uses Math-Verify's ``parse()`` which can locate LaTeX expressions in
    natural-language text.  When the result is an ``Eq(u(x), rhs)`` object
    the RHS is returned (that's the solution we care about).

    Args:
        response: Raw LLM response text.

    Returns:
        ``(sympy_expr, raw_string)`` on success, or ``None`` if extraction
        fails or Math-Verify is unavailable.
    """
    if not HAS_MATH_VERIFY:
        return None

    try:
        parsed = parse(response)
        if not parsed or len(parsed) < 1:
            return None

        candidate = parsed[0]
        raw_str = str(parsed[1]) if len(parsed) >= 2 else str(candidate)

        if not isinstance(candidate, sp.Basic):
            return None

        # Unwrap Eq(u(x), rhs) → rhs
        if isinstance(candidate, sp.Equality):
            candidate = candidate.rhs
            # Update raw_str to reflect the RHS
            if "=" in raw_str:
                raw_str = raw_str.split("=", 1)[1].strip()

        # Reject trivial results
        if candidate.is_Number:
            return None

        # Reject results that are just products of single-letter symbols
        # (e.g. i*l*n*n*o*o*o*s*t*u from misparse of "No solution")
        if _is_scrambled_text(candidate):
            return None

        return _normalize_symbols(candidate), raw_str

    except Exception as exc:
        logger.debug(f"Math-Verify response extraction failed: {exc}")
        return None


def _is_scrambled_text(expr: sp.Basic) -> bool:
    """
    Detect expressions that are just products of single-letter symbols.

    Math-Verify sometimes parses English words like "No solution" as
    ``n*o*s*o*l*u*t*i*o*n``.  These are useless for evaluation.
    """
    if not isinstance(expr, sp.Mul):
        return False
    # If every factor is a single-letter symbol, it's likely scrambled text
    atoms = expr.as_ordered_factors()
    symbol_count = sum(1 for a in atoms if isinstance(a, sp.Symbol) and len(a.name) == 1)
    return symbol_count >= 4 and symbol_count == len(atoms)


def extract_solution_from_response(
    response: str,
) -> Optional[tuple[sp.Expr, str]]:
    """
    Multi-strategy Math-Verify extraction from a raw LLM response.

    Combines three strategies (most-specific first) to maximise extraction rate:

    1. **Targeted ``u(x)=`` line** – find the last line containing ``u(x) = ...``,
       parse the RHS with Math-Verify.
    2. **Structured ``SOLUTION:`` marker** – parse the content after a ``SOLUTION:``
       line with Math-Verify.
    3. **Full-response parse** – hand the entire response to Math-Verify's ``parse()``
       and look for ``Eq(u(x), rhs)`` or standalone expressions.

    Returns:
        ``(sympy_expr, raw_string)`` on success, or ``None``.
    """
    if not HAS_MATH_VERIFY:
        return None

    # Strategy 1 – targeted u(x) = ... line (most specific)
    result = _mv_targeted_ux(response)
    if result is not None:
        return result

    # Strategy 2 – SOLUTION: structured output
    result = _mv_structured_output(response)
    if result is not None:
        return result

    # Strategy 3 – full response parse (broadest)
    result = extract_answer_from_response(response)
    if result is not None:
        return result

    return None


def _clean_mv_content(text: str) -> str:
    """Strip LaTeX delimiters and trailing explanatory text for MV parsing."""
    # Strip leading \(
    text = re.sub(r"^\s*\\\(\s*", "", text)
    # Strip \) followed by optional punctuation and/or "where ..." text
    text = re.sub(
        r"\s*\\\)\s*[,;.]?\s*(?:where|with|for|if|when)\b.*$",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*\\\)\s*[,;.]*\s*$", "", text)
    # Strip trailing "where ..." without preceding \)
    text = re.sub(
        r"\s*[,;]\s+(?:where|with|for|if|when)\s+.*$",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s+(?:where|with|for|if|when)\s+.*$",
        "", text, flags=re.IGNORECASE,
    )
    # Strip trailing punctuation
    text = re.sub(r"[,;.]+$", "", text.strip())
    return text.strip()


def _mv_targeted_ux(
    response: str,
) -> Optional[tuple[sp.Expr, str]]:
    """Find the last ``u(x) = ...`` line and parse the RHS with Math-Verify."""
    if not HAS_MATH_VERIFY:
        return None

    lines = response.strip().split("\n")
    for line in reversed(lines):
        m = re.search(r"u\s*\(\s*x\s*\)\s*=\s*(.+)", line)
        if m:
            rhs = _clean_mv_content(m.group(1))
            if not rhs:
                continue
            mv_expr = _try_math_verify_parse(rhs)
            if mv_expr is not None:
                return mv_expr, rhs
    return None


def _mv_structured_output(
    response: str,
) -> Optional[tuple[sp.Expr, str]]:
    """Find a ``SOLUTION:`` line and parse its content with Math-Verify."""
    if not HAS_MATH_VERIFY:
        return None

    lines = response.strip().split("\n")
    for line in lines:
        stripped = line.strip()
        m = re.match(r"SOLUTION\s*[:=]\s*(.+)", stripped, re.IGNORECASE)
        if m:
            content = m.group(1).strip()
            if not content:
                continue
            # Strip leading \( and u(x) = prefix
            content = re.sub(r"^\s*\\\(\s*", "", content)
            ux_m = re.search(r"u\s*\(\s*x\s*\)\s*=\s*", content)
            if ux_m:
                content = content[ux_m.end() :].strip()
            # Clean delimiters and trailing text
            content = _clean_mv_content(content)
            if not content:
                continue
            mv_expr = _try_math_verify_parse(content)
            if mv_expr is not None:
                return mv_expr, content
    return None


def math_verify_compare(
    solution: sp.Expr,
    ground_truth: sp.Expr,
    timeout: int = 5,
) -> Optional[bool]:
    """
    Fast-path boolean comparison using Math-Verify's ``verify()``.

    Returns ``True``/``False`` when Math-Verify gives a confident answer,
    or ``None`` if Math-Verify is unavailable or errors out (so the caller
    should continue with the existing comparison pipeline).

    Args:
        solution: Predicted SymPy expression.
        ground_truth: Expected SymPy expression.
        timeout: Timeout in seconds for the verify call.

    Returns:
        ``True`` if equivalent, ``False`` if definitely not, ``None`` if uncertain.
    """
    if not HAS_MATH_VERIFY:
        return None

    try:
        result = verify(
            gold=ground_truth,
            target=solution,
            timeout_seconds=timeout,
        )
        return bool(result)
    except Exception as exc:
        logger.debug(f"Math-Verify verify failed: {exc}")
        return None
