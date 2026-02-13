"""
Postprocessing utilities for LLM outputs.

Parses and validates mathematical expressions from LLM responses.
"""

import re
from typing import Any, Optional

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ParseError(Exception):
    """Exception raised when parsing fails."""

    pass


def parse_llm_output(
    response: str,
    extract_solution: bool = True,
    validate: bool = True,
) -> dict[str, Any]:
    """
    Parse LLM response and extract mathematical solution with metadata.

    Args:
        response: Raw LLM response text.
        extract_solution: Whether to extract the solution expression.
        validate: Whether to validate the parsed expression.

    Returns:
        Dictionary with parsed components:
        - solution_str: Solution as string
        - solution_sympy: Solution as SymPy expression
        - has_solution: Whether solution exists (bool or None)
        - solution_type: Type of solution (str or None)
        - reasoning: Extracted reasoning steps
        - confidence: Estimated confidence score
    """
    result = {
        "solution_str": None,
        "solution_sympy": None,
        "has_solution": None,
        "solution_type": None,
        "discrete_points": None,  # For discrete_points solution type
        "reasoning": None,
        "confidence": 0.0,
        "raw_response": response,
    }

    if not response or not response.strip():
        logger.warning("Empty response received")
        return result

    # Extract structured fields
    result["has_solution"] = _extract_has_solution(response)
    result["solution_type"] = _extract_solution_type(response)

    # Special handling for discrete_points solution type
    if result["solution_type"] == "discrete_points":
        points = extract_discrete_points(response)
        if points:
            result["discrete_points"] = points
            result["solution_str"] = str(points)
            result["confidence"] = 0.8
        else:
            logger.warning("discrete_points type specified but no points extracted")
            result["confidence"] = 0.3
        # Don't try to parse as SymPy expression - it's a point list
        return result

    # Extract solution from response
    if extract_solution:
        from src.llm.math_verify_adapter import HAS_MATH_VERIFY

        # Primary path: Math-Verify multi-strategy extraction (no manual regex)
        if HAS_MATH_VERIFY and validate:
            from src.llm.math_verify_adapter import extract_solution_from_response

            mv_result = extract_solution_from_response(response)
            if mv_result is not None:
                expr, raw_str = mv_result
                result["solution_str"] = raw_str
                result["solution_sympy"] = expr
                result["confidence"] = 0.8

        # Fallback: legacy regex pipeline (only when MV unavailable or failed)
        if result["solution_sympy"] is None:
            solution_str = _extract_solution(response)
            result["solution_str"] = solution_str

            if solution_str and validate:
                try:
                    result["solution_sympy"] = _parse_to_sympy(solution_str)
                    result["confidence"] = 0.7
                except Exception as e:
                    logger.warning(f"Failed to parse solution to SymPy: {e}")
                    result["confidence"] = 0.3

    # Infer has_solution=False when solution is empty and response says "no solution"
    if result["has_solution"] is None and not result["solution_str"]:
        no_solution_patterns = [
            r"\bno\s+solution\b",
            r"\bdoes\s+not\s+(?:have|exist|admit)\b",
            r"\bno\s+(?:unique\s+)?solution\s+exists?\b",
            r"\bcannot\s+be\s+solved\b",
            r"\bno\s+closed[- ]form\b",
            r"\bunsolvable\b",
            r"\binconsistent\b",
        ]
        for pat in no_solution_patterns:
            if re.search(pat, response, re.IGNORECASE):
                result["has_solution"] = False
                break

    # Extract reasoning
    result["reasoning"] = _extract_reasoning(response)

    return result


def _extract_solution(response: str) -> Optional[str]:
    """
    Extract the solution expression from LLM response.

    Looks for patterns like:
    - u(x) = ...
    - Solution: ...
    - Therefore, u(x) = ...
    - The solution is u(x) = ...
    - LaTeX inline: backslash-paren u(x) = ... backslash-paren
    """
    patterns = [
        # Prioritize structured output format (SOLUTION: u(x) = ...)
        r"^SOLUTION\s*:\s*u\s*\(\s*x\s*\)\s*=\s*(.+?)(?:\n|$)",
        r"^SOLUTION\s*:\s*(.+?)(?:\n|$)",
        # LaTeX delimited patterns - \( ... \) or $ ... $
        r"\\\(\s*u\s*\(\s*x\s*\)\s*=\s*(.+?)\s*\\\)",
        r"\\\(\s*u\s*\(\s*x\s*\)\s*=\s*(.+?)(?:\n|$)",
        r"\$\$?\s*u\s*\(\s*x\s*\)\s*=\s*(.+?)\s*\$\$?",
        # Then look for other patterns
        r"[Ss]olution[:\s]+u\s*\(\s*x\s*\)\s*=\s*(.+?)(?:\n|$)",
        r"[Tt]herefore[,:\s]+u\s*\(\s*x\s*\)\s*=\s*(.+?)(?:\n|$)",
        r"[Ff]inal\s+[Aa]nswer[:\s]+(.+?)(?:\n|$)",
        # Generic u(x) = pattern (last resort, may match reasoning)
        r"u\s*\(\s*x\s*\)\s*=\s*(.+?)(?:\n|$|\.(?:\s|$))",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            solution = match.group(1).strip()
            # Clean up the solution
            solution = _clean_expression(solution)
            if solution:
                return solution

    # Fallback: try to find any mathematical expression
    logger.debug("No explicit solution pattern found, using fallback extraction")
    return _fallback_extract(response)


def _clean_expression(expr: str) -> str:
    """Clean up a mathematical expression string."""
    # Remove LaTeX delimiters first (before punctuation stripping)
    expr = re.sub(r"\$+", "", expr.strip())
    expr = re.sub(r"^\s*\\\(\s*", "", expr)  # leading \(

    # Remove u(x) = prefix if present (we only want the RHS)
    expr = re.sub(r"^\s*u\s*\(\s*x\s*\)\s*=\s*", "", expr)

    # Remove trailing explanatory text after the expression
    # Handle combined \), where / \). / \), patterns in one pass
    expr = re.sub(r"\s*\\\)\s*[,;.]?\s*(?:where|with|for|if|when)\b.*$", "", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\s*\\\)\s*[,;.]*\s*$", "", expr)  # remaining \) at end
    # Common patterns: "where C = ...", "(where ...)", "\) where ...", etc.
    expr = re.sub(
        r"\s*\((?:where|with|for|if|when)\s+.*$", "", expr, flags=re.IGNORECASE
    )
    expr = re.sub(
        r"\s*[,;]\s+(?:where|with|for|if|when)\s+.*$", "", expr, flags=re.IGNORECASE
    )
    expr = re.sub(r"\s+(?:where|with|for|if|when)\s+.*$", "", expr, flags=re.IGNORECASE)

    # Remove trailing punctuation
    expr = re.sub(r"[,;.]+$", "", expr.strip())
    expr = re.sub(r"\\left|\\right", "", expr)  # \left and \right

    # Handle "No solution" responses - not a parseable expression
    if re.match(r"^\s*\\?\)?\s*no\s+solution\b", expr, re.IGNORECASE):
        return ""
    if re.match(r"^\s*no\s+solution\b", expr, re.IGNORECASE):
        return ""

    # Handle "requires numerical methods" type responses - mark as unparseable
    if re.search(
        r"requires?\s+(?:numerical|iterative|computational)", expr, re.IGNORECASE
    ):
        return ""
    if re.search(r"(?:cannot|can't|no\s+closed[- ]form)", expr, re.IGNORECASE):
        return ""
    if re.search(r"^\s*\[.*\]\s*$", expr):  # [text in brackets]
        return ""
    if re.search(
        r"^\s*\(.*(?:iterative|series|expansion|solution).*\)\s*$", expr, re.IGNORECASE
    ):
        return ""

    # Remove any remaining \) followed by explanation in parentheses
    expr = re.sub(r"\s*\\\)\s*\(.*$", "", expr)

    # Convert LaTeX to infix notation
    expr = _latex_to_infix(expr)

    # Normalize whitespace (but don't add spaces around operators)
    expr = " ".join(expr.split())

    return expr


def _latex_to_infix(expr: str) -> str:
    """
    Convert LaTeX mathematical notation to infix notation for SymPy parsing.

    Handles common LaTeX patterns:
    - \\sin, \\cos, \\tan, \\exp, \\log, \\ln, \\cosh, \\sinh, \\tanh
    - \\frac{a}{b} -> (a)/(b)
    - x^{n} or x^n -> x**n
    - \\sqrt{x} -> sqrt(x)
    - \\int_{a}^{b} -> (integral notation, simplified)
    - \\cdot -> *
    - \\times -> *
    """
    # Remove display math markers
    expr = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", "", expr)
    expr = re.sub(r"\\\[|\\\]", "", expr)  # \[ and \]
    expr = re.sub(r"\\[,;:!]", " ", expr)  # \, \; \: \! (LaTeX spaces)

    # Handle common functions FIRST (before other replacements)
    # Order matters: longer names first (cosh before cos)
    latex_functions = [
        (r"\\arcsinh", "asinh"),
        (r"\\arccosh", "acosh"),
        (r"\\arctanh", "atanh"),
        (r"\\arcsin", "asin"),
        (r"\\arccos", "acos"),
        (r"\\arctan", "atan"),
        (r"\\sinh", "sinh"),
        (r"\\cosh", "cosh"),
        (r"\\tanh", "tanh"),
        (r"\\sin", "sin"),
        (r"\\cos", "cos"),
        (r"\\tan", "tan"),
        (r"\\cot", "cot"),
        (r"\\sec", "sec"),
        (r"\\csc", "csc"),
        (r"\\exp", "exp"),
        (r"\\log", "log"),
        (r"\\ln", "log"),  # SymPy uses log for natural log
        (r"\\abs", "Abs"),
        (r"\\sqrt", "sqrt"),
    ]

    for latex_cmd, sympy_func in latex_functions:
        # Handle \func^{n}{arg} or \func^{n}\!(arg) -> func(arg)**n
        # LaTeX convention: \sin^{2}{x} means sin(x)^2
        expr = re.sub(
            latex_cmd + r"\^[\{]?(\d+)[\}]?\s*\\?[!;,]?\s*(?:\\(?:left|bigl?)\s*)?[\{(]([^})]+)[\})](?:\s*\\(?:right|bigr?))?\s*",
            sympy_func + r"(\2)**\1",
            expr,
        )
        # Handle \func{arg} -> func(arg)
        expr = re.sub(latex_cmd + r"\s*\{([^}]+)\}", sympy_func + r"(\1)", expr)
        # Handle \func(arg) -> func(arg) (already has parens)
        expr = re.sub(latex_cmd + r"\s*\(", sympy_func + r"(", expr)
        # Handle \func followed by space and variable -> func(var)
        expr = re.sub(latex_cmd + r"\s+([a-zA-Z])", sympy_func + r"(\1)", expr)
        # Handle \func at end of string or before operator
        expr = re.sub(latex_cmd + r"(?=[\s+\-*/^)]|$)", sympy_func, expr)

    # Handle integrals BEFORE exponent/subscript conversion to preserve _{a}^{b}
    # Convert to SymPy Integral() notation instead of removing
    # Definite integral: \int_{a}^{b} ... dt -> Integral(..., (t, a, b))
    def _replace_definite_integral(m: re.Match) -> str:
        lower = m.group(1)
        upper = m.group(2)
        integrand = m.group(3).strip().rstrip("\\,")
        var = m.group(4)
        return f"Integral({integrand}, ({var}, {lower}, {upper}))"

    expr = re.sub(
        r"\\int_\{([^}]*)\}\^\{([^}]*)\}\s*(.+?)\s*d([a-z])",
        _replace_definite_integral,
        expr,
    )

    # Indefinite integral: \int ... dt -> Integral(..., t)
    def _replace_indefinite_integral(m: re.Match) -> str:
        integrand = m.group(1).strip().rstrip("\\,")
        var = m.group(2)
        return f"Integral({integrand}, {var})"

    expr = re.sub(
        r"\\int\s+(.+?)\s*d([a-z])",
        _replace_indefinite_integral,
        expr,
    )

    # Handle fractions: \frac{num}{den} -> (num)/(den)
    def replace_frac(match: re.Match) -> str:
        content = match.group(0)
        brace_pattern = r"\\frac\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        m = re.match(brace_pattern, content)
        if m:
            num, den = m.groups()
            return f"(({num})/({den}))"
        return content

    # Iteratively replace fractions (handle nested)
    for _ in range(5):  # Max 5 levels of nesting
        new_expr = re.sub(
            r"\\frac\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
            replace_frac,
            expr,
        )
        if new_expr == expr:
            break
        expr = new_expr

    # Handle square roots with nth root: \sqrt[n]{x} -> x**(1/n)
    expr = re.sub(r"sqrt\[([^\]]+)\]\{([^}]+)\}", r"((\2)**(1/(\1)))", expr)

    # Handle exponents: x^{n} -> x**(n), x^(n) -> x**(n), x^n -> x**n
    expr = re.sub(r"\^\{([^}]+)\}", r"**(\1)", expr)
    expr = re.sub(r"\^\(([^)]+)\)", r"**(\1)", expr)  # ^(n) hybrid notation
    expr = re.sub(r"\^(\d+)", r"**\1", expr)
    expr = re.sub(r"\^([a-zA-Z])", r"**\1", expr)

    # Handle subscripts (usually just remove for solution parsing)
    expr = re.sub(r"_\{([^}]+)\}", r"_\1", expr)
    expr = re.sub(r"_(\d+)", r"\1", expr)

    # Handle absolute value: |x| -> Abs(x)
    expr = re.sub(r"\|([^|]+)\|", r"Abs(\1)", expr)

    # Handle multiplication symbols
    expr = re.sub(r"\\cdot", "*", expr)
    expr = re.sub(r"\\times", "*", expr)

    # Handle special constants
    expr = re.sub(r"\\pi", "pi", expr)
    expr = re.sub(r"\\infty", "oo", expr)

    # Handle e^{x} -> exp(x), including bare e**x without braces or parens
    expr = re.sub(r"\be\s*\*\*\s*\{([^}]+)\}", r"exp(\1)", expr)
    expr = re.sub(r"\be\s*\*\*\s*\(([^)]+)\)", r"exp(\1)", expr)
    expr = re.sub(r"\be\s*\*\*\s*([a-zA-Z0-9]+)", r"exp(\1)", expr)

    # Clean up residual integral/kernel patterns not caught by earlier conversion
    expr = re.sub(
        r"_-?[\d.]+\*\*\([^)]+\)\s*K\([^)]+\)\s*u\([^)]+\)\s*d[a-z]", "", expr
    )
    expr = re.sub(r"\s*K\(x,\s*t\)\s*u\(t\)\s*d[a-z]", "", expr)

    # Remove remaining LaTeX commands that might cause issues
    expr = re.sub(r"\\[a-zA-Z]+", "", expr)

    # Clean up braces - convert remaining {} to ()
    expr = expr.replace("{", "(").replace("}", ")")

    # Handle implicit multiplication in function arguments: exp(2 x) -> exp(2*x)
    expr = re.sub(r"(\d)\s+([a-zA-Z])", r"\1*\2", expr)

    # Clean up multiple operators or spaces
    expr = re.sub(r"\s+", " ", expr)
    expr = re.sub(r"\s*\+\s*\+\s*", " + ", expr)
    expr = re.sub(r"\s*-\s*-\s*", " + ", expr)
    expr = re.sub(r"\s*\+\s*$", "", expr)  # Remove trailing +
    expr = re.sub(r"^\s*\+\s*", "", expr)  # Remove leading +

    return expr.strip()


def _fallback_extract(response: str) -> Optional[str]:
    """Fallback extraction when no clear solution pattern is found."""
    lines = response.strip().split("\n")

    # First pass: look for u(x) = ... pattern (most reliable)
    for line in reversed(lines):
        line = line.strip()
        m = re.match(r".*u\s*\(\s*x\s*\)\s*=\s*(.+)", line)
        if m:
            cleaned = _clean_expression(m.group(1))
            if cleaned:
                return cleaned

    # Second pass: look for any line with = and u, but skip reasoning lines
    skip_prefixes = (
        "substitut",
        "let ",
        "if ",
        "when ",
        "assume",
        "where ",
        "since ",
        "because ",
        "note ",
        "recall ",
        "using ",
    )
    for line in reversed(lines):
        stripped = line.strip().lower()
        if any(stripped.startswith(p) for p in skip_prefixes):
            continue
        if "=" in line and "u" in line.lower():
            parts = line.split("=")
            if len(parts) >= 2:
                cleaned = _clean_expression(parts[-1])
                if cleaned:
                    return cleaned

    # Third pass: try Math-Verify extraction on the full response
    from src.llm.math_verify_adapter import HAS_MATH_VERIFY

    if HAS_MATH_VERIFY:
        try:
            from math_verify import parse as mv_parse

            parsed = mv_parse(response)
            if parsed and len(parsed) >= 2:
                raw_str = parsed[1]
                if isinstance(raw_str, str) and raw_str.strip():
                    cleaned = _clean_expression(raw_str)
                    if cleaned:
                        return cleaned
        except Exception:
            pass

    return None


def _parse_to_sympy(expr_str: str) -> sp.Expr:
    """
    Parse a string expression to SymPy.

    Delegates to the Math-Verify adapter which tries Math-Verify first,
    then falls back to the custom LaTeX-to-infix + parse_expr pipeline.

    Args:
        expr_str: Mathematical expression string.

    Returns:
        SymPy expression.

    Raises:
        ParseError: If parsing fails.
    """
    from src.llm.math_verify_adapter import parse_latex_to_sympy

    return parse_latex_to_sympy(expr_str)


def _preprocess_for_sympy(expr: str) -> str:
    """Preprocess expression string for SymPy parsing."""
    # Run LaTeX conversion again in case there are remnants
    expr = _latex_to_infix(expr)

    # Replace common patterns
    expr = expr.replace("^", "**")
    expr = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", expr)  # 2x -> 2*x
    expr = re.sub(r"([a-zA-Z])(\d)", r"\1*\2", expr)  # x2 -> x*2
    expr = re.sub(r"\)(\w)", r")*\1", expr)  # )x -> )*x

    # Add implicit multiplication before ( but NOT for function calls
    # e.g., x( -> x*( but exp( stays exp(
    _known_funcs = {
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "sqrt",
        "sinh",
        "cosh",
        "tanh",
        "asin",
        "acos",
        "atan",
        "asinh",
        "acosh",
        "atanh",
        "cot",
        "sec",
        "csc",
        "Abs",
        "Integral",
        "Sum",
    }

    def _implicit_mul_before_paren(m: re.Match) -> str:
        word = m.group(1)
        if word in _known_funcs:
            return m.group(0)  # Don't break function calls
        return f"{word}*("

    expr = re.sub(r"(\w+)\(", _implicit_mul_before_paren, expr)

    # Handle coefficient followed by function: 2sin(x) -> 2*sin(x)
    funcs = ["sin", "cos", "tan", "exp", "log", "sqrt", "sinh", "cosh", "tanh", "Abs"]
    for func in funcs:
        expr = re.sub(rf"(\d)({func})", rf"\1*\2", expr)
        expr = re.sub(rf"([a-zA-Z])({func})", rf"\1*\2", expr)

    # Remove any remaining backslashes
    expr = expr.replace("\\", "")

    # Handle e^x -> exp(x)
    expr = re.sub(r"\be\s*\*\*\s*\(([^)]+)\)", r"exp(\1)", expr)
    expr = re.sub(r"\be\s*\*\*\s*([a-zA-Z0-9]+)", r"exp(\1)", expr)

    return expr


def _extract_reasoning(response: str) -> Optional[str]:
    """Extract reasoning steps from the response."""
    # Look for step-by-step reasoning
    steps = []

    # Pattern for numbered steps
    step_pattern = (
        r"(?:Step\s*\d+[:\.]?\s*|^\d+[.)\]]\s*)(.+?)(?=(?:Step\s*\d+|^\d+[.)\]]|\Z))"
    )
    matches = re.findall(
        step_pattern, response, re.MULTILINE | re.DOTALL | re.IGNORECASE
    )

    if matches:
        steps = [m.strip() for m in matches if m.strip()]

    return "\n".join(steps) if steps else None


def _extract_has_solution(response: str) -> Optional[bool]:
    """
    Extract has_solution flag from structured output.

    Looks for patterns like:
    - HAS_SOLUTION: yes/no
    - Has solution: Yes/No
    """
    pattern = r"HAS[_\s]SOLUTION\s*[:\s]+\s*(yes|no|true|false)\b"
    match = re.search(pattern, response, re.IGNORECASE)

    if match:
        value = match.group(1).lower()
        return value in ("yes", "true")

    # If structured yes/no not found, look at the full HAS_SOLUTION line content
    hs_line_pattern = r"HAS[_\s]SOLUTION\s*[:\s]+\s*(.+?)(?:\n|$)"
    hs_match = re.search(hs_line_pattern, response, re.IGNORECASE)
    if hs_match:
        value = hs_match.group(1).strip().lower()
        if re.search(
            r"\bno\b|\bnot\b|\bfalse\b|\bn/?a\b|\bnone\b|\bdoes\s+not\b", value
        ):
            return False
        if re.search(r"\byes\b|\btrue\b|\bexists?\b", value):
            return True

    # Fallback: look for natural language indicators
    if re.search(r"\bno\s+solution\s+exists?\b", response, re.IGNORECASE):
        return False
    if re.search(r"\bsolution\s+does\s+not\s+exist\b", response, re.IGNORECASE):
        return False

    return None


def _extract_solution_type(response: str) -> Optional[str]:
    """
    Extract solution_type from structured output.

    Looks for patterns like:
    - SOLUTION_TYPE: exact_symbolic
    - Solution type: approx_coef
    """
    pattern = r"SOLUTION[_\s]TYPE\s*[:\s]+\s*([\w]+(?:[_\s-][\w]+)*)"
    match = re.search(pattern, response, re.IGNORECASE)

    if match:
        # Normalize separators: spaces and dashes to underscores
        value = re.sub(r"[\s-]+", "_", match.group(1).strip()).lower()
        # Validate against known types
        valid_types = {
            "exact_symbolic",
            "approx_coef",
            "discrete_points",
            "series",
            "family",
            "regularized",
            "none",
        }
        if value in valid_types:
            return value

    return None


def extract_discrete_points(response: str) -> Optional[list[tuple[float, float]]]:
    """
    Extract discrete point list from LLM response.

    Looks for format: [(x1, y1), (x2, y2), ...]
    or variations like: [(0.0, 1.2), (0.5, 3.4), (1.0, 2.1)]

    Args:
        response: LLM response text.

    Returns:
        List of (x, y) tuples if found and valid, None otherwise.
    """
    # Pattern to match list of tuples with floating point or scientific notation
    # Matches: [(x1, y1), (x2, y2), ...] with optional whitespace
    pattern = r"\[\s*\(\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\)(?:\s*,\s*\(\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\))*\s*\]"

    # First try to find the SOLUTION: line with discrete points
    solution_line_pattern = r"SOLUTION\s*:\s*(\[.*?\])(?:\n|$)"
    solution_match = re.search(
        solution_line_pattern, response, re.IGNORECASE | re.MULTILINE
    )

    if solution_match:
        points_str = solution_match.group(1)
    else:
        # Try to find any list of tuples in the response
        list_pattern = r"(\[\s*\([\d.eE+-]+\s*,\s*[\d.eE+-]+\s*\)(?:\s*,\s*\([\d.eE+-]+\s*,\s*[\d.eE+-]+\s*\))*\s*\])"
        list_match = re.search(list_pattern, response)
        if list_match:
            points_str = list_match.group(1)
        else:
            return None

    # Parse individual tuples from the matched string
    tuple_pattern = r"\(\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\)"
    tuple_matches = re.findall(tuple_pattern, points_str)

    if not tuple_matches:
        return None

    # Convert to list of float tuples
    try:
        points = [(float(x), float(y)) for x, y in tuple_matches]

        # Validation: at least 2 points, reasonable values
        if len(points) < 2:
            logger.warning(f"Too few discrete points: {len(points)} (need >= 2)")
            return None

        # Check for reasonable x-values (should be ordered or at least finite)
        if not all(abs(x) < 1e10 and abs(y) < 1e10 for x, y in points):
            logger.warning("Discrete points contain unreasonably large values")
            return None

        return points

    except ValueError as e:
        logger.warning(f"Failed to parse discrete points: {e}")
        return None


def extract_latex(response: str) -> list[str]:
    """
    Extract LaTeX expressions from response.

    Args:
        response: LLM response text.

    Returns:
        List of LaTeX expression strings.
    """
    patterns = [
        r"\$\$(.+?)\$\$",  # Display math
        r"\$(.+?)\$",  # Inline math
        r"\\begin\{equation\}(.+?)\\end\{equation\}",
        r"\\begin\{align\}(.+?)\\end\{align\}",
    ]

    expressions = []
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        expressions.extend(matches)

    return [e.strip() for e in expressions if e.strip()]


def sympy_to_latex(expr: sp.Expr) -> str:
    """Convert SymPy expression to LaTeX string."""
    return sp.latex(expr)


def sympy_to_python(expr: sp.Expr) -> str:
    """Convert SymPy expression to Python code string."""
    from sympy.printing.pycode import pycode

    return pycode(expr)
