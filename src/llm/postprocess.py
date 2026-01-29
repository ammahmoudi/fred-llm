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

    # Extract solution from response
    if extract_solution:
        solution_str = _extract_solution(response)
        result["solution_str"] = solution_str

        if solution_str and validate:
            try:
                result["solution_sympy"] = _parse_to_sympy(solution_str)
                result["confidence"] = 0.8  # Base confidence if parsing succeeds
            except Exception as e:
                logger.warning(f"Failed to parse solution to SymPy: {e}")
                result["confidence"] = 0.3

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
    # Remove trailing punctuation
    expr = re.sub(r"[,;.]+$", "", expr.strip())

    # Remove LaTeX delimiters
    expr = re.sub(r"\$+", "", expr)
    expr = re.sub(r"^\s*\\\(\s*|\s*\\\)\s*$", "", expr)  # \( and \)
    expr = re.sub(r"\\left|\\right", "", expr)  # \left and \right

    # Remove u(x) = prefix if present (we only want the RHS)
    expr = re.sub(r"^\s*u\s*\(\s*x\s*\)\s*=\s*", "", expr)

    # Remove trailing explanatory text after the expression
    # Common patterns: "where C = ...", "(where ...)", "\) where ...", etc.
    expr = re.sub(r"\s*\\\)\s*(?:where|with|for|if|when).*$", "", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\s*\((?:where|with|for|if|when)\s+.*$", "", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\s+(?:where|with|for|if|when)\s+.*$", "", expr, flags=re.IGNORECASE)

    # Remove any remaining \) at end of expression
    expr = re.sub(r"\s*\\\)\s*\.?\s*$", "", expr)

    # Handle "requires numerical methods" type responses - mark as unparseable
    if re.search(r"requires?\s+(?:numerical|iterative|computational)", expr, re.IGNORECASE):
        return ""
    if re.search(r"(?:cannot|can't|no\s+closed[- ]form)", expr, re.IGNORECASE):
        return ""
    if re.search(r"^\s*\[.*\]\s*$", expr):  # [text in brackets]
        return ""
    if re.search(r"^\s*\(.*(?:iterative|series|expansion|solution).*\)\s*$", expr, re.IGNORECASE):
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
    expr = re.sub(r"\\,", " ", expr)  # \, (thin space in LaTeX)

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
        # Handle \func{arg} -> func(arg)
        expr = re.sub(latex_cmd + r"\s*\{([^}]+)\}", sympy_func + r"(\1)", expr)
        # Handle \func(arg) -> func(arg) (already has parens)
        expr = re.sub(latex_cmd + r"\s*\(", sympy_func + r"(", expr)
        # Handle \func followed by space and variable -> func(var)
        expr = re.sub(latex_cmd + r"\s+([a-zA-Z])", sympy_func + r"(\1)", expr)
        # Handle \func at end of string or before operator
        expr = re.sub(latex_cmd + r"(?=[\s+\-*/^)]|$)", sympy_func, expr)

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

    # Handle exponents: x^{n} -> x**(n), x^n -> x**n (single char)
    expr = re.sub(r"\^\{([^}]+)\}", r"**(\1)", expr)
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

    # Handle e^{x} -> exp(x)
    expr = re.sub(r"\be\s*\*\*\s*\{([^}]+)\}", r"exp(\1)", expr)
    expr = re.sub(r"\be\s*\*\*\s*\(([^)]+)\)", r"exp(\1)", expr)

    # Handle integrals - remove them as they can't be easily parsed
    # Match \int_{a}^{b} ... dt patterns and remove the whole integral
    expr = re.sub(r"\\int_\{[^}]*\}\^\{[^}]*\}[^,\n]*\\?,?\s*d[a-z]", "", expr)
    expr = re.sub(r"\\int[^,\n]*\\?,?\s*d[a-z]", "", expr)
    # Clean up standalone Integral notations and patterns like _a**(b) K(x,t) u(t) dt
    expr = re.sub(r"Integral_[^\s]*\s*", "", expr)
    expr = re.sub(r"Integral[^\s]*\s*", "", expr)
    expr = re.sub(r"_-?[\d.]+\*\*\([^)]+\)\s*K\([^)]+\)\s*u\([^)]+\)\s*d[a-z]", "", expr)
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
    # Look for the last mathematical expression in the response
    lines = response.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if "=" in line and "u" in line.lower():
            # Try to extract the right side of the equation
            parts = line.split("=")
            if len(parts) >= 2:
                return _clean_expression(parts[-1])
    return None


def _parse_to_sympy(expr_str: str) -> sp.Expr:
    """
    Parse a string expression to SymPy.

    Args:
        expr_str: Mathematical expression string.

    Returns:
        SymPy expression.

    Raises:
        ParseError: If parsing fails.
    """
    # Define common symbols
    x, t = sp.symbols("x t")
    local_dict = {"x": x, "t": t, "e": sp.E, "pi": sp.pi}

    # Transformations for parsing
    transformations = standard_transformations + (implicit_multiplication,)

    try:
        # Try standard parsing
        parsed = parse_expr(
            expr_str,
            local_dict=local_dict,
            transformations=transformations,
        )
        return parsed
    except Exception as e:
        # Try with additional preprocessing
        cleaned = _preprocess_for_sympy(expr_str)
        try:
            parsed = parse_expr(
                cleaned,
                local_dict=local_dict,
                transformations=transformations,
            )
            return parsed
        except Exception as e2:
            raise ParseError(f"Failed to parse expression: {expr_str}. Error: {e2}")


def _preprocess_for_sympy(expr: str) -> str:
    """Preprocess expression string for SymPy parsing."""
    # Run LaTeX conversion again in case there are remnants
    expr = _latex_to_infix(expr)

    # Replace common patterns
    expr = expr.replace("^", "**")
    expr = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", expr)  # 2x -> 2*x
    expr = re.sub(r"([a-zA-Z])(\d)", r"\1*\2", expr)  # x2 -> x*2
    expr = re.sub(r"\)(\w)", r")*\1", expr)  # )x -> )*x
    expr = re.sub(r"(\w)\(", r"\1*(", expr)  # x( -> x*(

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
    pattern = r"HAS[_\s]SOLUTION\s*[:\s]+\s*(yes|no|true|false)"
    match = re.search(pattern, response, re.IGNORECASE)

    if match:
        value = match.group(1).lower()
        return value in ("yes", "true")

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
    pattern = r"SOLUTION[_\s]TYPE\s*[:\s]+\s*(\w+)"
    match = re.search(pattern, response, re.IGNORECASE)

    if match:
        value = match.group(1).lower()
        # Validate against known types
        valid_types = {
            "exact_symbolic",
            "exact_coef",
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
