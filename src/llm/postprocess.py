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
    Parse LLM response and extract mathematical solution.

    Args:
        response: Raw LLM response text.
        extract_solution: Whether to extract the solution expression.
        validate: Whether to validate the parsed expression.

    Returns:
        Dictionary with parsed components:
        - solution_str: Solution as string
        - solution_sympy: Solution as SymPy expression
        - reasoning: Extracted reasoning steps
        - confidence: Estimated confidence score
    """
    result = {
        "solution_str": None,
        "solution_sympy": None,
        "reasoning": None,
        "confidence": 0.0,
        "raw_response": response,
    }

    if not response or not response.strip():
        logger.warning("Empty response received")
        return result

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
    """
    patterns = [
        r"u\s*\(\s*x\s*\)\s*=\s*(.+?)(?:\n|$|\.(?:\s|$))",
        r"[Ss]olution[:\s]+u\s*\(\s*x\s*\)\s*=\s*(.+?)(?:\n|$)",
        r"[Tt]herefore[,:\s]+u\s*\(\s*x\s*\)\s*=\s*(.+?)(?:\n|$)",
        r"[Ff]inal\s+[Aa]nswer[:\s]+(.+?)(?:\n|$)",
        r"\$\$?\s*u\s*\(\s*x\s*\)\s*=\s*(.+?)\s*\$\$?",  # LaTeX
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
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

    # Remove LaTeX markers
    expr = re.sub(r"\$+", "", expr)
    expr = re.sub(r"\\[a-zA-Z]+\{([^}]+)\}", r"\1", expr)  # Simple LaTeX commands

    # Normalize whitespace
    expr = " ".join(expr.split())

    return expr


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
    # Replace common patterns
    expr = expr.replace("^", "**")
    expr = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", expr)  # 2x -> 2*x
    expr = re.sub(r"([a-zA-Z])(\d)", r"\1*\2", expr)  # x2 -> x*2
    expr = re.sub(r"\)(\w)", r")*\1", expr)  # )x -> )*x
    expr = re.sub(r"(\w)\(", r"\1*(", expr)  # x( -> x*(

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
    return pycode(expr)
    return pycode(expr)


def sympy_to_python(expr: sp.Expr) -> str:
    """Convert SymPy expression to Python code string."""
    from sympy.printing.pycode import pycode

    return pycode(expr)
