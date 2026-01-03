"""
Auto-detection utilities for dataset format and structure.

Detects format (infix/latex/rpn) by analyzing the actual equations.
"""

import re
from pathlib import Path
from typing import Literal

import pandas as pd

FormatType = Literal["infix", "latex", "rpn"]


def detect_format_from_expression(expression: str) -> FormatType:
    """
    Detect the format of a mathematical expression.

    Args:
        expression: Mathematical expression string

    Returns:
        Detected format: 'infix', 'latex', or 'rpn'
    """
    if not expression or not isinstance(expression, str):
        return "infix"  # default

    # RPN indicators: operators after operands, space-separated tokens
    # e.g., "x 2 ^ x sin +" or "x t * exp"
    # Check for pattern: tokens with operators not between operands
    tokens = expression.split()
    if len(tokens) >= 3:
        # RPN characteristics:
        # 1. Space-separated tokens
        # 2. Operators appear after operands (postfix)
        # 3. Common single-char operators at end: +, -, *, /, ^
        rpn_operators = {"+", "-", "*", "/", "^"}
        # Check if expression ends with a single-char operator
        if tokens[-1] in rpn_operators:
            # Also check it doesn't look like infix (no operators between operands)
            if not re.search(r"[a-zA-Z0-9]\s*[\+\-\*/\^]\s*[a-zA-Z0-9]", expression):
                return "rpn"
        # Check for function names as operators in postfix position
        rpn_functions = {"exp", "sin", "cos", "tan", "log", "sqrt", "neg"}
        if any(tok in rpn_functions for tok in tokens[1:]):  # Functions not at start
            # Check it doesn't look like infix function calls
            if not re.search(r"\w+\([^)]+\)", expression):
                return "rpn"

    # LaTeX indicators: \sin, \cos, \exp, ^{...}, \frac{...}{...}
    latex_indicators = [
        r"\\sin",
        r"\\cos",
        r"\\tan",
        r"\\exp",
        r"\\log",
        r"\\sqrt",
        r"\\frac",
        r"\^\{",
        r"_\{",
    ]
    if any(re.search(indicator, expression) for indicator in latex_indicators):
        return "latex"

    # Infix indicators: standard operators with operands
    # e.g., "x**2 + sin(x)" or "exp(-x)*cos(t)"
    infix_indicators = [
        r"\*\*",  # Power operator
        r"[a-zA-Z0-9]\s*[+\-*/]\s*[a-zA-Z0-9]",  # Binary operators
        r"\w+\([^)]+\)",  # Function calls like sin(x)
    ]
    if any(re.search(indicator, expression) for indicator in infix_indicators):
        return "infix"

    # Default to infix if unsure
    return "infix"


def detect_format_from_file(file_path: Path, sample_size: int = 10) -> FormatType:
    """
    Detect format by sampling expressions from a CSV file.

    Args:
        file_path: Path to CSV file
        sample_size: Number of rows to sample for detection

    Returns:
        Detected format
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read sample
    df = pd.read_csv(file_path, nrows=sample_size)

    # Check columns that typically contain expressions
    expression_columns = ["u", "f", "kernel"]

    format_votes = {"infix": 0, "latex": 0, "rpn": 0}

    for col in expression_columns:
        if col in df.columns:
            for expr in df[col].dropna():
                detected = detect_format_from_expression(str(expr))
                format_votes[detected] += 1

    # Return format with most votes
    detected_format = max(format_votes, key=format_votes.get)

    return detected_format


def detect_format_from_filename(file_path: Path) -> FormatType | None:
    """
    Try to detect format from filename pattern.

    Args:
        file_path: Path to file

    Returns:
        Detected format or None if not detected from filename
    """
    filename = file_path.stem.lower()  # e.g., "train_infix"

    if "infix" in filename:
        return "infix"
    elif "latex" in filename:
        return "latex"
    elif "rpn" in filename:
        return "rpn"

    return None


def auto_detect_format(file_path: Path, prefer_filename: bool = True) -> FormatType:
    """
    Auto-detect format using multiple strategies.

    Strategy:
    1. Try filename pattern (fast)
    2. Analyze file content (accurate)

    Args:
        file_path: Path to dataset file
        prefer_filename: If True, trust filename pattern when found

    Returns:
        Detected format
    """
    # Strategy 1: Filename pattern (fast and usually reliable)
    if prefer_filename:
        filename_format = detect_format_from_filename(file_path)
        if filename_format:
            return filename_format

    # Strategy 2: Content analysis (accurate but slower)
    return detect_format_from_file(file_path)


def validate_format_consistency(
    train_path: Path, val_path: Path | None = None, test_path: Path | None = None
) -> tuple[FormatType, dict[str, FormatType]]:
    """
    Check if all splits use the same format.

    Args:
        train_path: Path to training data
        val_path: Path to validation data (optional)
        test_path: Path to test data (optional)

    Returns:
        Tuple of (detected_format, individual_formats_dict)

    Raises:
        ValueError: If splits have inconsistent formats
    """
    formats = {}

    formats["train"] = auto_detect_format(train_path)

    if val_path and val_path.exists():
        formats["val"] = auto_detect_format(val_path)

    if test_path and test_path.exists():
        formats["test"] = auto_detect_format(test_path)

    # Check consistency
    unique_formats = set(formats.values())

    if len(unique_formats) > 1:
        raise ValueError(
            f"Inconsistent formats detected across splits: {formats}. "
            f"All splits should use the same format."
        )

    return formats["train"], formats


if __name__ == "__main__":
    # Test detection
    test_cases = [
        ("x**2 + sin(x)", "infix"),
        ("x^2 + \\sin(x)", "latex"),
        ("x 2 ^ x sin +", "rpn"),
        ("exp(-x)*cos(t)", "infix"),
        ("e^{-x}\\cos(t)", "latex"),
        ("x neg exp t cos *", "rpn"),
    ]

    print("Testing format detection:")
    for expr, expected in test_cases:
        detected = detect_format_from_expression(expr)
        status = "✓" if detected == expected else "✗"
        print(f"{status} '{expr}' → {detected} (expected: {expected})")
