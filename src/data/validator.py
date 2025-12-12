"""
Validator for Fredholm integral equation data.

Validates equation structure, mathematical correctness, and data integrity.
"""

from typing import Any

import sympy as sp

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Exception raised for validation failures."""

    pass


def validate_equation(
    equation: dict[str, Any],
    strict: bool = False,
) -> dict[str, Any]:
    """
    Validate a Fredholm integral equation entry.

    Args:
        equation: Equation dictionary to validate.
        strict: If True, raise on validation failures.

    Returns:
        Validation result dictionary.

    Raises:
        ValidationError: If strict=True and validation fails.
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
    }

    # Check required fields
    required_fields = ["kernel", "f", "lambda_val"]
    for field in required_fields:
        if field not in equation:
            result["errors"].append(f"Missing required field: {field}")
            result["valid"] = False

    # Check optional but recommended fields
    recommended_fields = ["domain", "solution"]
    for field in recommended_fields:
        if field not in equation:
            result["warnings"].append(f"Missing recommended field: {field}")

    # Validate kernel expression
    if "kernel" in equation:
        kernel_valid, kernel_msg = _validate_expression(equation["kernel"], ["x", "t"])
        if not kernel_valid:
            result["errors"].append(f"Invalid kernel: {kernel_msg}")
            result["valid"] = False

    # Validate f(x) expression
    if "f" in equation:
        f_valid, f_msg = _validate_expression(equation["f"], ["x"])
        if not f_valid:
            result["errors"].append(f"Invalid f(x): {f_msg}")
            result["valid"] = False

    # Validate lambda value
    if "lambda_val" in equation:
        try:
            lambda_val = float(equation["lambda_val"])
            if lambda_val == 0:
                result["warnings"].append("lambda_val is 0 (trivial case)")
        except (ValueError, TypeError):
            result["errors"].append("lambda_val must be a number")
            result["valid"] = False

    # Validate domain
    if "domain" in equation:
        domain = equation["domain"]
        if isinstance(domain, (list, tuple)) and len(domain) == 2:
            a, b = domain
            if a >= b:
                result["errors"].append(
                    f"Invalid domain: a ({a}) must be less than b ({b})"
                )
                result["valid"] = False
        elif isinstance(domain, dict) and "a" in domain and "b" in domain:
            if domain["a"] >= domain["b"]:
                result["errors"].append("Invalid domain: a must be less than b")
                result["valid"] = False
        else:
            result["errors"].append("Invalid domain format")
            result["valid"] = False

    # Validate solution if present
    if "solution" in equation and equation["solution"]:
        sol_valid, sol_msg = _validate_expression(equation["solution"], ["x"])
        if not sol_valid:
            result["warnings"].append(f"Invalid solution format: {sol_msg}")

    if strict and not result["valid"]:
        raise ValidationError("; ".join(result["errors"]))

    return result


def _validate_expression(
    expr: str | sp.Expr,
    expected_vars: list[str],
) -> tuple[bool, str]:
    """
    Validate a mathematical expression.

    Args:
        expr: Expression to validate.
        expected_vars: Expected free variables.

    Returns:
        Tuple of (is_valid, message).
    """
    try:
        if isinstance(expr, str):
            sympy_expr = sp.sympify(expr)
        else:
            sympy_expr = expr

        # Check for undefined operations
        if sympy_expr.has(sp.zoo):  # Complex infinity
            return False, "Expression contains undefined value (zoo)"
        if sympy_expr.has(sp.nan):
            return False, "Expression contains NaN"

        # Check free symbols
        free_symbols = {str(s) for s in sympy_expr.free_symbols}
        expected_set = set(expected_vars)

        extra_vars = free_symbols - expected_set
        if extra_vars:
            # This is a warning, not an error
            pass

        return True, "Valid"

    except Exception as e:
        return False, str(e)


def validate_dataset(
    data: list[dict[str, Any]],
    strict: bool = False,
) -> dict[str, Any]:
    """
    Validate an entire dataset.

    Args:
        data: List of equation dictionaries.
        strict: If True, raise on first validation failure.

    Returns:
        Validation summary dictionary.
    """
    summary = {
        "total": len(data),
        "valid": 0,
        "invalid": 0,
        "errors": [],
        "warnings": [],
    }

    for i, equation in enumerate(data):
        result = validate_equation(equation, strict=strict)

        if result["valid"]:
            summary["valid"] += 1
        else:
            summary["invalid"] += 1
            summary["errors"].append(
                {
                    "index": i,
                    "errors": result["errors"],
                }
            )

        if result["warnings"]:
            summary["warnings"].append(
                {
                    "index": i,
                    "warnings": result["warnings"],
                }
            )

    logger.info(
        f"Validation complete: {summary['valid']}/{summary['total']} valid, "
        f"{summary['invalid']} invalid"
    )

    return summary


class DataValidator:
    """Configurable data validator."""

    def __init__(
        self,
        strict: bool = False,
        custom_rules: list[callable] | None = None,
    ) -> None:
        """
        Initialize the validator.

        Args:
            strict: If True, raise on validation failures.
            custom_rules: Additional validation functions.
        """
        self.strict = strict
        self.custom_rules = custom_rules or []

    def validate_one(self, equation: dict[str, Any]) -> dict[str, Any]:
        """Validate a single equation."""
        result = validate_equation(equation, strict=self.strict)

        # Apply custom rules
        for rule in self.custom_rules:
            try:
                rule_result = rule(equation)
                if isinstance(rule_result, dict):
                    result["errors"].extend(rule_result.get("errors", []))
                    result["warnings"].extend(rule_result.get("warnings", []))
                    if rule_result.get("errors"):
                        result["valid"] = False
            except Exception as e:
                result["warnings"].append(f"Custom rule failed: {e}")

        return result

    def validate_all(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate all equations in dataset."""
        return validate_dataset(data, strict=self.strict)
