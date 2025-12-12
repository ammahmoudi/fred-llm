#!/usr/bin/env python3
"""
Convert mathematical expressions to RPN (Reverse Polish Notation).

This script converts equations from infix/LaTeX notation to RPN format
for use with sequence-to-sequence models.

Usage:
    python scripts/convert_to_rpn.py --input data/processed/train.json --output data/processed/train_rpn.json
"""

import argparse
import json
from pathlib import Path

# TODO: Import from src once package is installed
# from src.data.format_converter import FormatConverter


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert equations to RPN format")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSON file with equations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for RPN equations",
    )
    parser.add_argument(
        "--source-format",
        type=str,
        default="infix",
        choices=["infix", "latex", "sympy"],
        help="Source expression format",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["kernel", "f", "solution"],
        help="Fields to convert",
    )
    return parser.parse_args()


def infix_to_rpn(expr: str) -> str:
    """
    Convert infix expression to RPN.

    This is a simplified implementation. The full version should use
    the FormatConverter class from src.data.format_converter.
    """
    # TODO: Use proper converter
    # converter = FormatConverter()
    # return converter.convert(expr, "infix", "rpn")

    # Placeholder: return tokenized version
    import re

    # Add spaces around operators
    expr = re.sub(r"([+\-*/^()])", r" \1 ", expr)
    tokens = expr.split()

    # Simple Shunting Yard algorithm
    output = []
    operators = []

    precedence = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}
    right_assoc = {"^"}

    for token in tokens:
        if token in precedence:
            while (
                operators
                and operators[-1] != "("
                and (
                    precedence.get(operators[-1], 0) > precedence[token]
                    or (
                        precedence.get(operators[-1], 0) == precedence[token]
                        and token not in right_assoc
                    )
                )
            ):
                output.append(operators.pop())
            operators.append(token)
        elif token == "(":
            operators.append(token)
        elif token == ")":
            while operators and operators[-1] != "(":
                output.append(operators.pop())
            if operators:
                operators.pop()  # Remove "("
        else:
            output.append(token)

    while operators:
        output.append(operators.pop())

    return " ".join(output)


def convert_equation(item: dict, fields: list[str]) -> dict:
    """Convert specified fields in an equation to RPN."""
    result = item.copy()

    for field in fields:
        if field in item and item[field]:
            try:
                result[f"{field}_rpn"] = infix_to_rpn(str(item[field]))
            except Exception as e:
                print(f"Warning: Failed to convert {field}: {e}")
                result[f"{field}_rpn"] = None

    return result


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load input data
    print(f"Loading data from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    print(f"Converting {len(data)} equations to RPN...")

    # Convert each equation
    converted = []
    for item in data:
        converted.append(convert_equation(item, args.fields))

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2)

    print(f"Saved converted data to {args.output}")


if __name__ == "__main__":
    main()
