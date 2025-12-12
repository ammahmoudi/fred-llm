#!/usr/bin/env python3
"""
Generate prompts for LLM inference.

Creates formatted prompts from equation data using various prompting styles.

Usage:
    python scripts/generate_prompts.py --input data/processed/test.json --output data/prompts/
    python scripts/generate_prompts.py --input data/processed/test.json --style chain-of-thought
"""

import argparse
import json
from pathlib import Path

# TODO: Import from src once package is installed
# from src.llm.prompt_templates import generate_prompt, get_template


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate prompts for equations")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSON file with equations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prompts/"),
        help="Output directory for prompts",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="chain-of-thought",
        choices=["basic", "chain-of-thought", "few-shot", "tool-assisted"],
        help="Prompting style",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jsonl",
        choices=["jsonl", "json", "txt"],
        help="Output format",
    )
    parser.add_argument(
        "--include-solution",
        action="store_true",
        help="Include solution in output (for evaluation)",
    )
    return parser.parse_args()


def create_prompt(equation: dict, style: str) -> str:
    """
    Create a prompt for the given equation.
    
    TODO: Use the full prompt_templates module.
    """
    kernel = equation.get("kernel", "K(x,t)")
    f = equation.get("f", "f(x)")
    lambda_val = equation.get("lambda_val", 1.0)
    domain = equation.get("domain", [0, 1])
    
    if isinstance(domain, dict):
        a, b = domain.get("a", 0), domain.get("b", 1)
    else:
        a, b = domain[0], domain[1]
    
    if style == "basic":
        prompt = f"""Solve the following Fredholm integral equation of the second kind:

u(x) - {lambda_val} * ∫_{a}^{b} {kernel} * u(t) dt = {f}

Provide the solution u(x)."""

    elif style == "chain-of-thought":
        prompt = f"""Solve the following Fredholm integral equation of the second kind step by step.

Equation: u(x) - {lambda_val} * ∫_{a}^{b} {kernel} * u(t) dt = {f}

Please:
1. Identify the kernel type and structure
2. Choose an appropriate solution method
3. Apply the method step by step
4. Verify your solution
5. State the final answer for u(x)"""

    elif style == "few-shot":
        prompt = f"""I'll show you how to solve Fredholm integral equations, then ask you to solve one.

Example 1:
Equation: u(x) - ∫_0^1 x*t * u(t) dt = x
Solution: The kernel K(x,t) = x*t is separable. Let c = ∫_0^1 t*u(t) dt.
Then u(x) = x + x*c = x(1+c). Solving for c gives c = 1/2.
Answer: u(x) = 3x/2

Now solve:
u(x) - {lambda_val} * ∫_{a}^{b} {kernel} * u(t) dt = {f}

Solution:"""

    elif style == "tool-assisted":
        prompt = f"""Solve the following Fredholm integral equation using available mathematical tools.

u(x) - {lambda_val} * ∫_{a}^{b} {kernel} * u(t) dt = {f}

Available tools:
- integrate(expr, var, a, b): Compute definite integral
- simplify(expr): Simplify expression
- solve(equation, var): Solve for variable

Show your work and state the final u(x)."""

    else:
        prompt = f"Solve: u(x) - {lambda_val} * ∫_{a}^{b} {kernel} * u(t) dt = {f}"
    
    return prompt


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Load input data
    print(f"Loading equations from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    
    print(f"Generating {args.style} prompts for {len(data)} equations...")
    
    # Generate prompts
    prompts = []
    for item in data:
        prompt_text = create_prompt(item, args.style)
        
        prompt_entry = {
            "id": item.get("id", f"eq_{len(prompts)}"),
            "prompt": prompt_text,
            "style": args.style,
        }
        
        if args.include_solution and "solution" in item:
            prompt_entry["expected_solution"] = item["solution"]
        
        prompts.append(prompt_entry)
    
    # Save output
    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / f"prompts_{args.style}.{args.format}"
    
    if args.format == "jsonl":
        with open(output_file, "w", encoding="utf-8") as f:
            for p in prompts:
                f.write(json.dumps(p) + "\n")
    elif args.format == "json":
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2)
    else:  # txt
        with open(output_file, "w", encoding="utf-8") as f:
            for p in prompts:
                f.write(f"=== {p['id']} ===\n")
                f.write(p["prompt"])
                f.write("\n\n")
    
    print(f"Saved {len(prompts)} prompts to {output_file}")


if __name__ == "__main__":
    main()
