"""
Prompt templates for Fredholm integral equation solving.

Supports multiple prompting styles:
- basic: Simple direct prompt
- chain-of-thought: Step-by-step reasoning
- few-shot: Include examples
- tool-assisted: Enable tool use for computation

Note: This is legacy code. For new code, use the OOP-based style classes in src/prompts/styles/
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.prompts.templates import FEW_SHOT_EXAMPLES
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplate:
    """Container for prompt template with metadata."""

    name: str
    system_prompt: str
    user_template: str
    style: str
    examples: Optional[list[dict[str, str]]] = None


# System prompts for different styles
SYSTEM_PROMPTS = {
    "basic": """You are an expert mathematician specializing in integral equations.
Given a Fredholm integral equation of the second kind, find the solution u(x).

The general form is: u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)

Provide your answer as a mathematical expression for u(x).""",
    "chain-of-thought": """You are an expert mathematician specializing in integral equations.
Your task is to solve Fredholm integral equations of the second kind.

The general form is: u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)

Approach each problem systematically:
1. Identify the kernel K(x, t), the known function f(x), and the parameter λ
2. Determine the type of kernel (separable, symmetric, etc.)
3. Choose an appropriate solution method
4. Apply the method step by step
5. Verify the solution satisfies the original equation
6. Present the final solution u(x)

Show your reasoning at each step.""",
    "few-shot": """You are an expert mathematician specializing in integral equations.
I will show you examples of solved Fredholm integral equations, then ask you to solve a new one.

The general form is: u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)""",
    "tool-assisted": """You are an expert mathematician with access to computational tools.
Your task is to solve Fredholm integral equations of the second kind.

The general form is: u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)

You can use the following tools:
- integrate(expr, var, a, b): Compute definite integral
- simplify(expr): Simplify mathematical expression
- solve(equation, var): Solve equation for variable
- series(expr, var, n): Expand in series

Show your work using these tools when helpful.""",
}


# User prompt templates
USER_TEMPLATES = {
    "basic": """Solve the following Fredholm integral equation:

u(x) - {lambda_val} * ∫_{a}^{b} {kernel} * u(t) dt = {f_x}

Domain: [{a}, {b}]

Provide the solution u(x).""",
    "chain-of-thought": """Solve the following Fredholm integral equation step by step:

u(x) - {lambda_val} * ∫_{a}^{b} {kernel} * u(t) dt = {f_x}

Domain: [{a}, {b}]

Please show your complete reasoning process.""",
    "few-shot": """Now solve this equation:

u(x) - {lambda_val} * ∫_{a}^{b} {kernel} * u(t) dt = {f_x}

Domain: [{a}, {b}]

Solution:""",
    "tool-assisted": """Solve the following Fredholm integral equation using available tools:

u(x) - {lambda_val} * ∫_{a}^{b} {kernel} * u(t) dt = {f_x}

Domain: [{a}, {b}]

Use tools as needed to find u(x).""",
}


def get_template(style: str) -> PromptTemplate:
    """
    Get prompt template for the specified style.

    Args:
        style: Prompting style (basic, chain-of-thought, few-shot, tool-assisted).

    Returns:
        PromptTemplate object.

    Raises:
        ValueError: If style is unknown.
    """
    if style not in SYSTEM_PROMPTS:
        raise ValueError(
            f"Unknown prompt style: {style}. Available: {list(SYSTEM_PROMPTS.keys())}"
        )

    examples = FEW_SHOT_EXAMPLES if style == "few-shot" else None

    return PromptTemplate(
        name=f"{style}_template",
        system_prompt=SYSTEM_PROMPTS[style],
        user_template=USER_TEMPLATES[style],
        style=style,
        examples=examples,
    )


def generate_prompt(
    equation: str | dict[str, str],
    style: str = "chain-of-thought",
    include_examples: bool = True,
    num_examples: int = 2,
) -> str:
    """
    Generate a complete prompt for the given equation.

    Args:
        equation: Either a string equation or dict with kernel, f_x, lambda_val, a, b.
        style: Prompting style.
        include_examples: Whether to include examples (for few-shot).
        num_examples: Number of examples to include.

    Returns:
        Complete formatted prompt string.
    """
    template = get_template(style)

    # Parse equation if string
    if isinstance(equation, str):
        # TODO: Parse equation string into components
        equation_dict = {"equation": equation}
    else:
        equation_dict = equation

    # Build the prompt
    prompt_parts = [template.system_prompt]

    # Add examples for few-shot
    if style == "few-shot" and include_examples and template.examples:
        prompt_parts.append("\n\nHere are some solved examples:\n")
        for i, example in enumerate(template.examples[:num_examples]):
            prompt_parts.append(f"\nExample {i + 1}:")
            prompt_parts.append(f"Problem: {example['problem']}")
            prompt_parts.append(f"{example['solution']}\n")

    # Add the user prompt
    if "equation" in equation_dict:
        prompt_parts.append(f"\n\n{equation_dict['equation']}")
    else:
        user_prompt = template.user_template.format(
            kernel=equation_dict.get("kernel", "K(x,t)"),
            f_x=equation_dict.get("f_x", "f(x)"),
            lambda_val=equation_dict.get("lambda_val", 1),
            a=equation_dict.get("a", 0),
            b=equation_dict.get("b", 1),
        )
        prompt_parts.append(f"\n\n{user_prompt}")

    return "\n".join(prompt_parts)


def load_custom_template(template_path: Path | str) -> PromptTemplate:
    """
    Load a custom prompt template from file.

    Args:
        template_path: Path to YAML or JSON template file.

    Returns:
        PromptTemplate object.
    """
    # TODO: Implement custom template loading
    logger.info(f"Loading custom template from {template_path}")
    raise NotImplementedError("Custom template loading not yet implemented")
