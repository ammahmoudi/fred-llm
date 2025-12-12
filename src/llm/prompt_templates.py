"""
Prompt templates for Fredholm integral equation solving.

Supports multiple prompting styles:
- basic: Simple direct prompt
- chain-of-thought: Step-by-step reasoning
- few-shot: Include examples
- tool-assisted: Enable tool use for computation
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


# Example problems for few-shot prompting
FEW_SHOT_EXAMPLES = [
    {
        "problem": "u(x) - ∫_0^1 x*t * u(t) dt = x",
        "solution": """Let me solve this step by step.

The kernel K(x,t) = x*t is separable, so I can write:
∫_0^1 x*t * u(t) dt = x * ∫_0^1 t * u(t) dt = x * c

where c = ∫_0^1 t * u(t) dt is a constant.

Substituting back: u(x) = x + x*c = x(1 + c)

To find c, multiply both sides by t and integrate:
c = ∫_0^1 t * x(1+c) dt = (1+c) * ∫_0^1 t * x dt

Since this involves both x and t, let's reconsider...
Actually: c = ∫_0^1 t * u(t) dt = ∫_0^1 t * t(1+c) dt = (1+c) * ∫_0^1 t² dt = (1+c)/3

Solving: c = (1+c)/3 → 3c = 1 + c → 2c = 1 → c = 1/2

Therefore: u(x) = x(1 + 1/2) = 3x/2""",
    },
    {
        "problem": "u(x) - ∫_0^1 e^(x+t) * u(t) dt = 1",
        "solution": """The kernel K(x,t) = e^(x+t) = e^x * e^t is separable.

Let c = ∫_0^1 e^t * u(t) dt (a constant)

Then: u(x) = 1 + e^x * c

Substituting to find c:
c = ∫_0^1 e^t * (1 + e^t * c) dt
c = ∫_0^1 e^t dt + c * ∫_0^1 e^(2t) dt
c = (e - 1) + c * (e² - 1)/2

Solving for c:
c - c(e² - 1)/2 = e - 1
c(1 - (e² - 1)/2) = e - 1
c(2 - e² + 1)/2 = e - 1
c(3 - e²)/2 = e - 1
c = 2(e - 1)/(3 - e²)

Therefore: u(x) = 1 + 2(e-1)/(3-e²) * e^x""",
    },
]


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
        raise ValueError(f"Unknown prompt style: {style}. Available: {list(SYSTEM_PROMPTS.keys())}")
    
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
            prompt_parts.append(f"\nExample {i+1}:")
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
