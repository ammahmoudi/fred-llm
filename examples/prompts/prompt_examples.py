"""
Example prompts demonstrating different styles for Fredholm integral equations.

These examples show how prompts are generated for the equation:
    u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)
"""

# Example equation data
EXAMPLE_EQUATION = {
    "u": "x**2",
    "f": "x**2 + (x**3)/3",
    "kernel": "x*t",
    "lambda_val": 1.0,
    "a": 0.0,
    "b": 1.0,
    "equation_id": "example_001",
}


# ============================================================================
# BASIC STYLE - Simple and direct
# ============================================================================

BASIC_PROMPT = """You are an expert mathematician specializing in integral equations.
Given a Fredholm integral equation of the second kind, find the solution u(x).

The general form is: u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)

Provide your answer as a mathematical expression for u(x).

Solve the following Fredholm integral equation:

u(x) - 1.0 * ∫_0.0^1.0 x*t * u(t) dt = x**2 + (x**3)/3

Domain: [0.0, 1.0]

Provide the solution u(x)."""


# ============================================================================
# CHAIN-OF-THOUGHT STYLE - Structured reasoning
# ============================================================================

CHAIN_OF_THOUGHT_PROMPT = """You are an expert mathematician specializing in integral equations.
Your task is to solve Fredholm integral equations of the second kind.

The general form is: u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)

Approach each problem systematically:
1. Identify the kernel K(x, t), the known function f(x), and the parameter λ
2. Determine the type of kernel (separable, symmetric, etc.)
3. Choose an appropriate solution method
4. Apply the method step by step
5. Verify the solution satisfies the original equation
6. Present the final solution u(x)

Show your reasoning at each step.

Solve the following Fredholm integral equation step by step:

u(x) - 1.0 * ∫_0.0^1.0 x*t * u(t) dt = x**2 + (x**3)/3

Domain: [0.0, 1.0]

Please show your complete reasoning process."""


# ============================================================================
# FEW-SHOT STYLE - With worked examples
# ============================================================================

FEW_SHOT_PROMPT = """You are an expert mathematician specializing in integral equations.
I will show you examples of solved Fredholm integral equations, then ask you to solve a new one.

The general form is: u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)


Example 1:
u(x) - ∫_0^1 x*t * u(t) dt = x

Solution:
Let me solve this step by step.

The kernel K(x,t) = x*t is separable, so I can write:
∫_0^1 x*t * u(t) dt = x * ∫_0^1 t * u(t) dt = x * c

where c = ∫_0^1 t * u(t) dt is a constant.

Substituting back: u(x) = x + x*c = x(1 + c)

To find c, multiply both sides by t and integrate:
c = ∫_0^1 t * u(t) dt = ∫_0^1 t * t(1+c) dt = (1+c) * ∫_0^1 t² dt = (1+c)/3

Solving: c = (1+c)/3 → 3c = 1 + c → 2c = 1 → c = 1/2

Therefore: u(x) = x(1 + 1/2) = 3x/2


Example 2:
u(x) - ∫_0^1 e^(x+t) * u(t) dt = 1

Solution:
The kernel K(x,t) = e^(x+t) = e^x * e^t is separable.

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

Therefore: u(x) = 1 + 2(e-1)/(3-e²) * e^x



Now solve this equation:

u(x) - 1.0 * ∫_0.0^1.0 x*t * u(t) dt = x**2 + (x**3)/3

Domain: [0.0, 1.0]

Solution:"""


# ============================================================================
# TOOL-ASSISTED STYLE - With computational tools
# ============================================================================

TOOL_ASSISTED_PROMPT = """You are an expert mathematician with access to computational tools.
Your task is to solve Fredholm integral equations of the second kind.

The general form is: u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)

You can use the following tools:
- integrate(expr, var, a, b): Compute definite integral
- simplify(expr): Simplify mathematical expression
- solve(equation, var): Solve equation for variable
- series(expr, var, n): Expand in series

Show your work using these tools when helpful.

Solve the following Fredholm integral equation using available tools:

u(x) - 1.0 * ∫_0.0^1.0 x*t * u(t) dt = x**2 + (x**3)/3

Domain: [0.0, 1.0]

Show your work and state the final answer for u(x)."""


# ============================================================================
# METADATA EXAMPLE
# ============================================================================

PROMPT_METADATA = {
    "equation_id": "example_001",
    "style": "chain-of-thought",
    "format_type": "infix",
    "ground_truth": "x**2",
    "metadata": {
        "kernel": "x*t",
        "f": "x**2 + (x**3)/3",
        "lambda_val": 1.0,
        "domain": [0.0, 1.0],
    },
}


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def show_examples():
    """Print all prompt examples."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax

    console = Console()

    examples = [
        ("Basic Style", BASIC_PROMPT),
        ("Chain-of-Thought Style", CHAIN_OF_THOUGHT_PROMPT),
        ("Few-Shot Style", FEW_SHOT_PROMPT),
        ("Tool-Assisted Style", TOOL_ASSISTED_PROMPT),
    ]

    for title, prompt in examples:
        console.print(Panel.fit(title, style="bold cyan"))
        console.print(prompt)
        console.print("\n" + "=" * 80 + "\n")


def generate_custom_prompt():
    """Example: Generate a custom prompt using the API."""
    from src.prompts import create_prompt_style, EquationData

    # Create equation
    eq = EquationData(
        u="x**2",
        f="x**2 + (x**3)/3",
        kernel="x*t",
        lambda_val=1.0,
        a=0.0,
        b=1.0,
        equation_id="custom_001",
    )

    # Generate different styles
    styles = ["basic", "chain-of-thought", "few-shot", "tool-assisted"]
    
    for style_name in styles:
        style = create_prompt_style(style_name)
        prompt = style.generate(eq, include_ground_truth=True)
        
        print(f"\n{'='*80}")
        print(f"Style: {prompt.style}")
        print(f"{'='*80}")
        print(prompt.prompt)


if __name__ == "__main__":
    show_examples()
