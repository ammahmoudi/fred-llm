"""Prompt templates and examples for Fredholm equations."""

from src.prompts.base import EquationData

# Few-shot examples for demonstration
FEW_SHOT_EXAMPLES = [
    {
        "problem": "u(x) - ∫_0^1 x*t * u(t) dt = x",
        "solution": """Let me solve this step by step.

The kernel K(x,t) = x*t is separable, so I can write:
∫_0^1 x*t * u(t) dt = x * ∫_0^1 t * u(t) dt = x * c

where c = ∫_0^1 t * u(t) dt is a constant.

Substituting back: u(x) = x + x*c = x(1 + c)

To find c, multiply both sides by t and integrate:
c = ∫_0^1 t * u(t) dt = ∫_0^1 t * t(1+c) dt = (1+c) * ∫_0^1 t² dt = (1+c)/3

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


def format_equation_line(equation: EquationData, format_type: str) -> str:
    """
    Render the full equation statement for prompts.

    lambda_val == 0 is the first-kind sentinel set by the ill_posed
    augmentation (first-kind equations have no λ). Rendering it in the
    second-kind template would produce the trivial u(x) = f(x) and hide
    the ill-posedness from the model.
    """
    f_x, kernel = format_equation(equation, format_type)
    try:
        first_kind = float(equation.lambda_val) == 0
    except (TypeError, ValueError):
        first_kind = False
    if first_kind:
        return f"∫_{equation.a}^{equation.b} {kernel} * u(t) dt = {f_x}"
    return (
        f"u(x) - {equation.lambda_val} * "
        f"∫_{equation.a}^{equation.b} {kernel} * u(t) dt = {f_x}"
    )


def format_equation(equation: EquationData, format_type: str) -> tuple[str, str]:
    """
    Format equation components based on format type.

    Args:
        equation: Equation data
        format_type: Format type (infix/latex/rpn)

    Returns:
        Tuple of (f_x, kernel) formatted strings
    """
    if format_type == "latex":
        # LaTeX formatting
        f_x = equation.f.replace("**", "^").replace("*", r"\cdot ")
        kernel = equation.kernel.replace("**", "^").replace("*", r"\cdot ")
    elif format_type == "rpn":
        # RPN already formatted, display as-is
        f_x = f"RPN: {equation.f}"
        kernel = f"RPN: {equation.kernel}"
    else:
        # Infix (default)
        f_x = equation.f
        kernel = equation.kernel

    return f_x, kernel
