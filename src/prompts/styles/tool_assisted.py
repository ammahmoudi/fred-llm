"""Tool-assisted prompt style - enables computational tools."""

from src.prompts.base import EquationData, PromptStyle
from src.prompts.templates import format_equation


class ToolAssistedPromptStyle(PromptStyle):
    """Tool-assisted prompt style with computational tools."""

    def __init__(self, **kwargs):
        super().__init__(style_name="tool-assisted", **kwargs)

    def get_system_prompt(self, format_type: str = "infix") -> str:
        """Get system prompt for tool-assisted style.

        Args:
            format_type: Output format (infix/latex/rpn)
        """
        # Format-specific instructions
        format_instructions = {
            "infix": "Express your solution in infix notation (e.g., x**2 + sin(x), exp(-x)*cos(x)).",
            "latex": "Express your solution in LaTeX notation (e.g., x^2 + \\sin(x), e^{-x}\\cos(x)).",
            "rpn": "Express your solution in Reverse Polish Notation (e.g., x 2 ^ x sin +, x neg exp x cos *).",
        }
        format_instruction = format_instructions.get(
            format_type, format_instructions["infix"]
        )

        return f"""You are an expert mathematician with access to computational tools.
Your task is to solve Fredholm integral equations.

The equation may be of the second kind:
  u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)

Or of the first kind (ill-posed, requires regularization):
  ∫_a^b K(x, t) u(t) dt = g(x)

**IMPORTANT**: {format_instruction}
You can use the following tools:
- integrate(expr, var, a, b): Compute definite integral
- simplify(expr): Simplify mathematical expression
- solve(equation, var): Solve equation for variable
- series(expr, var, n): Expand in series

Show your work using these tools when helpful, then provide the final answer in this format:
SOLUTION: u(x) = [your solution here]
HAS_SOLUTION: [yes/no]
SOLUTION_TYPE: [exact_symbolic/approx_coef/discrete_points/series/family/regularized/none]

For SOLUTION_TYPE:
- exact_symbolic: Closed-form symbolic solution
- approx_coef: Approximate with NUMERIC coefficients (e.g., 0.5 + 1.2*x)
- discrete_points: Point values only. Format: [(x1, y1), (x2, y2), ...]
- series: Infinite series solution
- family: Non-unique solutions (arbitrary c_1, c_2, ...)
- regularized: Ill-posed, requires regularization
- none: No solution exists"""

    def get_user_prompt(
        self,
        equation: EquationData,
        format_type: str = "infix",
    ) -> str:
        """Generate user prompt for tool-assisted style."""
        f_x, kernel = format_equation(equation, format_type)

        return f"""Solve the following Fredholm integral equation using available tools:

u(x) - {equation.lambda_val} * ∫_{equation.a}^{equation.b} {kernel} * u(t) dt = {f_x}

Domain: [{equation.a}, {equation.b}]

Show your work and state the final answer for u(x)."""
