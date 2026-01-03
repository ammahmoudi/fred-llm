"""Tool-assisted prompt style - enables computational tools."""

from src.prompts.base import EquationData, PromptStyle
from src.prompts.templates import format_equation


class ToolAssistedPromptStyle(PromptStyle):
    """Tool-assisted prompt style with computational tools."""

    def __init__(self, **kwargs):
        super().__init__(style_name="tool-assisted", **kwargs)

    def get_system_prompt(self) -> str:
        """Get system prompt for tool-assisted style."""
        return """You are an expert mathematician with access to computational tools.
Your task is to solve Fredholm integral equations.

The equation may be of the second kind:
  u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)

Or of the first kind (ill-posed, requires regularization):
  ∫_a^b K(x, t) u(t) dt = g(x)

You can use the following tools:
- integrate(expr, var, a, b): Compute definite integral
- simplify(expr): Simplify mathematical expression
- solve(equation, var): Solve equation for variable
- series(expr, var, n): Expand in series

Show your work using these tools when helpful, then provide the final answer in this format:
SOLUTION: u(x) = [your solution here]
HAS_SOLUTION: [yes/no]
SOLUTION_TYPE: [exact_symbolic/exact_coef/approx_coef/discrete_points/series/family/regularized/none]

For SOLUTION_TYPE:
- exact_symbolic: Closed-form symbolic solution
- exact_coef: Exact with unknown coefficients
- approx_coef: Approximate with coefficients
- discrete_points: Solution only at discrete points
- series: Infinite series solution
- family: Family of solutions (non-unique)
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
