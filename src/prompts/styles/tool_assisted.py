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
Your task is to solve Fredholm integral equations of the second kind.

The general form is: u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)

You can use the following tools:
- integrate(expr, var, a, b): Compute definite integral
- simplify(expr): Simplify mathematical expression
- solve(equation, var): Solve equation for variable
- series(expr, var, n): Expand in series

Show your work using these tools when helpful."""

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
