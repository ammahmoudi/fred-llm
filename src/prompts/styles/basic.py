"""Basic prompt style - simple and direct."""

from src.prompts.base import EquationData, PromptStyle
from src.prompts.templates import format_equation


class BasicPromptStyle(PromptStyle):
    """Basic prompt style with minimal scaffolding."""

    def __init__(self, **kwargs):
        super().__init__(style_name="basic", **kwargs)

    def get_system_prompt(self) -> str:
        """Get system prompt for basic style."""
        return """You are an expert mathematician specializing in integral equations.
Given a Fredholm integral equation of the second kind, find the solution u(x).

The general form is: u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)

Provide your answer as a mathematical expression for u(x)."""

    def get_user_prompt(
        self,
        equation: EquationData,
        format_type: str = "infix",
    ) -> str:
        """Generate user prompt for basic style."""
        f_x, kernel = format_equation(equation, format_type)

        return f"""Solve the following Fredholm integral equation:

u(x) - {equation.lambda_val} * ∫_{equation.a}^{equation.b} {kernel} * u(t) dt = {f_x}

Domain: [{equation.a}, {equation.b}]

Provide the solution u(x)."""
