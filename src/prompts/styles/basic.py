"""Basic prompt style - simple and direct."""

from src.prompts.base import EquationData, PromptStyle
from src.prompts.templates import format_equation


class BasicPromptStyle(PromptStyle):
    """Basic prompt style with minimal scaffolding."""

    def __init__(self, **kwargs):
        super().__init__(style_name="basic", **kwargs)

    def get_system_prompt(self, format_type: str = "infix") -> str:
        """Get system prompt for basic style.
        
        Args:
            format_type: Output format (infix/latex/rpn)
        """
        # Format-specific instructions
        format_instructions = {
            "infix": "Express your solution in infix notation (e.g., x**2 + sin(x), exp(-x)*cos(x)).",
            "latex": "Express your solution in LaTeX notation (e.g., x^2 + \\sin(x), e^{-x}\\cos(x)).",
            "rpn": "Express your solution in Reverse Polish Notation (e.g., x 2 ^ x sin +, x neg exp x cos *)."
        }
        format_instruction = format_instructions.get(format_type, format_instructions["infix"])
        
        return f"""You are an expert mathematician specializing in integral equations.
Given a Fredholm integral equation, find the solution u(x).

The equation may be of the second kind:
  u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)

Or of the first kind (ill-posed, requires regularization):
  ∫_a^b K(x, t) u(t) dt = g(x)

**IMPORTANT**: {format_instruction}

Provide your answer in the following format:
SOLUTION: u(x) = [your solution here]
HAS_SOLUTION: [yes/no]
SOLUTION_TYPE: [exact_symbolic/exact_coef/approx_coef/discrete_points/series/family/regularized/none]

For SOLUTION_TYPE:
- exact_symbolic: Closed-form symbolic solution (e.g., u(x) = sin(x))
- exact_coef: Exact with unknown coefficients (e.g., u(x) = c₁sin(x) + c₂cos(x))
- approx_coef: Approximate with coefficients (e.g., u(x) ≈ a₀ + a₁x + a₂x²)
- discrete_points: Solution only at discrete points
- series: Infinite series solution (e.g., u(x) = Σ aₙxⁿ)
- family: Family of solutions (non-unique)
- regularized: Ill-posed, requires regularization
- none: No solution exists

If no solution exists, write "No solution" for SOLUTION."""

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
