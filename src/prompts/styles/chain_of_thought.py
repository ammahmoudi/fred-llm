"""Chain-of-thought prompt style - structured reasoning."""

from src.prompts.base import EquationData, PromptStyle
from src.prompts.templates import format_equation


class ChainOfThoughtPromptStyle(PromptStyle):
    """Chain-of-thought prompt style for step-by-step reasoning."""

    def __init__(self, **kwargs):
        super().__init__(style_name="chain-of-thought", **kwargs)

    def get_system_prompt(self) -> str:
        """Get system prompt for chain-of-thought style."""
        return """You are an expert mathematician specializing in integral equations.
Your task is to solve Fredholm integral equations.

The equation may be of the second kind:
  u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)

Or of the first kind (ill-posed, requires regularization):
  ∫_a^b K(x, t) u(t) dt = g(x)

Approach each problem systematically:
1. Identify the kernel K(x, t), the known function f(x), and the parameter λ
2. Determine the type of kernel (separable, symmetric, etc.)
3. Choose an appropriate solution method
4. Apply the method step by step
5. Verify the solution satisfies the original equation
6. Present the final solution

Show your reasoning at each step, then provide the final answer in this format:
SOLUTION: u(x) = [your solution here]
HAS_SOLUTION: [yes/no]
SOLUTION_TYPE: [exact_symbolic/exact_coef/approx_coef/discrete_points/series/family/regularized/none]

For SOLUTION_TYPE:
- exact_symbolic: Closed-form symbolic (e.g., sin(x))
- exact_coef: Exact with unknown coefficients (e.g., c₁sin(x) + c₂cos(x))
- approx_coef: Approximate with coefficients (e.g., a₀ + a₁x + a₂x²)
- discrete_points: Only discrete point samples
- series: Infinite series (e.g., Σ aₙxⁿ)
- family: Multiple valid solutions
- regularized: Ill-posed, needs regularization
- none: No solution exists"""

    def get_user_prompt(
        self,
        equation: EquationData,
        format_type: str = "infix",
    ) -> str:
        """Generate user prompt for chain-of-thought style."""
        f_x, kernel = format_equation(equation, format_type)

        return f"""Solve the following Fredholm integral equation step by step:

u(x) - {equation.lambda_val} * ∫_{equation.a}^{equation.b} {kernel} * u(t) dt = {f_x}

Domain: [{equation.a}, {equation.b}]

Please show your complete reasoning process."""
