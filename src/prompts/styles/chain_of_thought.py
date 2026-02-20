"""Chain-of-thought prompt style - structured reasoning."""

from src.prompts.base import EquationData, PromptStyle
from src.prompts.templates import format_equation


class ChainOfThoughtPromptStyle(PromptStyle):
    """Chain-of-thought prompt style for step-by-step reasoning."""

    def __init__(self, **kwargs):
        super().__init__(style_name="chain-of-thought", **kwargs)

    def get_system_prompt(self, format_type: str = "infix") -> str:
        """Get system prompt for chain-of-thought style.

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

        return f"""You are an expert mathematician specializing in integral equations.
Your task is to solve Fredholm integral equations.

The equation may be of the second kind:
  u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)

Or of the first kind (ill-posed, requires regularization):
  ∫_a^b K(x, t) u(t) dt = g(x)

**IMPORTANT**: {format_instruction}

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
SOLUTION_TYPE: [exact_symbolic/approx_coef/discrete_points/series/family/regularized/none]

For SOLUTION_TYPE:
- exact_symbolic: Closed-form symbolic (e.g., sin(x))
- approx_coef: Approximate with NUMERIC coefficients (e.g., 0.5*sin(x) + 1.2*x)
- discrete_points: Point values only. Format: [(x1, y1), (x2, y2), ...]
- series: Truncated series with exactly 4 explicit terms in SOLUTION
- family: Non-unique solutions (arbitrary c_1, c_2, ...)
- regularized: Ill-posed, needs regularization
- none: No solution exists

After your reasoning, you MUST provide the final answer as a **single, self-contained mathematical expression** for u(x):
- Do NOT leave unevaluated integrals — compute them numerically if needed.
- Do NOT define auxiliary variables — substitute everything into one expression in x only.
- Do NOT use LaTeX formatting commands like \\Bigl, \\Bigr, \\displaystyle, or \\qquad.

FINAL_ANSWER: u(x) = [single expression in x only]"""

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
