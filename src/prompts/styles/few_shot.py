"""Few-shot prompt style - includes examples."""

from src.prompts.base import EquationData, PromptStyle
from src.prompts.templates import FEW_SHOT_EXAMPLES, format_equation


class FewShotPromptStyle(PromptStyle):
    """Few-shot prompt style with worked examples."""

    def __init__(self, include_examples: bool = True, num_examples: int = 2, **kwargs):
        super().__init__(
            style_name="few-shot",
            include_examples=include_examples,
            num_examples=num_examples,
            **kwargs,
        )

    def get_system_prompt(self, format_type: str = "infix") -> str:
        """Get system prompt with examples for few-shot style.

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

        base_prompt = f"""You are an expert mathematician specializing in integral equations.
I will show you examples of solved Fredholm integral equations, then ask you to solve a new one.

The equation may be of the second kind:
  u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)

Or of the first kind (ill-posed, requires regularization):
  ∫_a^b K(x, t) u(t) dt = g(x)

**IMPORTANT**: {format_instruction}
Provide your final answer in this format:
SOLUTION: u(x) = [your solution here]
HAS_SOLUTION: [yes/no]
SOLUTION_TYPE: [exact_symbolic/approx_coef/discrete_points/series/family/regularized/none]

For SOLUTION_TYPE:
- exact_symbolic: Closed-form symbolic solution
- approx_coef: Approximate with NUMERIC coefficients (e.g., 0.5 + 1.2*x)
- discrete_points: Point values only. Format: [(x1, y1), (x2, y2), ...]
- series: Truncated series with exactly 4 explicit terms in SOLUTION
- family: Non-unique solutions (arbitrary c_1, c_2, ...)
- regularized: Ill-posed, requires regularization
- none: No solution exists"""

        if not self.include_examples:
            return base_prompt

        # Add examples
        examples_text = "\n\n"
        for i, example in enumerate(FEW_SHOT_EXAMPLES[: self.num_examples], 1):
            examples_text += f"""Example {i}:
{example["problem"]}

Solution:
{example["solution"]}

"""

        return base_prompt + examples_text

    def get_user_prompt(
        self,
        equation: EquationData,
        format_type: str = "infix",
    ) -> str:
        """Generate user prompt for few-shot style."""
        f_x, kernel = format_equation(equation, format_type)

        return f"""Now solve this equation:

u(x) - {equation.lambda_val} * ∫_{equation.a}^{equation.b} {kernel} * u(t) dt = {f_x}

Domain: [{equation.a}, {equation.b}]

Solution:"""
