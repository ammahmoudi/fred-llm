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

    def get_system_prompt(self) -> str:
        """Get system prompt with examples for few-shot style."""
        base_prompt = """You are an expert mathematician specializing in integral equations.
I will show you examples of solved Fredholm integral equations, then ask you to solve a new one.

The general form is: u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)"""

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
