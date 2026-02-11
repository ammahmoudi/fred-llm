"""
Base classes for prompt generation.

Defines data structures and abstract base class for prompt styles.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class EquationData:
    """Container for equation data from CSV."""

    u: str  # Solution
    f: str  # Known function
    kernel: str  # Kernel K(x,t)
    lambda_val: float  # Lambda parameter
    a: float  # Lower bound
    b: float  # Upper bound
    equation_id: str | None = None  # Optional ID for tracking

    # Edge case fields (optional)
    has_solution: bool | None = None  # Whether a solution exists
    solution_type: str | None = (
        None  # exact_symbolic, approx_coef, discrete_points, series, family, regularized, none
    )


@dataclass
class GeneratedPrompt:
    """Container for generated prompt with metadata."""

    prompt: str
    equation_id: str | None
    style: str
    format_type: str
    ground_truth: str | None = None
    metadata: dict[str, Any] | None = None


# Available edge case hint fields that can be included in prompts
EDGE_CASE_HINT_FIELDS = {
    "has_solution",
    "solution_type",
}


@dataclass
class EdgeCaseMode:
    """
    Configuration for edge case handling in prompts.

    Modes:
    - "none": No hints (pure inference mode)
    - "guardrails": Add instructions to handle edge cases without hints
    - "hints": Include selected hint fields in prompt
    """

    mode: str = "none"  # "none", "guardrails", "hints"
    hint_fields: list[str] | None = None  # Which fields to include when mode="hints"

    def __post_init__(self):
        if self.mode not in {"none", "guardrails", "hints"}:
            raise ValueError(
                f"Invalid mode: {self.mode}. Use 'none', 'guardrails', or 'hints'"
            )
        if self.mode == "hints" and not self.hint_fields:
            # Default to all available fields
            self.hint_fields = list(EDGE_CASE_HINT_FIELDS)
        if self.hint_fields:
            invalid = set(self.hint_fields) - EDGE_CASE_HINT_FIELDS
            if invalid:
                raise ValueError(
                    f"Invalid hint fields: {invalid}. Valid: {EDGE_CASE_HINT_FIELDS}"
                )


class PromptStyle(ABC):
    """Abstract base class for prompt generation styles."""

    def __init__(
        self,
        style_name: str,
        include_examples: bool = True,
        num_examples: int = 2,
        edge_case_mode: EdgeCaseMode | None = None,
    ):
        """
        Initialize prompt style.

        Args:
            style_name: Name of the prompt style
            include_examples: Whether to include examples (for few-shot)
            num_examples: Number of examples to include
            edge_case_mode: Configuration for edge case handling
        """
        self.style_name = style_name
        self.include_examples = include_examples
        self.num_examples = num_examples
        self.edge_case_mode = edge_case_mode or EdgeCaseMode(mode="none")

    @abstractmethod
    def get_system_prompt(self, format_type: str = "infix") -> str:
        """Get the system prompt for this style.

        Args:
            format_type: Output format (infix/latex/rpn)
        """
        pass

    @abstractmethod
    def get_user_prompt(
        self,
        equation: EquationData,
        format_type: str = "infix",
    ) -> str:
        """
        Generate user prompt for the equation.

        Args:
            equation: Equation data
            format_type: Format type (infix/latex/rpn)

        Returns:
            User prompt string
        """
        pass

    def _get_guardrails_text(self) -> str:
        """Get guardrails text for edge case handling."""
        return """
Note: Solutions may take different forms:
- Exact symbolic (e.g., u(x) = sin(x))
- Approximate with NUMERIC coefficients (e.g., u(x) = 0.5 + 1.2*x - 0.3*x²)
- Discrete points only
- Infinite series (e.g., u(x) = Σ aₙxⁿ)
- Family of solutions with ARBITRARY parameters (e.g., u(x) = c₁*sin(x) + c₂*cos(x))
- Ill-posed requiring regularization
- No solution exists

State clearly which type applies and provide appropriate representation."""

    def _get_hints_text(self, equation: EquationData) -> str:
        """Get hints text based on edge case fields."""
        if not self.edge_case_mode.hint_fields:
            return ""

        hints = []
        for field in self.edge_case_mode.hint_fields:
            value = getattr(equation, field, None)
            if value is not None:
                if field == "has_solution":
                    hints.append(f"Has solution: {'Yes' if value else 'No'}")
                elif field == "solution_type":
                    hints.append(f"Type: {value}")

        if hints:
            return "\n[" + ", ".join(hints) + "]"
        return ""

    def generate(
        self,
        equation: EquationData,
        format_type: str = "infix",
        include_ground_truth: bool = True,
    ) -> GeneratedPrompt:
        """
        Generate a complete prompt for the equation.

        Args:
            equation: Equation data
            format_type: Format type (infix/latex/rpn)
            include_ground_truth: Whether to include the solution

        Returns:
            GeneratedPrompt object
        """
        system_prompt = self.get_system_prompt(format_type=format_type)
        user_prompt = self.get_user_prompt(equation, format_type)

        # Build full prompt with optional edge case handling
        full_prompt = system_prompt

        if self.edge_case_mode.mode == "guardrails":
            full_prompt += self._get_guardrails_text()
        elif self.edge_case_mode.mode == "hints":
            full_prompt += self._get_guardrails_text()

        full_prompt += f"\n\n{user_prompt}"

        if self.edge_case_mode.mode == "hints":
            full_prompt += self._get_hints_text(equation)

        metadata = {
            "kernel": equation.kernel,
            "f": equation.f,
            "lambda_val": equation.lambda_val,
            "domain": [equation.a, equation.b],
        }

        # Add edge case info to metadata if present
        if equation.has_solution is not None:
            metadata["has_solution"] = equation.has_solution
        if equation.solution_type:
            metadata["solution_type"] = equation.solution_type

        return GeneratedPrompt(
            prompt=full_prompt,
            equation_id=equation.equation_id,
            style=self.style_name,
            format_type=format_type,
            ground_truth=equation.u if include_ground_truth else None,
            metadata=metadata,
        )

    def generate_batch(
        self,
        equations: list[EquationData],
        format_type: str = "infix",
        include_ground_truth: bool = True,
    ) -> list[GeneratedPrompt]:
        """
        Generate prompts for multiple equations.

        Args:
            equations: List of equation data
            format_type: Format type (infix/latex/rpn)
            include_ground_truth: Whether to include solutions

        Returns:
            List of GeneratedPrompt objects
        """
        prompts = []
        for equation in equations:
            prompt = self.generate(equation, format_type, include_ground_truth)
            prompts.append(prompt)

        logger.info(f"Generated {len(prompts)} prompts using {self.style_name} style")
        return prompts
