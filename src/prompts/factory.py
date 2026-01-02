"""Factory function for creating prompt styles."""

from src.prompts.base import EdgeCaseMode, PromptStyle
from src.prompts.styles.basic import BasicPromptStyle
from src.prompts.styles.chain_of_thought import ChainOfThoughtPromptStyle
from src.prompts.styles.few_shot import FewShotPromptStyle
from src.prompts.styles.tool_assisted import ToolAssistedPromptStyle
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_edge_case_mode(
    mode: str = "none",
    hint_fields: list[str] | None = None,
) -> EdgeCaseMode:
    """
    Factory function to create edge case mode configuration.

    Args:
        mode: Mode for edge case handling
            - "none": No edge case instructions (pure inference)
            - "guardrails": Add instructions to handle edge cases without hints
            - "hints": Include selected hint fields in prompt
        hint_fields: Which fields to include when mode="hints"
            Options: has_solution, solution_type
            If None with mode="hints", all available fields are included.

    Returns:
        EdgeCaseMode instance
    """
    return EdgeCaseMode(mode=mode, hint_fields=hint_fields)


def create_prompt_style(
    style: str,
    include_examples: bool = True,
    num_examples: int = 2,
    edge_case_mode: EdgeCaseMode | str | None = None,
    hint_fields: list[str] | None = None,
) -> PromptStyle:
    """
    Factory function to create prompt style instances.

    Args:
        style: Style name (basic, chain-of-thought, few-shot, tool-assisted)
        include_examples: Whether to include examples (for few-shot)
        num_examples: Number of examples to include
        edge_case_mode: EdgeCaseMode instance, or string mode ("none", "guardrails", "hints")
        hint_fields: Hint fields to include (only used if edge_case_mode is a string)

    Returns:
        PromptStyle instance

    Raises:
        ValueError: If style is unknown
    """
    styles = {
        "basic": BasicPromptStyle,
        "chain-of-thought": ChainOfThoughtPromptStyle,
        "few-shot": FewShotPromptStyle,
        "tool-assisted": ToolAssistedPromptStyle,
    }

    if style not in styles:
        raise ValueError(
            f"Unknown prompt style: {style}. Available: {list(styles.keys())}"
        )

    # Handle edge_case_mode
    if isinstance(edge_case_mode, str):
        edge_case_mode = create_edge_case_mode(mode=edge_case_mode, hint_fields=hint_fields)
    elif edge_case_mode is None:
        edge_case_mode = EdgeCaseMode(mode="none")

    style_class = styles[style]

    if style == "few-shot":
        return style_class(
            include_examples=include_examples,
            num_examples=num_examples,
            edge_case_mode=edge_case_mode,
        )
    else:
        return style_class(edge_case_mode=edge_case_mode)
