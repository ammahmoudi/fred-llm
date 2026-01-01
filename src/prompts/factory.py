"""Factory function for creating prompt styles."""

from src.prompts.base import PromptStyle
from src.prompts.styles.basic import BasicPromptStyle
from src.prompts.styles.chain_of_thought import ChainOfThoughtPromptStyle
from src.prompts.styles.few_shot import FewShotPromptStyle
from src.prompts.styles.tool_assisted import ToolAssistedPromptStyle
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_prompt_style(
    style: str,
    include_examples: bool = True,
    num_examples: int = 2,
) -> PromptStyle:
    """
    Factory function to create prompt style instances.

    Args:
        style: Style name (basic, chain-of-thought, few-shot, tool-assisted)
        include_examples: Whether to include examples (for few-shot)
        num_examples: Number of examples to include

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

    style_class = styles[style]

    if style == "few-shot":
        return style_class(include_examples=include_examples, num_examples=num_examples)
    else:
        return style_class()
