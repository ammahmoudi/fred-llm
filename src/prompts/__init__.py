"""
Prompt generation module for Fredholm integral equations.

Provides base classes and style implementations for prompt generation.
"""

from src.prompts.base import EquationData, GeneratedPrompt, PromptStyle
from src.prompts.batch_processor import BatchPromptProcessor, create_processor
from src.prompts.factory import create_prompt_style
from src.prompts.styles.basic import BasicPromptStyle
from src.prompts.styles.chain_of_thought import ChainOfThoughtPromptStyle
from src.prompts.styles.few_shot import FewShotPromptStyle
from src.prompts.styles.tool_assisted import ToolAssistedPromptStyle

__all__ = [
    "EquationData",
    "GeneratedPrompt",
    "PromptStyle",
    "BasicPromptStyle",
    "ChainOfThoughtPromptStyle",
    "FewShotPromptStyle",
    "ToolAssistedPromptStyle",
    "create_prompt_style",
    "BatchPromptProcessor",
    "create_processor",
]
