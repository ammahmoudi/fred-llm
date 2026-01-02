"""
Prompt generation module for Fredholm integral equations.

Provides base classes and style implementations for prompt generation.
"""

from src.prompts.base import (
    EDGE_CASE_HINT_FIELDS,
    EdgeCaseMode,
    EquationData,
    GeneratedPrompt,
    PromptStyle,
)
from src.prompts.batch_processor import BatchPromptProcessor, create_processor
from src.prompts.factory import create_edge_case_mode, create_prompt_style
from src.prompts.styles.basic import BasicPromptStyle
from src.prompts.styles.chain_of_thought import ChainOfThoughtPromptStyle
from src.prompts.styles.few_shot import FewShotPromptStyle
from src.prompts.styles.tool_assisted import ToolAssistedPromptStyle

__all__ = [
    # Data classes
    "EquationData",
    "GeneratedPrompt",
    # Base class
    "PromptStyle",
    # Edge case handling
    "EdgeCaseMode",
    "EDGE_CASE_HINT_FIELDS",
    # Style implementations
    "BasicPromptStyle",
    "ChainOfThoughtPromptStyle",
    "FewShotPromptStyle",
    "ToolAssistedPromptStyle",
    # Factory functions
    "create_prompt_style",
    "create_edge_case_mode",
    "BatchPromptProcessor",
    "create_processor",
]
