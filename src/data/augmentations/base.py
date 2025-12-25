"""
Base class for augmentation strategies.

Provides interface for dataset augmentation with edge cases.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAugmentation(ABC):
    """Base class for augmentation strategies."""

    @abstractmethod
    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Apply augmentation to a single equation.

        Args:
            item: Original equation dictionary with keys:
                - u: Solution function
                - f: Right-hand side function
                - kernel: Kernel function K(x,t)
                - lambda: Lambda parameter (or lambda_val)
                - a: Lower integration bound
                - b: Upper integration bound

        Returns:
            List of augmented equation dictionaries with same structure
            plus augmentation metadata.
        """
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name of this augmentation strategy."""
        pass

    @property
    def description(self) -> str:
        """Return a description of what this augmentation does."""
        return ""
