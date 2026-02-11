"""
Base class for augmentation strategies.

Provides interface for dataset augmentation with edge cases.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import sympy as sp
import logging

logger = logging.getLogger(__name__)


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

    def _generate_evaluation_points(
        self, u_expr: str, a: float, b: float, n_points: int = 50
    ) -> dict[str, Any]:
        """
        Generate fixed evaluation points for consistent numeric evaluation.

        This method pre-computes evaluation points and their corresponding
        solution values for use in evaluation. This ensures consistent
        RMSE/MAE metrics across evaluation runs.

        Args:
            u_expr: Solution function as a string (e.g., "x**2 + sin(x)")
            a: Lower bound of domain
            b: Upper bound of domain
            n_points: Number of evaluation points (default: 50)

        Returns:
            Dictionary containing:
                - x_values: List of x coordinates
                - u_values: List of corresponding u(x) values
                - n_points: Total number of points

        Raises:
            Exception: If symbolic parsing or evaluation fails
        """
        try:
            x = sp.Symbol('x')
            u_func = sp.sympify(u_expr)
            u_lambda = sp.lambdify(x, u_func, modules=['numpy'])

            # Generate uniform grid
            x_uniform = np.linspace(a, b, n_points)

            # Add critical points: boundaries, midpoint, near-boundary points
            critical_points = [
                a,  # Left boundary
                b,  # Right boundary
                (a + b) / 2,  # Midpoint
                a + 0.1 * (b - a),  # Near left boundary
                b - 0.1 * (b - a),  # Near right boundary
            ]

            # Combine and remove duplicates
            x_values = np.concatenate([x_uniform, critical_points])
            x_values = np.sort(np.unique(x_values))

            # Evaluate solution at all points, filtering non-finite values
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                u_values = np.array(
                    [u_lambda(float(xi)) for xi in x_values],
                    dtype=float,
                )

            finite_mask = np.isfinite(u_values)
            if not np.any(finite_mask):
                raise ValueError("All evaluation points produced non-finite values")

            x_values = x_values[finite_mask]
            u_values = u_values[finite_mask]

            return {
                "x_values": x_values.tolist(),
                "u_values": u_values.tolist(),
                "n_points": len(x_values),
            }

        except Exception as e:
            logger.debug(
                f"Failed to generate evaluation points for u='{u_expr}': {e}"
            )
            raise
