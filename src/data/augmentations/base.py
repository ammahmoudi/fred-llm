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
                - u_values_samples: Optional list of u(x) values for each constant sample
                - constant_samples: Optional list of constant samples used

        Raises:
            Exception: If symbolic parsing or evaluation fails
        """
        try:
            x = sp.Symbol('x')
            t = sp.Symbol('t')
            u_func = sp.sympify(u_expr)
            free_constants = u_func.free_symbols - {x, t}
            constant_samples = [-1.0, 1.0, 2.0]

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
            if free_constants:
                u_values_samples = []
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    for sample in constant_samples:
                        substituted = u_func.subs({sym: sample for sym in free_constants})
                        u_lambda = sp.lambdify(x, substituted, modules=['numpy'])
                        u_values_samples.append(
                            np.array([u_lambda(float(xi)) for xi in x_values], dtype=float)
                        )

                finite_mask = np.isfinite(u_values_samples[0])
                for values in u_values_samples[1:]:
                    finite_mask &= np.isfinite(values)
            else:
                u_lambda = sp.lambdify(x, u_func, modules=['numpy'])
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    u_values = np.array(
                        [u_lambda(float(xi)) for xi in x_values],
                        dtype=float,
                    )
                finite_mask = np.isfinite(u_values)
            if not np.any(finite_mask):
                raise ValueError("All evaluation points produced non-finite values")

            x_values = x_values[finite_mask]

            if free_constants:
                filtered_samples = [values[finite_mask] for values in u_values_samples]
                sample_index = constant_samples.index(1.0) if 1.0 in constant_samples else 0
                u_values = filtered_samples[sample_index]
                return {
                    "x_values": x_values.tolist(),
                    "u_values": u_values.tolist(),
                    "u_values_samples": [values.tolist() for values in filtered_samples],
                    "constant_samples": constant_samples,
                    "n_points": len(x_values),
                }

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
