"""
Boundary layer augmentation strategy.

Generates equations with solutions that have sharp gradients near boundaries,
requiring adaptive mesh refinement for accurate numerical solutions.
"""

from typing import Any

import sympy as sp

from src.data.augmentations.base import BaseAugmentation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BoundaryLayerAugmentation(BaseAugmentation):
    """
    Generate boundary layer solution cases.

    Mathematical Background:
        Boundary layer solutions have the form:
            u(x) = u_outer(x) + u_layer(x)
        where u_layer(x) decays rapidly as you move away from boundary.

        Common example:
            u(x) = exp(-(x-a)/ε) for small ε > 0

        The solution changes very rapidly near x=a (boundary layer of width ~ε).

    The Challenge:
        Standard uniform discretization misses the rapid variation.
        Requires:
        - Adaptive mesh refinement (AMR)
        - Exponentially graded mesh near boundary
        - Very fine spacing in layer region

    Physical Context:
        Boundary layers appear in:
        - Fluid dynamics (thin viscous layers near walls)
        - Reaction-diffusion equations (transition zones)
        - Semiconductor modeling (depletion regions)

    Example:
        u(x) - λ∫[0,1] K(x,t) u(t) dt = exp(-x/ε)

        If ε = 0.01, the solution will have boundary layer at x=0 of width ~0.01.
        Standard quadrature with spacing > 0.01 will fail.

    Label:
        {
            "has_solution": true,
            "solution_type": "numerical",
            "edge_case": "boundary_layer",
            "layer_location": "left|right|both",
            "layer_width_estimate": epsilon_value,
            "recommended_methods": ["adaptive_mesh", "exponential_grading"]
        }
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        num_sample_points: int = 20
    ) -> None:
        """
        Initialize boundary layer augmentation.

        Args:
            epsilon: Boundary layer thickness parameter (smaller = sharper layer)
            num_sample_points: Number of sample points (will be graded)
        """
        self.epsilon = epsilon
        self.num_sample_points = num_sample_points

    @property
    def strategy_name(self) -> str:
        return "boundary_layer"

    @property
    def description(self) -> str:
        return "Generate solutions with sharp gradients near boundaries"

    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate boundary layer cases."""
        results = []

        try:
            # Extract base parameters
            a = float(sp.sympify(item.get("a", "0")))
            b = float(sp.sympify(item.get("b", "1")))
            lambda_val = float(sp.sympify(item.get("lambda", item.get("lambda_val", "1"))))

            # Reduce lambda to prevent numerical instability
            lambda_scaled = lambda_val * 0.2

            # Case 1: Left boundary layer - exp(-(x-a)/ε)
            # Sharp gradient at left boundary x=a
            case1 = {
                "u": f"exp(-(x - {a})/{self.epsilon})",  # Boundary layer solution
                "f": f"exp(-(x - {a})/{self.epsilon})",  # RHS matches layer
                "kernel": "x * t",  # Simple separable kernel
                "lambda_val": str(lambda_scaled),
                "lambda_val": str(lambda_scaled),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "numerical",
                "edge_case": "boundary_layer",
                "layer_location": "left",
                "layer_width_estimate": self.epsilon,
                "gradient_scale": 1.0 / self.epsilon,  # du/dx ~ 1/ε in layer
                "recommended_methods": [
                    "adaptive_mesh_refinement",
                    "exponential_mesh_grading",
                    "shishkin_mesh"
                ],
                "numerical_challenge": f"Rapid variation in layer of width {self.epsilon} at x={a}",
                "minimum_points_in_layer": int(10),  # Need at least 10 points in layer
                "augmented": True,
                "augmentation_type": "boundary_layer",
                "augmentation_variant": "left_exponential_layer",
            }
            results.append(case1)

            # Case 2: Right boundary layer - exp((x-b)/ε)
            # Sharp gradient at right boundary x=b
            case2 = {
                "u": f"exp((x - {b})/{self.epsilon})",
                "f": f"exp((x - {b})/{self.epsilon})",
                "kernel": "1 + x + t",  # Slightly more complex kernel
                "lambda_val": str(lambda_scaled),
                "lambda_val": str(lambda_scaled),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "numerical",
                "edge_case": "boundary_layer",
                "layer_location": "right",
                "layer_width_estimate": self.epsilon,
                "gradient_scale": 1.0 / self.epsilon,
                "recommended_methods": [
                    "adaptive_mesh_refinement",
                    "exponential_mesh_grading",
                    "shishkin_mesh"
                ],
                "numerical_challenge": f"Rapid variation in layer of width {self.epsilon} at x={b}",
                "minimum_points_in_layer": int(10),
                "augmented": True,
                "augmentation_type": "boundary_layer",
                "augmentation_variant": "right_exponential_layer",
            }
            results.append(case2)

            # Case 3: Double boundary layers - tanh profile
            # Layers at both boundaries
            mid = (a + b) / 2
            case3 = {
                "u": f"(tanh((x - {a})/{self.epsilon}) + tanh(({b} - x)/{self.epsilon}))/2",
                "f": f"(tanh((x - {a})/{self.epsilon}) + tanh(({b} - x)/{self.epsilon}))/2",
                "kernel": "sin(x) * cos(t)",
                "lambda_val": str(lambda_scaled * 0.5),  # Reduce further for stability
                "lambda_val": str(lambda_scaled * 0.5),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "numerical",
                "edge_case": "boundary_layer",
                "layer_location": "both",
                "layer_width_estimate": self.epsilon,
                "gradient_scale": 1.0 / self.epsilon,
                "recommended_methods": [
                    "adaptive_mesh_refinement",
                    "double_exponential_grading",
                    "shishkin_mesh_double"
                ],
                "numerical_challenge": f"Two boundary layers of width {self.epsilon} at x={a} and x={b}",
                "minimum_points_in_layer": int(15),  # More points needed for 2 layers
                "augmented": True,
                "augmentation_type": "boundary_layer",
                "augmentation_variant": "double_tanh_layers",
            }
            results.append(case3)

        except Exception as e:
            logger.warning(f"Failed to generate boundary layer case: {e}")

        return results

    def _generate_boundary_layer_mesh(
        self, a: float, b: float, layer_position: float, epsilon: float
    ) -> list[list[float]]:
        """
        Generate mesh with exponential grading near boundary layer.

        Args:
            a: Lower bound
            b: Upper bound
            layer_position: Position of boundary layer (a or b)
            epsilon: Layer width

        Returns:
            List of [x, t] sample points
        """
        import numpy as np

        n = self.num_sample_points

        # Cap the exponential argument to prevent overflow
        # exp(x) overflows around x > 700, so we cap at a reasonable value
        max_exp_arg = 50  # exp(50) ≈ 5e21, sufficient for graded mesh
        exp_arg = min((b - a) / epsilon, max_exp_arg)
        
        if layer_position == a:
            # Graded mesh near x=a
            # Use transformation: x = a + ε*log(1 + α*ξ) where ξ ∈ [0,1]
            alpha = np.exp(exp_arg) - 1
            xi = np.linspace(0, 1, n)
            # Ensure no division issues
            x_points = a + epsilon * np.log(1 + alpha * xi + 1e-10)
        else:
            # Graded mesh near x=b (mirror)
            alpha = np.exp(exp_arg) - 1
            xi = np.linspace(0, 1, n)
            x_points = b - epsilon * np.log(1 + alpha * (1 - xi) + 1e-10)

        # Generate sample points
        samples = []
        for xi in x_points:
            for ti in x_points[:n//2]:  # Use half for t to limit total points
                samples.append([float(xi), float(ti)])
                if len(samples) >= self.num_sample_points:
                    break
            if len(samples) >= self.num_sample_points:
                break

        return samples[:self.num_sample_points]

    def _generate_double_layer_mesh(
        self, a: float, b: float, epsilon: float
    ) -> list[list[float]]:
        """
        Generate mesh for double boundary layer (Shishkin-type).

        Piecewise uniform mesh with fine regions near both boundaries.
        """
        import numpy as np

        n = self.num_sample_points
        n_layer = n // 3  # Points in each layer
        n_interior = n - 2 * n_layer  # Points in interior

        # Left layer [a, a + 3ε]
        left_layer = np.linspace(a, a + 3 * epsilon, n_layer)
        
        # Interior [a + 3ε, b - 3ε]
        interior = np.linspace(a + 3 * epsilon, b - 3 * epsilon, n_interior)
        
        # Right layer [b - 3ε, b]
        right_layer = np.linspace(b - 3 * epsilon, b, n_layer)

        x_points = np.concatenate([left_layer, interior, right_layer])

        # Generate sample points
        samples = []
        for xi in x_points:
            for ti in x_points[::2]:  # Subsample t to reduce total count
                samples.append([float(xi), float(ti)])
                if len(samples) >= self.num_sample_points:
                    break
            if len(samples) >= self.num_sample_points:
                break

        return samples[:self.num_sample_points]
