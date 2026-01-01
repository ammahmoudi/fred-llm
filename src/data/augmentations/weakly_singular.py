"""
Weakly singular kernel augmentation strategy.

Generates equations with kernels that have integrable singularities,
requiring specialized numerical techniques (singularity subtraction).
"""

from typing import Any

import sympy as sp

from src.data.augmentations.base import BaseAugmentation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class WeaklySingularAugmentation(BaseAugmentation):
    """
    Generate weakly singular kernel cases.

    Mathematical Background:
        Weakly singular kernels have the form K(x,t) ~ |x-t|^(-α) where 0 < α < 1.
        Common examples:
        - Logarithmic: K(x,t) = log|x-t| (α = 0, but still singular)
        - Power law: K(x,t) = |x-t|^(-1/2) (α = 0.5)
        - Abel kernel: K(x,t) = (x-t)^(-α) for Volterra-type

    The Challenge:
        While the integral exists (converges), standard quadrature fails near
        singularities. Requires:
        - Product integration methods
        - Singularity subtraction techniques
        - Graded mesh refinement

    Example:
        u(x) - λ∫[0,1] log|x-t| u(t) dt = f(x)

        The kernel explodes at x=t but the integral is well-defined.
        Solution exists but standard Nyström or collocation will be inaccurate.

    Label:
        {
            "has_solution": true,
            "solution_type": "numerical",
            "edge_case": "weakly_singular",
            "singularity_type": "logarithmic|power_law|algebraic",
            "recommended_methods": ["product_integration", "singularity_subtraction"]
        }
    """

    def __init__(self, num_sample_points: int = 15) -> None:
        """
        Initialize weakly singular augmentation.

        Args:
            num_sample_points: Number of sample points for numerical solution
        """
        self.num_sample_points = num_sample_points

    @property
    def strategy_name(self) -> str:
        return "weakly_singular"

    @property
    def description(self) -> str:
        return "Generate kernels with integrable singularities"

    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate weakly singular kernel cases."""
        results = []

        try:
            # Extract base parameters
            a = float(sp.sympify(item.get("a", "0")))
            b = float(sp.sympify(item.get("b", "1")))
            lambda_val = float(
                sp.sympify(item.get("lambda", item.get("lambda_val", "1")))
            )

            # Scale lambda to ensure stability
            lambda_scaled = lambda_val * 0.1  # Reduce to prevent dominance

            # Case 1: Logarithmic singularity - log|x-t|
            # Most common in potential theory and crack problems
            case1 = {
                "u": item["u"],  # Keep original solution as reference
                "f": item["f"],
                "kernel": "log(abs(x - t) + 1e-10)",  # Add epsilon to avoid log(0)
                "lambda": str(lambda_scaled),
                "lambda_val": str(lambda_scaled),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "numerical",
                "edge_case": "weakly_singular",
                "singularity_type": "logarithmic",
                "singularity_order": 0,  # log singularity has order 0
                "recommended_methods": [
                    "product_integration",
                    "singularity_subtraction",
                    "graded_mesh",
                ],
                "numerical_challenge": "Kernel diverges at x=t (logarithmically)",
                "augmented": True,
                "augmentation_type": "weakly_singular",
                "augmentation_variant": "logarithmic_kernel",
            }
            results.append(case1)

            # Case 2: Power law singularity - |x-t|^(-1/2)
            # Common in Abel integral equations
            case2 = {
                "u": item["u"],
                "f": item["f"],
                "kernel": "1/sqrt(abs(x - t) + 1e-8)",  # Avoid division by zero
                "lambda": str(
                    lambda_scaled * 0.5
                ),  # Further reduce for stronger singularity
                "lambda_val": str(lambda_scaled * 0.5),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "numerical",
                "edge_case": "weakly_singular",
                "singularity_type": "power_law",
                "singularity_order": 0.5,  # α = 0.5
                "recommended_methods": [
                    "product_integration",
                    "abel_transform",
                    "collocation_with_grading",
                ],
                "numerical_challenge": "Kernel ~ |x-t|^(-0.5) near singularity",
                "augmented": True,
                "augmentation_type": "weakly_singular",
                "augmentation_variant": "power_law_kernel",
            }
            results.append(case2)

            # Case 3: Algebraic singularity with smooth part
            # K(x,t) = (x + t) / sqrt(|x - t|)
            case3 = {
                "u": item["u"],
                "f": item["f"],
                "kernel": "(x + t + 1) / sqrt(abs(x - t) + 1e-8)",
                "lambda": str(lambda_scaled * 0.3),
                "lambda_val": str(lambda_scaled * 0.3),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "numerical",
                "edge_case": "weakly_singular",
                "singularity_type": "algebraic_mixed",
                "singularity_order": 0.5,
                "recommended_methods": [
                    "product_integration",
                    "singularity_extraction",
                    "nyström_modified",
                ],
                "numerical_challenge": "Mixed smooth and singular parts require decomposition",
                "augmented": True,
                "augmentation_type": "weakly_singular",
                "augmentation_variant": "algebraic_mixed_kernel",
            }
            results.append(case3)

        except Exception as e:
            logger.warning(f"Failed to generate weakly singular case: {e}")

        return results

    def _generate_sample_points(
        self, a: float, b: float, mesh_type: str = "graded"
    ) -> list[list[float]]:
        """
        Generate sample points for validation.

        Args:
            a: Lower bound
            b: Upper bound
            mesh_type: "uniform" or "graded" (concentrated near singularities)

        Returns:
            List of [x, t] coordinate pairs
        """
        import numpy as np

        if mesh_type == "graded":
            # Use graded mesh: more points near boundaries where singularities occur
            # Transform uniform points through r^p to get grading
            n = self.num_sample_points
            p = 2  # Grading parameter
            uniform = np.linspace(0, 1, n)
            graded = uniform**p  # Concentration near 0
            x_points = a + (b - a) * graded

            # Create sample points avoiding diagonal x=t
            samples = []
            for xi in x_points:
                for ti in x_points:
                    if abs(xi - ti) > (b - a) / (2 * n):  # Avoid near-diagonal
                        samples.append([float(xi), float(ti)])
                        if len(samples) >= self.num_sample_points:
                            break
                if len(samples) >= self.num_sample_points:
                    break
        else:
            # Uniform mesh
            n = self.num_sample_points
            x_points = np.linspace(a, b, n)
            samples = []
            for xi in x_points:
                for ti in x_points:
                    if abs(xi - ti) > (b - a) / (2 * n):
                        samples.append([float(xi), float(ti)])
                        if len(samples) >= self.num_sample_points:
                            break
                if len(samples) >= self.num_sample_points:
                    break

        return samples[: self.num_sample_points]
