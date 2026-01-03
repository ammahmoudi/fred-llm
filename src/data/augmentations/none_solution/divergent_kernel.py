"""
Non-integrable singularity augmentation strategy.

Generates equations with kernels that have non-integrable singularities,
making the integral undefined and the equation mathematically ill-formed.
"""

from typing import Any

import sympy as sp

from src.data.augmentations.base import BaseAugmentation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DivergentKernelAugmentation(BaseAugmentation):
    """
    Generate non-integrable singularity cases.

    Mathematical Background:
        A kernel K(x,t) has a non-integrable (strong) singularity if:
            ∫∫ |K(x,t)| dx dt = ∞

        Common examples:
        - K(x,t) = 1/|x-t| (Cauchy kernel) - diverges logarithmically
        - K(x,t) = 1/|x-t|^α for α ≥ 1 - power law divergence
        - K(x,t) = 1/(x-t)² - second order pole

    The Problem:
        The integral ∫K(x,t)u(t)dt does not exist (diverges), making the
        Fredholm equation undefined. This is fundamentally different from
        weakly singular kernels where the integral converges.

    Contrast with Weakly Singular:
        - Weakly singular: 0 < α < 1, integral converges but needs special methods
        - Non-integrable: α ≥ 1, integral diverges → equation is meaningless

    The Challenge for LLM:
        Model must:
        - Detect the singularity order
        - Recognize when singularity is too strong (non-integrable)
        - Distinguish from weakly singular (integrable) cases
        - Conclude equation is mathematically undefined

    Physical Context:
        - Point charges in electrostatics (1/r singularity in 3D is OK, but 1/r in 1D diverges)
        - Crack tips with wrong singularity exponent
        - Inappropriate Green's functions

    Label:
        {
            "has_solution": false,
            "solution_type": "none",
            "edge_case": "divergent_kernel",
            "reason": "kernel has non-integrable singularity",
            "singularity_strength": "strong|very_strong",
            "divergence_type": "logarithmic|power_law|exponential"
        }
    """

    @property
    def strategy_name(self) -> str:
        return "divergent_kernel"

    @property
    def description(self) -> str:
        return "Kernels with non-integrable singularities"

    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate divergent kernel cases."""
        results = []

        try:
            # Extract base parameters
            a = float(sp.sympify(item.get("a", "0")))
            b = float(sp.sympify(item.get("b", "1")))
            lambda_val = float(
                sp.sympify(item.get("lambda", item.get("lambda_val", "1")))
            )

            # Case 1: Cauchy kernel 1/|x-t| (diverges logarithmically in 1D)
            case1 = {
                "u": "",  # Integral doesn't exist
                "f": item["f"],  # Keep original RHS
                "kernel": "1 / (abs(x - t) + 1e-10)",  # Non-integrable
                "lambda_val": str(lambda_val * 0.1),
                "lambda_val": str(lambda_val * 0.1),
                "a": str(a),
                "b": str(b),
                "has_solution": False,
                "solution_type": "none",
                "edge_case": "divergent_kernel",
                "reason": "Kernel K(x,t) = 1/|x-t| has non-integrable singularity",
                "singularity_order": 1.0,  # α = 1
                "singularity_strength": "strong",
                "divergence_type": "logarithmic",
                "mathematical_explanation": "∫₀¹ 1/|x-t| dt diverges logarithmically",
                "contrast_with": "weakly_singular (α < 1 is integrable)",
                "recommended_methods": [
                    "reject_as_ill_formed",
                    "propose_regularization",
                    "suggest_cauchy_principal_value",
                ],
                "augmented": True,
                "augmentation_type": "divergent_kernel",
                "augmentation_variant": "cauchy_singularity",
            }
            results.append(case1)

            # Case 2: Second-order pole 1/(x-t)² (strongly divergent)
            case2 = {
                "u": "",  # Integral doesn't exist
                "f": item["f"],
                "kernel": "1 / (abs(x - t) + 1e-10)**2",  # Very strong singularity
                "lambda_val": str(lambda_val * 0.05),
                "lambda_val": str(lambda_val * 0.05),
                "a": str(a),
                "b": str(b),
                "has_solution": False,
                "solution_type": "none",
                "edge_case": "divergent_kernel",
                "reason": "Kernel K(x,t) = 1/|x-t|² diverges too strongly",
                "singularity_order": 2.0,  # α = 2
                "singularity_strength": "very_strong",
                "divergence_type": "power_law",
                "mathematical_explanation": "∫₀¹ 1/|x-t|² dt = ∞ (power law divergence)",
                "physically_invalid": "Second-order pole in 1D integral equation",
                "recommended_methods": [
                    "reject_as_ill_formed",
                    "check_dimension_compatibility",
                    "suggest_finite_part_integral",
                ],
                "augmented": True,
                "augmentation_type": "divergent_kernel",
                "augmentation_variant": "second_order_pole",
            }
            results.append(case2)

            # Case 3: Mixed smooth and divergent
            # K(x,t) = (1 + x*t) / |x-t|
            # The 1/|x-t| dominates and causes divergence
            case3 = {
                "u": "",  # Integral doesn't exist
                "f": item["f"],
                "kernel": "(1 + x*t) / (abs(x - t) + 1e-10)",
                "lambda_val": str(lambda_val * 0.1),
                "lambda_val": str(lambda_val * 0.1),
                "a": str(a),
                "b": str(b),
                "has_solution": False,
                "solution_type": "none",
                "edge_case": "divergent_kernel",
                "reason": "Smooth part multiplied by non-integrable singularity",
                "singularity_order": 1.0,
                "singularity_strength": "strong",
                "divergence_type": "logarithmic_with_weight",
                "mathematical_explanation": "∫ (1+x*t)/|x-t| dt still diverges despite smooth factor",
                "kernel_structure": "smooth_function * singular_part",
                "recommended_methods": [
                    "singularity_extraction",
                    "dominant_singularity_analysis",
                    "reject_if_unfixable",
                ],
                "augmented": True,
                "augmentation_type": "divergent_kernel",
                "augmentation_variant": "mixed_smooth_divergent",
            }
            results.append(case3)

        except Exception as e:
            logger.warning(f"Failed to generate divergent kernel case: {e}")

        return results
