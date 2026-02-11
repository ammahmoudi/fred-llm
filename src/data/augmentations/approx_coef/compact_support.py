"""
Compact support kernel augmentation strategy.

Generates kernels that are zero over large portions of the domain,
creating "dead zones" and potentially rank-deficient matrices.
"""

from typing import Any

import sympy as sp

from src.data.augmentations.base import BaseAugmentation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CompactSupportAugmentation(BaseAugmentation):
    """
    Generate compact support kernel cases.

    Mathematical Background:
        A kernel K(x,t) has compact support if it is zero outside a bounded region:
            K(x,t) = 0  for (x,t) ∉ S
        where S is a compact set.

        Examples:
        - K(x,t) = { sin(x*t)  if |x-t| < δ     (band-limited)
                   { 0         otherwise

        - K(x,t) = { x*t       if x,t ∈ [c,d] ⊂ [a,b]  (localized)
                   { 0         otherwise

    The Problem:
        1. **Sparse influence**: Many points don't interact
        2. **Rank deficiency**: Discretized matrix has many zero rows/columns
        3. **Information loss**: Parts of domain are uncoupled
        4. **Numerical issues**: May not be invertible or well-conditioned

    The Challenge for LLM:
        Model must:
        - Recognize large zero regions in kernel
        - Identify support set (where K ≠ 0)
        - Detect potential rank deficiency
        - Recommend sparse matrix methods
        - Check if problem is well-posed

    Physical Context:
        - Local interactions (finite range forces)
        - Communication networks (limited connectivity)
        - Image processing (local convolution kernels)
        - Covariance matrices with spatial decay

    Label:
        {
            "has_solution": "depends",  # May have solution but with caveats
            "solution_type": "approx_coef",
            "edge_case": "compact_support",
            "support_type": "band|diagonal|localized",
            "zero_fraction": 0.7,  # Fraction of domain where K=0
            "rank_deficient_risk": "high|moderate|low"
        }
    """

    def __init__(self, bandwidth: float = 0.1) -> None:
        """
        Initialize compact support augmentation.

        Args:
            bandwidth: Width of support region (smaller = more sparse)
        """
        self.bandwidth = bandwidth

    @property
    def strategy_name(self) -> str:
        return "compact_support"

    @property
    def description(self) -> str:
        return "Kernels with limited spatial support"

    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate compact support cases."""
        results = []

        try:
            # Extract base parameters
            a = float(sp.sympify(item.get("a", "0")))
            b = float(sp.sympify(item.get("b", "1")))
            lambda_val = float(
                sp.sympify(item.get("lambda", item.get("lambda_val", "1")))
            )

            # Case 1: Band-limited kernel (near-diagonal)
            # K(x,t) ≠ 0 only if |x-t| < δ
            # Creates banded matrix structure
            delta = self.bandwidth
            zero_fraction1 = 1.0 - (2 * delta / (b - a))

            case1 = {
                "u": item["u"],  # Solution may exist but needs sparse methods
                "f": item["f"],
                "kernel": f"Piecewise((x*t, Abs(x - t) < {delta}), (0, True))",
                "kernel_description": f"K(x,t) = x*t if |x-t|<{delta}, else 0 (band-limited)",
                "lambda_val": str(lambda_val * 0.5),
                "lambda_val": str(lambda_val * 0.5),
                "a": str(a),
                "b": str(b),
                "has_solution": True,  # Usually exists but sparse
                "solution_type": "approx_coef",
                "edge_case": "compact_support",
                "support_type": "band",
                "support_width": delta,
                "zero_fraction": zero_fraction1,
                "matrix_structure": "banded",
                "bandwidth_parameter": delta,
                "rank_deficient_risk": "low" if delta > 0.05 else "moderate",
                "recommended_methods": [
                    "sparse_matrix_solvers",
                    "banded_matrix_algorithms",
                    "iterative_krylov_methods",
                ],
                "numerical_challenge": f"{100 * zero_fraction1:.0f}% of kernel is zero → sparse structure",
                "memory_efficiency": "High (can use sparse storage)",
                "augmented": True,
                "augmentation_type": "compact_support",
                "augmentation_variant": "band_limited",
            }
            # Generate evaluation points for consistent evaluation
            if case1.get("has_solution") and case1.get("u"):
                try:
                    a_val = float(sp.sympify(case1.get("a", "0")))
                    b_val = float(sp.sympify(case1.get("b", "1")))
                    case1["evaluation_points"] = self._generate_evaluation_points(
                        case1["u"], a_val, b_val
                    )
                except Exception as e:
                    logger.debug(f"Failed to generate evaluation points: {e}")
            results.append(case1)

            # Case 2: Localized support - kernel only nonzero in sub-region
            # K(x,t) ≠ 0 only if both x,t ∈ [c,d] where [c,d] ⊂ [a,b]
            c = a + 0.3 * (b - a)
            d = a + 0.6 * (b - a)
            region_fraction = (d - c) / (b - a)
            zero_fraction2 = 1.0 - region_fraction**2

            case2 = {
                "u": item["u"],
                "f": item["f"],
                "kernel": (
                    "Piecewise((sin(x)*cos(t), "
                    f"(x>={c}) & (x<={d}) & (t>={c}) & (t<={d})), (0, True))"
                ),
                "kernel_description": f"K nonzero only in [{c:.2f},{d:.2f}]×[{c:.2f},{d:.2f}]",
                "lambda_val": str(lambda_val * 0.3),
                "lambda_val": str(lambda_val * 0.3),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "approx_coef",
                "edge_case": "compact_support",
                "support_type": "localized_box",
                "support_region": f"[{c:.2f}, {d:.2f}] × [{c:.2f}, {d:.2f}]",
                "zero_fraction": zero_fraction2,
                "matrix_structure": "block_sparse",
                "active_fraction": region_fraction**2,
                "rank_deficient_risk": "moderate",
                "recommended_methods": [
                    "domain_decomposition",
                    "block_sparse_solvers",
                    "restricted_problem_reduction",
                ],
                "numerical_challenge": f"{100 * zero_fraction2:.0f}% zero → only {100 * region_fraction**2:.1f}% of domain interacts",
                "physical_interpretation": "Localized interaction region in larger domain",
                "augmented": True,
                "augmentation_type": "compact_support",
                "augmentation_variant": "localized_box",
            }
            # Generate evaluation points for consistent evaluation
            if case2.get("has_solution") and case2.get("u"):
                try:
                    a_val = float(sp.sympify(case2.get("a", "0")))
                    b_val = float(sp.sympify(case2.get("b", "1")))
                    case2["evaluation_points"] = self._generate_evaluation_points(
                        case2["u"], a_val, b_val
                    )
                except Exception as e:
                    logger.debug(f"Failed to generate evaluation points: {e}")
            results.append(case2)

        except Exception as e:
            logger.warning(f"Failed to generate compact support case: {e}")

        return results
