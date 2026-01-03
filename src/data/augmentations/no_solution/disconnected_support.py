"""
No-solution augmentation strategy for disconnected kernel support.

Generates equations where kernel support is split into disconnected regions,
creating rank-deficient operators with no solution.
"""

from typing import Any

import sympy as sp

from src.data.augmentations.base import BaseAugmentation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DisconnectedSupportAugmentation(BaseAugmentation):
    """
    Generate no-solution cases with disconnected kernel support.

    Mathematical Background:
        When K(x,t) has support in disconnected regions, the integral operator
        splits into decoupled subsystems with no information flow between them.
        This creates a rank-deficient matrix that cannot satisfy arbitrary RHS.

    Example:
        K(x,t) ≠ 0 only in [a, a+0.2] × [a, a+0.2] and [b-0.2, b] × [b-0.2, b]
        
        The equation splits into two independent sub-problems:
        - ∫[a, a+0.2] K₁(x,t) u(t) dt = f(x) for x ∈ [a, a+0.2]
        - ∫[b-0.2, b] K₂(x,t) u(t) dt = f(x) for x ∈ [b-0.2, b]
        
        But there's no constraint on f(x) for x in the gap region!
        → Generic f(x) cannot be satisfied → no solution

    The Challenge for LLM:
        Model must:
        - Recognize disconnected support structure
        - Identify decoupled subsystems
        - Conclude operator is rank-deficient
        - Determine no solution exists for generic RHS

    Label:
        {
            "has_solution": false,
            "solution_type": "none",
            "edge_case": "disconnected_support",
            "reason": "Disconnected kernel support → rank-deficient operator",
            "problem_structure": "Decoupled subsystems with no coupling"
        }
    """

    @property
    def strategy_name(self) -> str:
        return "disconnected_support"

    @property
    def description(self) -> str:
        return "Generate no-solution cases with disconnected kernel support"

    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate disconnected support cases."""
        results = []

        try:
            # Extract base parameters
            a = float(sp.sympify(item.get("a", "0")))
            b = float(sp.sympify(item.get("b", "1")))
            lambda_val = float(sp.sympify(item.get("lambda", item.get("lambda_val", "1"))))

            # Case 1: Two disconnected regions
            region1_x = (a, a + 0.2 * (b - a))
            region2_x = (a + 0.6 * (b - a), a + 0.8 * (b - a))
            
            case1 = {
                "u": "",  # No solution - rank deficient
                "f": item.get("f", "x"),
                "kernel": "Piecewise: nonzero in two disconnected regions",
                "kernel_description": f"K≠0 in [{region1_x[0]:.2f},{region1_x[1]:.2f}] and [{region2_x[0]:.2f},{region2_x[1]:.2f}], zero elsewhere",
                "lambda_val": str(lambda_val * 0.2),
                "a": str(a),
                "b": str(b),
                "has_solution": False,
                "solution_type": "none",
                "edge_case": "no_solution",
                "support_type": "disconnected_regions",
                "num_support_regions": 2,
                "zero_fraction": 0.8,  # 80% is zero
                "matrix_structure": "block_diagonal",
                "rank_deficient_risk": "high",
                "problem_structure": "Decoupled subsystems",
                "recommended_methods": [
                    "check_rank_deficiency",
                    "identify_decoupled_blocks",
                    "analyze_subsystems_separately"
                ],
                "numerical_challenge": "Disconnected regions → no information flow between subsystems",
                "reason": "Kernel support is disconnected → matrix is rank-deficient",
                "mathematical_issue": "Operator does not have full rank",
                "augmented": True,
                "augmentation_type": self.strategy_name,
                "augmentation_variant": "two_disconnected_regions",
            }
            results.append(case1)

            # Case 2: Three disconnected regions (more extreme)
            region1 = (a, a + 0.15 * (b - a))
            region2 = (a + 0.4 * (b - a), a + 0.55 * (b - a))
            region3 = (a + 0.8 * (b - a), b)
            
            case2 = {
                "u": "",
                "f": item.get("f", "sin(x)"),
                "kernel": "Piecewise: nonzero in three disconnected regions",
                "kernel_description": f"K≠0 in three separate regions, covering only 30% of domain",
                "lambda_val": str(lambda_val * 0.15),
                "a": str(a),
                "b": str(b),
                "has_solution": False,
                "solution_type": "none",
                "edge_case": "no_solution",
                "support_type": "disconnected_regions",
                "num_support_regions": 3,
                "zero_fraction": 0.7,
                "matrix_structure": "block_diagonal",
                "rank_deficient_risk": "very_high",
                "problem_structure": "Three decoupled subsystems",
                "recommended_methods": [
                    "rank_analysis",
                    "block_structure_detection",
                    "compatibility_check"
                ],
                "numerical_challenge": "Multiple gaps → severe rank deficiency",
                "reason": "Three disconnected support regions → highly rank-deficient operator",
                "mathematical_issue": "Operator has very low rank relative to domain size",
                "augmented": True,
                "augmentation_type": self.strategy_name,
                "augmentation_variant": "three_disconnected_regions",
            }
            results.append(case2)

        except Exception as e:
            logger.debug(f"Disconnected support augmentation failed: {e}")

        return results
