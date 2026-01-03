"""
Ill-posed augmentation strategy.

Generates Fredholm equations of the first kind (ill-posed problems).
These are extremely sensitive to noise and require regularization.
"""

from typing import Any

import numpy as np
import sympy as sp
from scipy import integrate

from src.data.augmentations.base import BaseAugmentation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class IllPosedAugmentation(BaseAugmentation):
    """
    Generate ill-posed cases (Fredholm 1st kind).

    Category 3: The "Ill-Posed" Case (Fredholm 1st Kind)

    The Logic:
        Change from 2nd kind: u(x) - λ∫K(x,t)u(t)dt = f(x)
        To 1st kind: ∫K(x,t)u(t)dt = f(x)

        This is obtained by setting the equation as:
        0 - ∫K(x,t)u(t)dt = -f(x)
        or simply: ∫K(x,t)u(t)dt = f(x)

    The Problem:
        First kind equations are ill-posed:
        - Tiny changes in f(x) cause huge changes in u(x)
        - Solution may not exist or may not be unique
        - Requires regularization (Tikhonov, Landweber, etc.)
        - High-frequency noise gets amplified

    LLM Task:
        The model should learn to:
        1. Identify this as Fredholm 1st kind (no u(x) term outside integral)
        2. Recognize it's ill-posed
        3. Recommend regularization techniques
        4. Note solution is unstable without regularization

    Label:
        {
            "equation_type": "fredholm_first_kind",
            "is_ill_posed": true,
            "requires_regularization": true,
            "recommended_methods": ["Tikhonov", "Landweber", "TSVD"],
            "solution_type": "regularized"
        }
    """

    def __init__(self, num_sample_points: int = 10, regularization_param: float = 0.01):
        """
        Initialize ill-posed augmentation.

        Args:
            num_sample_points: Number of points for numerical demonstration.
            regularization_param: Tikhonov regularization parameter α.
        """
        self.num_sample_points = num_sample_points
        self.alpha = regularization_param

    @property
    def strategy_name(self) -> str:
        return "ill_posed"

    @property
    def description(self) -> str:
        return (
            "Generate Fredholm 1st kind equations (ill-posed, require regularization)"
        )

    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate ill-posed (1st kind) cases."""
        results = []

        try:
            a = float(sp.sympify(item.get("a", "0")))
            b = float(sp.sympify(item.get("b", "1")))

            # Case 1: Simple kernel K(x,t) = x*t, f(x) = x²
            case1 = self._create_simple_first_kind(a, b)
            if case1:
                results.append(case1)

            # Case 2: Exponential kernel (more challenging)
            case2 = self._create_exponential_first_kind(a, b)
            if case2:
                results.append(case2)

            # Case 3: Oscillatory kernel (highly ill-posed)
            case3 = self._create_oscillatory_first_kind(a, b)
            if case3:
                results.append(case3)

        except Exception as e:
            logger.debug(f"Ill-posed augmentation failed: {e}")

        return results

    def _create_simple_first_kind(self, a: float, b: float) -> dict[str, Any] | None:
        """Create simple Fredholm 1st kind: ∫x*t*u(t)dt = x²."""
        try:
            # For demonstration, we know u(t) = t would give ∫x*t²dt = x*t³/3|_a^b
            # To get f(x) = x², we need careful construction

            # Simplified: just show the ill-posed nature
            x_points = np.linspace(a, b, self.num_sample_points)

            return {
                "u": "",  # No analytical solution - requires numerical regularization
                "f": "x**2",
                "kernel": "x*t",
                "lambda_val": "0",  # First kind has no λ parameter
                "a": str(a),
                "b": str(b),
                "has_solution": True,  # Solution exists but unstable without regularization
                "equation_type": "fredholm_first_kind",
                "equation_form": "∫K(x,t)u(t)dt = f(x)",
                "is_ill_posed": True,
                "requires_regularization": True,
                "recommended_methods": [
                    "Tikhonov",
                    "Truncated SVD",
                    "Landweber iteration",
                ],
                "solution_type": "regularized",
                "edge_case": "ill_posed",
                "reason": "First kind equation - extremely sensitive to noise in f(x)",
                "warning": "Solution unstable without regularization",
                "regularization_param": self.alpha,
                "augmented": True,
                "augmentation_type": "ill_posed",
                "augmentation_variant": "simple_first_kind",
            }
        except Exception as e:
            logger.debug(f"Simple first kind case failed: {e}")
            return None

    def _create_exponential_first_kind(
        self, a: float, b: float
    ) -> dict[str, Any] | None:
        """Create exponential kernel 1st kind: ∫exp(x*t)*u(t)dt = exp(x)."""
        try:
            return {
                "u": "",  # No analytical solution - requires numerical regularization
                "f": "exp(x)",
                "kernel": "exp(x*t)",
                "lambda_val": "0",
                "a": str(a),
                "b": str(b),
                "has_solution": True,  # Solution exists but unstable without regularization
                "equation_type": "fredholm_first_kind",
                "equation_form": "∫K(x,t)u(t)dt = f(x)",
                "is_ill_posed": True,
                "requires_regularization": True,
                "recommended_methods": [
                    "Tikhonov",
                    "Spectral cutoff",
                    "Iterative methods",
                ],
                "solution_type": "regularized",
                "edge_case": "ill_posed",
                "reason": "Exponential kernel amplifies high-frequency components",
                "warning": "Highly ill-conditioned - small noise in f causes large errors in u",
                "regularization_param": self.alpha,
                "augmented": True,
                "augmentation_type": "ill_posed",
                "augmentation_variant": "exponential_first_kind",
            }
        except Exception as e:
            logger.debug(f"Exponential first kind case failed: {e}")
            return None

    def _create_oscillatory_first_kind(
        self, a: float, b: float
    ) -> dict[str, Any] | None:
        """Create oscillatory kernel 1st kind: ∫sin(x-t)*u(t)dt = sin(2x)."""
        try:
            return {
                "u": "",  # No analytical solution - requires numerical regularization
                "f": "sin(2*x)",
                "kernel": "sin(x - t)",
                "lambda_val": "0",
                "a": str(a),
                "b": str(b),
                "has_solution": True,  # Solution exists but unstable without regularization
                "equation_type": "fredholm_first_kind",
                "equation_form": "∫K(x,t)u(t)dt = f(x)",
                "is_ill_posed": True,
                "requires_regularization": True,
                "recommended_methods": [
                    "Tikhonov with L² penalty",
                    "Total variation regularization",
                    "Wavelet-based methods",
                ],
                "solution_type": "regularized",
                "edge_case": "ill_posed",
                "reason": "Oscillatory kernel creates severely ill-conditioned problem",
                "warning": "Extremely unstable - discretization errors magnified exponentially",
                "notes": "Consider using specialized methods for convolution-type equations",
                "regularization_param": self.alpha,
                "augmented": True,
                "augmentation_type": "ill_posed",
                "augmentation_variant": "oscillatory_first_kind",
            }
        except Exception as e:
            logger.debug(f"Oscillatory first kind case failed: {e}")
            return None
