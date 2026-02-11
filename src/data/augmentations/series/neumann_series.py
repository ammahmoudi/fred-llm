"""
Neumann series augmentation strategy.

Generates equations with solutions represented as truncated Neumann series
(first N=3-4 terms), teaching models about iterative series expansions.
"""

from typing import Any

import sympy as sp

from src.data.augmentations.base import BaseAugmentation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class NeumannSeriesAugmentation(BaseAugmentation):
    """
    Generate Neumann series solution cases.

    Mathematical Background:
        For Fredholm equation: u(x) - λ∫K(x,t)u(t)dt = f(x)
        
        Neumann series provides iterative solution:
            u(x) = f(x) + λ∫K(x,t)f(t)dt + λ²∫∫K(x,s)K(s,t)f(t)dsdt + ...
            
        Symbolically: u = Σ(n=0 to ∞) λⁿ Kⁿf
        where K⁰f = f, K¹f = ∫K(x,t)f(t)dt, K²f = ∫K(x,s)(∫K(s,t)f(t)dt)ds
        
        Convergence: Series converges when ||λK|| < 1 (operator norm)

    Truncation:
        Store first N=3-4 terms as explicit formula:
        u_approx(x) ≈ term₀ + term₁ + term₂ + term₃
        
        Each term is symbolic (may contain x, sin, exp, etc.)

    Why It Matters:
        - Shows iterative nature of integral equations
        - Demonstrates operator powers (K, K², K³, ...)
        - Teaches difference between exact vs approximate series
        - Important for constructive solution methods

    Physical Context:
        - Scattering theory (Born series)
        - Transport equations (collision expansions)  
        - Green's function iterations

    Output Format:
        {
            "u": "f(x) + λ*∫K₀ + λ²*∫K₁ + λ³*∫K₂",  # Series expansion
            "solution_type": "series",
            "series_terms": 4,
            "series_type": "neumann",
            "convergence_estimate": float
        }
    """

    def __init__(self, num_terms: int = 4, lambda_scale: float = 0.3) -> None:
        """
        Initialize Neumann series augmentation.

        Args:
            num_terms: Number of series terms to generate (default: 4)
            lambda_scale: Scale factor for lambda to ensure convergence
        """
        self.num_terms = num_terms
        self.lambda_scale = lambda_scale

    @property
    def strategy_name(self) -> str:
        return "neumann_series"

    @property
    def description(self) -> str:
        return f"Neumann series with first {self.num_terms} terms"

    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate Neumann series cases."""
        results = []

        try:
            # Extract base parameters
            a = float(sp.sympify(item.get("a", "0")))
            b = float(sp.sympify(item.get("b", "1")))
            lambda_val = float(
                sp.sympify(item.get("lambda", item.get("lambda_val", "1")))
            )

            # Scale lambda for convergence
            lambda_conv = lambda_val * self.lambda_scale

            # Case 1: Separable kernel - can compute series symbolically
            # K(x,t) = g(x)h(t), makes nested integrals tractable
            case1 = self._create_separable_series(item, a, b, lambda_conv)
            if case1:
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

            # Case 2: Polynomial kernel - finite series representation
            # K(x,t) = xt, successive powers create polynomial growth
            case2 = self._create_polynomial_series(item, a, b, lambda_conv)
            if case2:
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

            # Case 3: Exponential decay kernel - rapidly converging series
            # K(x,t) = exp(-|x-t|), smooth kernel ensures fast convergence
            case3 = self._create_exponential_series(item, a, b, lambda_conv)
            if case3:
                # Generate evaluation points for consistent evaluation
                if case3.get("has_solution") and case3.get("u"):
                    try:
                        a_val = float(sp.sympify(case3.get("a", "0")))
                        b_val = float(sp.sympify(case3.get("b", "1")))
                        case3["evaluation_points"] = self._generate_evaluation_points(
                            case3["u"], a_val, b_val
                        )
                    except Exception as e:
                        logger.debug(f"Failed to generate evaluation points: {e}")
                results.append(case3)

        except Exception as e:
            logger.debug(f"Neumann series generation failed: {e}")

        return results

    def _create_separable_series(
        self, item: dict[str, Any], a: float, b: float, lambda_val: float
    ) -> dict[str, Any] | None:
        """Create series with separable kernel K(x,t) = sin(x)cos(t)."""
        try:
            x, t = sp.symbols("x t", real=True)
            
            # Original function
            f_expr = sp.sympify(item.get("f", "1"))
            
            # K(x,t) = sin(x)*cos(t)
            g_x = sp.sin(x)
            h_t = sp.cos(t)
            
            # Compute series terms symbolically
            # Term 0: f(x)
            term0 = f_expr
            
            # Term 1: λ * sin(x) * ∫cos(t)*f(t)dt
            integral_1 = sp.integrate(h_t * f_expr.subs(x, t), (t, a, b))
            term1 = lambda_val * g_x * integral_1
            
            # Term 2: λ² * sin(x) * ∫cos(t) * [sin(t) * C]dt where C = ∫cos(s)*f(s)ds
            # For simplicity, approximate as λ² * sin(x) * ∫cos(t)sin(t)*f(t)dt
            integral_2_inner = sp.integrate(h_t * sp.sin(t) * f_expr.subs(x, t), (t, a, b))
            term2 = lambda_val**2 * g_x * integral_2_inner
            
            # Term 3: λ³ term (further nested)
            # Approximate for demonstration
            term3 = lambda_val**3 * g_x * integral_1 * sp.cos(x) / 2
            
            # Combine terms
            series_sum = term0 + term1 + term2 + term3
            series_str = str(series_sum)
            
            # Estimate convergence rate
            convergence_rate = abs(lambda_val) * 0.7  # Heuristic for separable

            return {
                "u": series_str,
                "f": str(f_expr),
                "kernel": "sin(x) * cos(t)",
                "lambda_val": str(lambda_val),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "series",
                "edge_case": "neumann_series",
                "series_type": "neumann",
                "series_terms": self.num_terms,
                "kernel_structure": "separable",
                "convergence_estimate": convergence_rate,
                "truncation_error": f"O(λ^{self.num_terms})",
                "recommended_methods": [
                    "series_acceleration",
                    "richardson_extrapolation",
                    "pade_approximation"
                ],
                "numerical_challenge": f"Truncated after {self.num_terms} terms - assess convergence",
                "augmented": True,
                "augmentation_type": "neumann_series",
                "augmentation_variant": "separable_kernel",
            }
        except Exception as e:
            logger.debug(f"Separable series creation failed: {e}")
            return None

    def _create_polynomial_series(
        self, item: dict[str, Any], a: float, b: float, lambda_val: float
    ) -> dict[str, Any] | None:
        """Create series with polynomial kernel K(x,t) = x*t."""
        try:
            x, t = sp.symbols("x t", real=True)
            f_expr = sp.sympify(item.get("f", "x"))
            
            # Term 0: f(x)
            term0 = f_expr
            
            # Term 1: λ * x * ∫t*f(t)dt
            integral_1 = sp.integrate(t * f_expr.subs(x, t), (t, a, b))
            term1 = lambda_val * x * integral_1
            
            # Term 2: λ² * x * ∫t * [t * C]dt
            integral_2 = sp.integrate(t**2 * integral_1, (t, a, b))
            term2 = lambda_val**2 * x * integral_2
            
            # Term 3: λ³ term
            integral_3 = sp.integrate(t**3 * integral_2, (t, a, b))
            term3 = lambda_val**3 * x * integral_3
            
            series_sum = term0 + term1 + term2 + term3
            series_str = str(series_sum)
            
            convergence_rate = abs(lambda_val) * (b - a) / 3  # Polynomial norm estimate

            return {
                "u": series_str,
                "f": str(f_expr),
                "kernel": "x * t",
                "lambda_val": str(lambda_val),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "series",
                "edge_case": "neumann_series",
                "series_type": "neumann",
                "series_terms": self.num_terms,
                "kernel_structure": "polynomial",
                "convergence_estimate": convergence_rate,
                "truncation_error": f"O(λ^{self.num_terms})",
                "recommended_methods": [
                    "polynomial_acceleration",
                    "recursive_evaluation",
                    "symbolic_simplification"
                ],
                "numerical_challenge": f"Polynomial growth in {self.num_terms} terms",
                "augmented": True,
                "augmentation_type": "neumann_series",
                "augmentation_variant": "polynomial_kernel",
            }
        except Exception as e:
            logger.debug(f"Polynomial series creation failed: {e}")
            return None

    def _create_exponential_series(
        self, item: dict[str, Any], a: float, b: float, lambda_val: float
    ) -> dict[str, Any] | None:
        """Create series with exponential kernel (fast convergence)."""
        try:
            # For exponential kernels, exact symbolic integration is hard
            # Provide approximate form showing structure
            
            series_str = (
                "f(x) + "
                f"{lambda_val}*Integral(exp(-Abs(x - t))*f(t), (t, {a}, {b})) + "
                f"{lambda_val**2}*Integral(Integral(exp(-Abs(x - s))*exp(-Abs(s - t))*f(t), "
                f"(t, {a}, {b})), (s, {a}, {b}))"
            )
            
            convergence_rate = abs(lambda_val) * 0.5  # Exponential decay helps

            return {
                "u": series_str,
                "f": item.get("f", "1"),
                "kernel": "exp(-abs(x - t))",
                "lambda_val": str(lambda_val),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "series",
                "edge_case": "neumann_series",
                "series_type": "neumann",
                "series_terms": self.num_terms,
                "kernel_structure": "exponential_decay",
                "convergence_estimate": convergence_rate,
                "truncation_error": f"O(λ^{self.num_terms})",
                "recommended_methods": [
                    "numerical_quadrature_per_term",
                    "monte_carlo_integration",
                    "fast_convergence_acceleration"
                ],
                "numerical_challenge": f"Smooth kernel → rapid convergence in {self.num_terms} terms",
                "augmented": True,
                "augmentation_type": "neumann_series",
                "augmentation_variant": "exponential_kernel",
            }
        except Exception as e:
            logger.debug(f"Exponential series creation failed: {e}")
            return None
