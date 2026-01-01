"""
Mixed Volterra-Fredholm augmentation strategy.

Generates equations that are part Volterra (upper limit depends on x)
and part Fredholm (fixed limits), requiring hybrid solution methods.
"""

from typing import Any

import sympy as sp

from src.data.augmentations.base import BaseAugmentation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MixedTypeAugmentation(BaseAugmentation):
    """
    Generate mixed Volterra-Fredholm type equations.

    Mathematical Background:
        **Fredholm equation**: u(x) - λ∫ₐᵇ K(x,t) u(t) dt = f(x)
            - Upper limit b is constant (fixed)
            - Integral operator is compact
        
        **Volterra equation**: u(x) - λ∫ₐˣ K(x,t) u(t) dt = f(x)
            - Upper limit is x (variable)
            - Causal structure (only depends on past)
        
        **Mixed type**: Kernel behaves differently in different regions:
            K(x,t) = { K₁(x,t)  if t ≤ x  (Volterra part)
                     { K₂(x,t)  if t > x   (Fredholm part)
        
        Or split integral:
            u(x) - λ∫ₐˣ K₁(x,t)u(t)dt - λ∫ₓᵇ K₂(x,t)u(t)dt = f(x)

    The Challenge for LLM:
        Model must:
        - Recognize the equation is not purely Fredholm or Volterra
        - Identify the split point (usually x)
        - Choose appropriate solution method (iterative vs spectral)
        - Understand causality structure

    Why It Matters:
        - Volterra equations can be solved by marching (initial value problem)
        - Fredholm equations require boundary value problem methods
        - Mixed type needs hybrid approach

    Physical Context:
        - Hereditary mechanics (memory effects)
        - Viscoelasticity (stress depends on strain history + environment)
        - Option pricing (path-dependent + market conditions)

    Label:
        {
            "has_solution": true,
            "solution_type": "numerical",
            "edge_case": "mixed_type",
            "equation_type": "volterra_fredholm_mixed",
            "causal_structure": "partial",
            "recommended_methods": ["hybrid_method", "marching_plus_iteration"]
        }
    """

    @property
    def strategy_name(self) -> str:
        return "mixed_type"

    @property
    def description(self) -> str:
        return "Mixed Volterra-Fredholm equations"

    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate mixed type cases."""
        results = []

        try:
            # Extract base parameters
            a = float(sp.sympify(item.get("a", "0")))
            b = float(sp.sympify(item.get("b", "1")))
            lambda_val = float(sp.sympify(item.get("lambda", item.get("lambda_val", "1"))))

            # Case 1: Piecewise kernel - different behavior for t < x and t > x
            # K(x,t) = { t      if t ≤ x  (Volterra - causal)
            #          { x      if t > x  (Fredholm - non-causal)
            case1 = {
                "u": item["u"],  # Solution exists but needs hybrid method
                "f": item["f"],
                "kernel": "t if t <= x else x",  # Piecewise definition
                "kernel_description": "K(x,t) = t for t≤x (Volterra part), x for t>x (Fredholm part)",
                "lambda_val": str(lambda_val * 0.5),
                "lambda_val": str(lambda_val * 0.5),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "numerical",
                "edge_case": "mixed_type",
                "equation_type": "volterra_fredholm_mixed",
                "split_point": "x",
                "causal_structure": "partial",
                "volterra_region": f"[{a}, x]",
                "fredholm_region": f"[x, {b}]",
                "mathematical_form": "u(x) - λ∫ₐˣ t·u(t)dt - λ∫ₓᵇ x·u(t)dt = f(x)",
                "recommended_methods": [
                    "hybrid_method",
                    "marching_plus_boundary",
                    "domain_decomposition"
                ],
                "numerical_challenge": "Combine causal (Volterra) and acausal (Fredholm) parts",
                "augmented": True,
                "augmentation_type": "mixed_type",
                "augmentation_variant": "piecewise_split",
            }
            results.append(case1)

            # Case 2: Smooth transition kernel
            # K(x,t) = tanh((x-t)/ε) transitions from -1 to +1 at x=t
            # Behaves like Volterra for t << x, Fredholm for t >> x
            epsilon = 0.1
            case2 = {
                "u": item["u"],
                "f": item["f"],
                "kernel": f"tanh((x - t) / {epsilon})",  # Smooth transition
                "kernel_description": f"K(x,t) = tanh((x-t)/{epsilon}) smoothly transitions at x=t",
                "lambda_val": str(lambda_val * 0.3),
                "lambda_val": str(lambda_val * 0.3),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "numerical",
                "edge_case": "mixed_type",
                "equation_type": "volterra_fredholm_smooth_mixed",
                "transition_width": epsilon,
                "causal_structure": "approximate",
                "mathematical_explanation": "tanh smoothly varies from -1 (past) to +1 (future)",
                "recommended_methods": [
                    "adaptive_quadrature",
                    "implicit_marching",
                    "collocation"
                ],
                "numerical_challenge": "Handle smooth but steep transition region",
                "augmented": True,
                "augmentation_type": "mixed_type",
                "augmentation_variant": "smooth_transition",
            }
            results.append(case2)

            # Case 3: Explicit two-integral form
            # Split the integral explicitly into causal and non-causal parts
            # More representative of real mixed-type equations
            case3 = {
                "u": item["u"],
                "f": item["f"],
                "kernel": "x*t",  # Simple separable kernel
                "kernel_split": "Implicit split: ∫ₐˣ x*t*u(t)dt + ∫ₓᵇ x*t*u(t)dt",
                "lambda_val": str(lambda_val * 0.4),
                "lambda_val": str(lambda_val * 0.4),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "numerical",
                "edge_case": "mixed_type",
                "equation_type": "volterra_fredholm_explicit_split",
                "split_point": "x",
                "causal_structure": "explicit",
                "integral_form": "Two separate integrals with different limits",
                "mathematical_form": f"u(x) - λ∫ₐˣ x·t·u(t)dt - λ∫ₓᵇ x·t·u(t)dt = f(x)",
                "recommended_methods": [
                    "sequential_marching",
                    "block_iteration",
                    "shooting_method"
                ],
                "numerical_challenge": "Efficiently couple Volterra marching with Fredholm iteration",
                "solution_strategy": "1. March from a to x (Volterra), 2. Iterate on [x,b] (Fredholm)",
                "augmented": True,
                "augmentation_type": "mixed_type",
                "augmentation_variant": "explicit_two_integral",
            }
            results.append(case3)

        except Exception as e:
            logger.warning(f"Failed to generate mixed type case: {e}")

        return results
