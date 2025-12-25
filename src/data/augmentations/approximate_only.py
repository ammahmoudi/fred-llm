"""
Approximate-only augmentation strategy.

Generates equations with no symbolic solution - only numerical approximations possible.
Uses kernels without closed-form antiderivatives.
"""

from typing import Any

import numpy as np
import sympy as sp
from scipy import integrate

from src.data.augmentations.base import BaseAugmentation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ApproximateOnlyAugmentation(BaseAugmentation):
    """
    Generate approximate-only cases with no symbolic solution.
    
    Category 2: The "Approximate Only" Case
    
    The Logic:
        Use a kernel that has no symbolic antiderivative, such as:
        - K(x,t) = exp(-(x² + t²))  (Gaussian)
        - K(x,t) = exp(-|x-t|)      (Exponential decay)
        - K(x,t) = sin(x*t) / (x*t) (Sinc-like)
        
    The Problem:
        SymPy will fail to provide a symbolic answer like "sin(x)".
        Integration requires numerical methods (scipy, quadrature).
        
    Label:
        {
            "solution_type": "numerical",
            "numerical_method": "quadrature",
            "sample_points": [x0, x1, ..., xn],
            "sample_values": [u(x0), u(x1), ..., u(xn)]
        }
        
    This forces the LLM to:
        1. Recognize no symbolic solution exists
        2. Use numerical methods (Neumann series, quadrature)
        3. Return approximation with confidence bounds
    """

    def __init__(self, num_sample_points: int = 10):
        """
        Initialize approximate-only augmentation.
        
        Args:
            num_sample_points: Number of points to sample for numerical solution.
        """
        self.num_sample_points = num_sample_points

    @property
    def strategy_name(self) -> str:
        return "approximate_only"

    @property
    def description(self) -> str:
        return "Generate cases requiring numerical approximation (no symbolic solution)"

    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate approximate-only cases."""
        results = []
        
        try:
            a = float(sp.sympify(item.get("a", "0")))
            b = float(sp.sympify(item.get("b", "1")))
            
            # Case 1: Gaussian kernel K(x,t) = exp(-(x²+t²))
            case1 = self._create_gaussian_kernel_case(a, b)
            if case1:
                results.append(case1)
            
            # Case 2: Exponential decay K(x,t) = exp(-|x-t|)
            case2 = self._create_exponential_decay_case(a, b)
            if case2:
                results.append(case2)
            
            # Case 3: Non-integrable kernel K(x,t) = sin(x*t) / (1 + x*t)
            case3 = self._create_sinc_kernel_case(a, b)
            if case3:
                results.append(case3)
                
        except Exception as e:
            logger.debug(f"Approximate-only augmentation failed: {e}")
        
        return results

    def _create_gaussian_kernel_case(self, a: float, b: float) -> dict[str, Any] | None:
        """Create case with Gaussian kernel."""
        try:
            # For simplicity, use f(x) = 1 (constant RHS)
            # Numerical solution via fixed-point iteration could be attempted
            lambda_val = 0.5
            
            # Generate sample points
            x_points = np.linspace(a, b, self.num_sample_points)
            
            # Approximate solution using simple iterative method
            # u(x) ≈ f(x) + λ∫K(x,t)f(t)dt for first iteration
            u_approx = []
            for x in x_points:
                # First-order approximation: u ≈ f(x) + λ∫K(x,t)f(t)dt
                def kernel(t):
                    return np.exp(-(x**2 + t**2))
                integral, _ = integrate.quad(kernel, a, b)
                u_val = 1.0 + lambda_val * integral * 1.0  # f(t) = 1
                u_approx.append(float(u_val))
            
            return {
                "u": "Numerical",  # No symbolic form
                "f": "1",
                "kernel": "exp(-(x**2 + t**2))",
                "lambda": str(lambda_val),
                "lambda_val": str(lambda_val),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "numerical",
                "numerical_method": "fixed_point_iteration",
                "sample_points": [float(x) for x in x_points],
                "sample_values": u_approx,
                "edge_case": "approximate_only",
                "reason": "Gaussian kernel has no symbolic antiderivative",
                "augmented": True,
                "augmentation_type": "approximate_only",
                "augmentation_variant": "gaussian_kernel",
            }
        except Exception as e:
            logger.debug(f"Gaussian kernel case failed: {e}")
            return None

    def _create_exponential_decay_case(self, a: float, b: float) -> dict[str, Any] | None:
        """Create case with exponential decay kernel."""
        try:
            lambda_val = 0.3
            x_points = np.linspace(a, b, self.num_sample_points)
            
            # Use f(x) = x
            u_approx = []
            for x in x_points:
                def kernel(t):
                    return np.exp(-abs(x - t))
                def integrand(t):
                    return kernel(t) * t  # f(t) = t
                integral, _ = integrate.quad(integrand, a, b)
                u_val = x + lambda_val * integral
                u_approx.append(float(u_val))
            
            return {
                "u": "Numerical",
                "f": "x",
                "kernel": "exp(-abs(x - t))",
                "lambda": str(lambda_val),
                "lambda_val": str(lambda_val),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "numerical",
                "numerical_method": "quadrature",
                "sample_points": [float(x) for x in x_points],
                "sample_values": u_approx,
                "edge_case": "approximate_only",
                "reason": "Exponential decay kernel requires numerical integration",
                "augmented": True,
                "augmentation_type": "approximate_only",
                "augmentation_variant": "exponential_decay",
            }
        except Exception as e:
            logger.debug(f"Exponential decay case failed: {e}")
            return None

    def _create_sinc_kernel_case(self, a: float, b: float) -> dict[str, Any] | None:
        """Create case with sinc-like kernel."""
        try:
            # Avoid division by zero at origin
            if a <= 0 <= b:
                a = 0.1
            
            lambda_val = 0.2
            x_points = np.linspace(a, b, self.num_sample_points)
            
            # Use f(x) = cos(x)
            u_approx = []
            for x in x_points:
                def kernel(t):
                    product = x * t
                    if abs(product) < 1e-10:
                        return 1.0  # limit as x*t -> 0
                    return np.sin(product) / product
                
                def integrand(t):
                    return kernel(t) * np.cos(t)  # f(t) = cos(t)
                
                integral, _ = integrate.quad(integrand, a, b)
                u_val = np.cos(x) + lambda_val * integral
                u_approx.append(float(u_val))
            
            return {
                "u": "Numerical",
                "f": "cos(x)",
                "kernel": "sin(x*t) / (x*t)",
                "lambda": str(lambda_val),
                "lambda_val": str(lambda_val),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "numerical",
                "numerical_method": "quadrature",
                "sample_points": [float(x) for x in x_points],
                "sample_values": u_approx,
                "edge_case": "approximate_only",
                "reason": "Sinc-like kernel has no closed-form solution",
                "augmented": True,
                "augmentation_type": "approximate_only",
                "augmentation_variant": "sinc_kernel",
            }
        except Exception as e:
            logger.debug(f"Sinc kernel case failed: {e}")
            return None
