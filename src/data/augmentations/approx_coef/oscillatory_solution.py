"""
Highly oscillatory solution augmentation strategy.

Generates equations where the solution u(x) oscillates rapidly,
requiring very fine sampling to avoid aliasing (Nyquist criterion).
"""

from typing import Any

import sympy as sp

from src.data.augmentations.base import BaseAugmentation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OscillatorySolutionAugmentation(BaseAugmentation):
    """
    Generate highly oscillatory solution cases.

    Mathematical Background:
        Unlike "approximate_only" which has oscillatory KERNEL K(x,t),
        this strategy creates cases where the SOLUTION u(x) oscillates rapidly.

        Example: u(x) = sin(ωx) for large ω

        If ω = 100π, the solution has 50 complete cycles in [0,1].
        Standard quadrature with N=20 points will completely miss the oscillations
        (Nyquist: need at least 2 points per cycle → need N ≥ 100).

    Nyquist-Shannon Sampling Theorem:
        To capture oscillations of frequency f, need sampling rate ≥ 2f.
        For ω = 100π, maximum frequency ≈ 50 Hz, so need ≥100 samples in [0,1].

    The Challenge for LLM:
        Model must:
        - Recognize solution has high-frequency components
        - Estimate oscillation frequency
        - Recommend fine enough discretization (Nyquist criterion)
        - Suggest specialized quadrature (Filon, Levin methods)

    Contrast with Other Strategies:
        - **approximate_only**: Oscillatory KERNEL → hard to integrate K(x,t)
        - **oscillatory_solution**: Oscillatory SOLUTION → hard to sample u(x)
        - Both require fine sampling but for different reasons

    Physical Context:
        - High-frequency wave propagation
        - Rapidly varying electromagnetic fields
        - Quantum oscillations at high energy

    Label:
        {
            "has_solution": true,
            "solution_type": "approx_coef",
            "edge_case": "oscillatory_solution",
            "oscillation_frequency": omega,
            "nyquist_samples_required": int,
            "recommended_methods": ["filon", "levin", "fine_sampling"]
        }
    """

    def __init__(
        self, base_frequency: float = 10.0, num_sample_points: int = 100
    ) -> None:
        """
        Initialize oscillatory solution augmentation.

        Args:
            base_frequency: Base oscillation frequency (multiplied for variants)
            num_sample_points: Minimum number of sample points (should satisfy Nyquist)
        """
        self.base_frequency = base_frequency
        self.num_sample_points = num_sample_points

    @property
    def strategy_name(self) -> str:
        return "oscillatory_solution"

    @property
    def description(self) -> str:
        return "Solutions with rapid oscillations"

    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate oscillatory solution cases."""
        results = []

        try:
            # Extract base parameters
            a = float(sp.sympify(item.get("a", "0")))
            b = float(sp.sympify(item.get("b", "1")))
            lambda_val = float(
                sp.sympify(item.get("lambda", item.get("lambda_val", "1")))
            )

            # Reduce lambda to prevent equation instability
            lambda_scaled = lambda_val * 0.1

            # Case 1: High-frequency sine solution
            omega1 = self.base_frequency * 10  # ω = 100 for base_frequency=10
            cycles1 = omega1 / (2 * sp.pi)  # Number of cycles
            nyquist1 = int(2 * cycles1 * 1.5)  # 1.5x oversampling for safety

            case1 = {
                "u": f"sin({omega1} * pi * x)",  # Rapidly oscillating solution
                "f": f"sin({omega1} * pi * x)",  # Match solution
                "kernel": "x * t",  # Simple kernel (smooth)
                "lambda_val": str(lambda_scaled),
                "lambda_val": str(lambda_scaled),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "approx_coef",
                "edge_case": "oscillatory_solution",
                "oscillation_type": "sinusoidal",
                "oscillation_frequency": omega1 * sp.pi,
                "angular_frequency": f"{omega1}*pi",
                "num_cycles_in_domain": float(cycles1),
                "nyquist_samples_required": nyquist1,
                "sampling_rate_needed": 2 * omega1 * sp.pi,
                "recommended_methods": [
                    "filon_quadrature",
                    "levin_collocation",
                    "fine_uniform_sampling",
                    "adaptive_sampling",
                ],
                "numerical_challenge": f"Solution oscillates {cycles1:.1f} times, standard quadrature undersamples",
                "augmented": True,
                "augmentation_type": "oscillatory_solution",
                "augmentation_variant": "high_frequency_sine",
            }

            results.append(case1)

            # Case 2: Modulated oscillation - amplitude varies with position
            # u(x) = (1 + x) * sin(ωx) - harder because amplitude changes
            omega2 = self.base_frequency * 8
            cycles2 = omega2 / (2 * sp.pi)
            nyquist2 = int(2 * cycles2 * 1.5)

            case2 = {
                "u": f"(1 + x) * sin({omega2} * pi * x)",  # Amplitude modulated
                "f": f"(1 + x) * sin({omega2} * pi * x)",
                "kernel": "1 + x + t",
                "lambda_val": str(lambda_scaled * 0.8),
                "lambda_val": str(lambda_scaled * 0.8),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "approx_coef",
                "edge_case": "oscillatory_solution",
                "oscillation_type": "amplitude_modulated",
                "oscillation_frequency": omega2 * sp.pi,
                "modulation": "linear amplitude (1+x)",
                "num_cycles_in_domain": float(cycles2),
                "nyquist_samples_required": nyquist2,
                "recommended_methods": [
                    "adaptive_quadrature",
                    "filon_with_modulation",
                    "variable_step_runge_kutta",
                ],
                "numerical_challenge": "Both amplitude and phase vary → adaptive sampling essential",
                "augmented": True,
                "augmentation_type": "oscillatory_solution",
                "augmentation_variant": "amplitude_modulated",
            }
            results.append(case2)

            # Case 3: Multiple frequency components (beating)
            # u(x) = sin(ω₁x) + sin(ω₂x) where ω₂ ≈ ω₁
            # Creates beating pattern with envelope frequency |ω₁-ω₂|/2
            omega3a = self.base_frequency * 10
            omega3b = self.base_frequency * 12  # Close frequency
            max_freq = max(omega3a, omega3b)
            nyquist3 = int(2 * max_freq / (2 * sp.pi) * 1.5)

            case3 = {
                "u": f"sin({omega3a} * pi * x) + sin({omega3b} * pi * x)",  # Two frequencies
                "f": f"sin({omega3a} * pi * x) + sin({omega3b} * pi * x)",
                "kernel": "sin(x) * cos(t)",
                "lambda_val": str(lambda_scaled * 0.5),
                "lambda_val": str(lambda_scaled * 0.5),
                "a": str(a),
                "b": str(b),
                "has_solution": True,
                "solution_type": "approx_coef",
                "edge_case": "oscillatory_solution",
                "oscillation_type": "multi_frequency_beating",
                "frequencies": [omega3a * sp.pi, omega3b * sp.pi],
                "beat_frequency": abs(omega3a - omega3b) * sp.pi / 2,
                "nyquist_samples_required": nyquist3,
                "spectrum_complexity": "two main frequencies plus beat frequency",
                "recommended_methods": [
                    "fourier_based_quadrature",
                    "spectral_collocation",
                    "fine_uniform_sampling",
                ],
                "numerical_challenge": "Multiple frequencies create complex interference pattern",
                "augmented": True,
                "augmentation_type": "oscillatory_solution",
                "augmentation_variant": "multi_frequency_beating",
            }
            results.append(case3)

        except Exception as e:
            logger.warning(f"Failed to generate oscillatory solution case: {e}")

        return results

    def _generate_fine_mesh(
        self, a: float, b: float, n_points: int
    ) -> list[list[float]]:
        """
        Generate fine uniform mesh for oscillatory solutions.

        Args:
            a: Lower bound
            b: Upper bound
            n_points: Number of points (should satisfy Nyquist)

        Returns:
            List of [x, t] sample points
        """
        import numpy as np

        x_points = np.linspace(a, b, n_points)

        # Generate sample points (subsample to keep manageable size)
        samples = []
        step = max(1, n_points // self.num_sample_points)
        for i, xi in enumerate(x_points[::step]):
            for j, ti in enumerate(x_points[::step]):
                samples.append([float(xi), float(ti)])
                if len(samples) >= self.num_sample_points:
                    break
            if len(samples) >= self.num_sample_points:
                break

        return samples[: self.num_sample_points]

    def _generate_adaptive_mesh(
        self, a: float, b: float, n_base: int
    ) -> list[list[float]]:
        """
        Generate adaptive mesh with more points where needed.

        For modulated oscillations, concentrate points where amplitude varies most.
        """
        import numpy as np

        # Use graded mesh with more points near boundaries
        # where modulation (1+x) has larger gradient
        n = n_base // 2
        # Uniform in middle
        middle = np.linspace(a + 0.1 * (b - a), b - 0.1 * (b - a), n // 2)
        # Denser at boundaries
        left = np.linspace(a, a + 0.1 * (b - a), n // 4)
        right = np.linspace(b - 0.1 * (b - a), b, n // 4)

        x_points = np.sort(np.concatenate([left, middle, right]))

        # Generate samples
        samples = []
        step = max(1, len(x_points) // self.num_sample_points)
        for xi in x_points[::step]:
            for ti in x_points[::step]:
                samples.append([float(xi), float(ti)])
                if len(samples) >= self.num_sample_points:
                    break
            if len(samples) >= self.num_sample_points:
                break

        return samples[: self.num_sample_points]
