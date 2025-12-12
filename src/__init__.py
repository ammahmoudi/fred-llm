"""
Fred-LLM: Solving Fredholm Integral Equations using LLMs

This package provides tools for solving and approximating Fredholm integral
equations of the second kind using large language models.

Equation form: u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from src.config import load_config
from src.main import FredLLMPipeline

__all__ = ["load_config", "FredLLMPipeline", "__version__"]
