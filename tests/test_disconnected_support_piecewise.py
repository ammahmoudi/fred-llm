"""
Tests for disconnected_support augmentation Piecewise kernels.
"""

import sympy as sp

from src.data.augmentations.none_solution.disconnected_support import (
    DisconnectedSupportAugmentation,
)
from src.data.format_converter import FormatConverter


class TestDisconnectedSupportPiecewise:
    """Ensure Piecewise kernels are valid expressions."""

    def test_piecewise_kernels_parse(self):
        """Kernel strings should parse as SymPy Piecewise expressions."""
        aug = DisconnectedSupportAugmentation()
        item = {
            "a": "0",
            "b": "1",
            "lambda_val": "1",
            "f": "x",
        }

        results = aug.augment(item)
        assert len(results) == 2

        for case in results:
            kernel_expr = sp.sympify(case["kernel"])
            assert isinstance(kernel_expr, sp.Piecewise)

    def test_piecewise_kernel_latex_conversion(self):
        """Piecewise kernels should convert to LaTeX without errors."""
        aug = DisconnectedSupportAugmentation()
        item = {
            "a": "0",
            "b": "1",
            "lambda_val": "1",
            "f": "x",
        }

        results = aug.augment(item)
        converter = FormatConverter()

        for case in results:
            latex_str = converter.convert(case["kernel"], "infix", "latex")
            assert isinstance(latex_str, str)
            assert latex_str
