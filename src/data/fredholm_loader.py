"""
Specialized loader for the Fredholm-LLM dataset.

Handles the specific CSV schema from the Fredholm-LLM project:
https://github.com/alirezaafzalaghaei/Fredholm-LLM

Dataset DOI: 10.5281/zenodo.16784707
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.dataset_fetcher import download_fredholm_dataset
from src.utils.logging_utils import get_logger
from src.utils.math_utils import fix_implicit_multiplication

logger = get_logger(__name__)


class ExpressionType(Enum):
    """Type of mathematical expression."""

    REAL_VALUE = "real_value"
    POLYNOMIAL = "polynomial"
    TRIGONOMETRIC = "trigonometric"
    HYPERBOLIC = "hyperbolic"
    EXPONENTIAL = "exponential"


@dataclass
class FredholmEquation:
    """
    Represents a single Fredholm integral equation of the second kind.

    The equation form is:
        u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)

    Attributes:
        u: The solution function u(x) as a string expression.
        f: The right-hand side function f(x) as a string expression.
        kernel: The kernel function K(x, t) as a string expression.
        lambda_val: The λ parameter as a string (may be numeric or expression).
        a: Lower integration bound as a string.
        b: Upper integration bound as a string.
        metadata: Additional metadata about the equation.
    """

    u: str
    f: str
    kernel: str
    lambda_val: str
    a: str
    b: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_csv_row(cls, row: dict[str, Any]) -> "FredholmEquation":
        """
        Create a FredholmEquation from a CSV row dictionary.

        Args:
            row: Dictionary with CSV column values.

        Returns:
            FredholmEquation instance.
        """
        # Extract core fields (handle column name variations)
        # Fix implicit multiplication in expressions (e.g., "2x" -> "2*x")
        u = fix_implicit_multiplication(str(row.get("u", row.get("U", ""))).strip())
        f = fix_implicit_multiplication(str(row.get("f", row.get("F", ""))).strip())
        kernel = fix_implicit_multiplication(str(row.get("kernel", row.get("Kernel", ""))).strip())
        lambda_val = str(
            row.get("lambda", row.get("Lambda", row.get("lmbda", "")))
        ).strip()
        a = str(row.get("a", row.get("A", "0"))).strip()
        b = str(row.get("b", row.get("B", "1"))).strip()

        # Extract metadata
        metadata = {}

        # Expression type flags for u
        if "u_is_polynomial" in row:
            metadata["u_is_polynomial"] = _parse_bool(row["u_is_polynomial"])
            metadata["u_is_trigonometric"] = _parse_bool(
                row.get("u_is_trigonometric", False)
            )
            metadata["u_is_hyperbolic"] = _parse_bool(row.get("u_is_hyperbolic", False))
            metadata["u_is_exponential"] = _parse_bool(
                row.get("u_is_exponential", False)
            )
            metadata["u_max_degree"] = _parse_number(row.get("u_max_degree", 1))
            metadata["u_type"] = _determine_expression_type(metadata, "u")
        else:
            # Infer expression type from the expression itself
            metadata["u_type"] = _infer_expression_type(u)

        # Expression type flags for f
        if "f_is_polynomial" in row:
            metadata["f_is_polynomial"] = _parse_bool(row["f_is_polynomial"])
            metadata["f_is_trigonometric"] = _parse_bool(
                row.get("f_is_trigonometric", False)
            )
            metadata["f_is_hyperbolic"] = _parse_bool(row.get("f_is_hyperbolic", False))
            metadata["f_is_exponential"] = _parse_bool(
                row.get("f_is_exponential", False)
            )
            metadata["f_max_degree"] = _parse_number(row.get("f_max_degree", 1))
            metadata["f_type"] = _determine_expression_type(metadata, "f")
        else:
            metadata["f_type"] = _infer_expression_type(f)

        # Expression type flags for kernel
        if "kernel_is_polynomial" in row:
            metadata["kernel_is_polynomial"] = _parse_bool(row["kernel_is_polynomial"])
            metadata["kernel_is_trigonometric"] = _parse_bool(
                row.get("kernel_is_trigonometric", False)
            )
            metadata["kernel_is_hyperbolic"] = _parse_bool(
                row.get("kernel_is_hyperbolic", False)
            )
            metadata["kernel_is_exponential"] = _parse_bool(
                row.get("kernel_is_exponential", False)
            )
            metadata["kernel_max_degree"] = _parse_number(
                row.get("kernel_max_degree", 1)
            )
            metadata["kernel_type"] = _determine_expression_type(metadata, "kernel")
        else:
            metadata["kernel_type"] = _infer_expression_type(kernel)

        # Expression type flags for lambda (lambda is usually just numeric)
        if "lambda_is_polynomial" in row:
            metadata["lambda_is_polynomial"] = _parse_bool(row["lambda_is_polynomial"])
            metadata["lambda_is_trigonometric"] = _parse_bool(
                row.get("lambda_is_trigonometric", False)
            )
            metadata["lambda_is_hyperbolic"] = _parse_bool(
                row.get("lambda_is_hyperbolic", False)
            )
            metadata["lambda_is_exponential"] = _parse_bool(
                row.get("lambda_is_exponential", False)
            )
            metadata["lambda_max_degree"] = _parse_number(
                row.get("lambda_max_degree", 1)
            )
            metadata["lambda_type"] = _determine_expression_type(metadata, "lambda")

        return cls(
            u=u,
            f=f,
            kernel=kernel,
            lambda_val=lambda_val,
            a=a,
            b=b,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "u": self.u,
            "f": self.f,
            "kernel": self.kernel,
            "lambda": self.lambda_val,
            "a": self.a,
            "b": self.b,
            **self.metadata,
        }

    def to_equation_string(self, style: str = "symbolic") -> str:
        """
        Format the equation as a string.

        Args:
            style: Output style ('symbolic', 'latex', 'natural').

        Returns:
            Formatted equation string.
        """
        if style == "latex":
            return (
                f"u(x) - {self.lambda_val} \\int_{{{self.a}}}^{{{self.b}}} "
                f"{self.kernel} \\cdot u(t) \\, dt = {self.f}"
            )
        elif style == "natural":
            return (
                f"u(x) - {self.lambda_val} * integral from {self.a} to {self.b} "
                f"of {self.kernel} * u(t) dt = {self.f}"
            )
        else:  # symbolic
            return (
                f"u(x) - {self.lambda_val} * ∫_{self.a}^{self.b} "
                f"{self.kernel} * u(t) dt = {self.f}"
            )

    @property
    def solution(self) -> str:
        """Return the solution u(x)."""
        return self.u


def _parse_bool(value: Any) -> bool:
    """Parse a boolean value from various formats."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def _parse_number(value: Any) -> float:
    """Parse a numeric value."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 1.0


def _infer_expression_type(expr: str) -> ExpressionType:
    """
    Infer the expression type from the expression string.

    Uses pattern matching to detect mathematical functions.

    Args:
        expr: Expression string to analyze.

    Returns:
        Inferred ExpressionType.
    """
    import re

    expr_lower = expr.lower()

    # Check for exponential (exp function)
    if re.search(r"\bexp\s*\(", expr_lower):
        return ExpressionType.EXPONENTIAL

    # Check for hyperbolic functions
    if re.search(r"\b(sinh|cosh|tanh|coth|sech|csch)\s*\(", expr_lower):
        return ExpressionType.HYPERBOLIC

    # Check for trigonometric functions
    if re.search(r"\b(sin|cos|tan|cot|sec|csc|asin|acos|atan)\s*\(", expr_lower):
        return ExpressionType.TRIGONOMETRIC

    # Check for polynomial (contains x or t with power, or just variables)
    if re.search(r"\b[xt]\b|\*\*", expr_lower):
        return ExpressionType.POLYNOMIAL

    # Default to real value (constant)
    return ExpressionType.REAL_VALUE


def _determine_expression_type(metadata: dict[str, Any], prefix: str) -> ExpressionType:
    """Determine the expression type from metadata flags."""
    if metadata.get(f"{prefix}_is_exponential"):
        return ExpressionType.EXPONENTIAL
    if metadata.get(f"{prefix}_is_trigonometric"):
        return ExpressionType.TRIGONOMETRIC
    if metadata.get(f"{prefix}_is_hyperbolic"):
        return ExpressionType.HYPERBOLIC
    if metadata.get(f"{prefix}_is_polynomial"):
        return ExpressionType.POLYNOMIAL
    return ExpressionType.REAL_VALUE


class FredholmDatasetLoader:
    """
    Specialized loader for the Fredholm-LLM dataset.

    Supports automatic download from Zenodo and provides rich filtering options.
    """

    def __init__(
        self,
        data_path: Path | str | None = None,
        auto_download: bool = True,
        variant: str = "sample",
        max_samples: int | None = None,
    ) -> None:
        """
        Initialize the Fredholm dataset loader.

        Args:
            data_path: Path to the dataset CSV. If None, will download.
            auto_download: Automatically download if file not found.
            variant: Dataset variant ('full' or 'sample') for auto-download.
            max_samples: Maximum number of samples to load.
        """
        self.data_path = Path(data_path) if data_path else None
        self.auto_download = auto_download
        self.variant = variant
        self.max_samples = max_samples

        self._equations: list[FredholmEquation] | None = None
        self._df: pd.DataFrame | None = None

    def _ensure_data_available(self) -> Path:
        """Ensure dataset is available, downloading if necessary."""
        if self.data_path and self.data_path.exists():
            return self.data_path

        if not self.auto_download:
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Set auto_download=True or download manually."
            )

        logger.info(f"Dataset not found. Downloading '{self.variant}' variant...")
        self.data_path = download_fredholm_dataset(variant=self.variant)
        return self.data_path

    def load(self) -> list[FredholmEquation]:
        """
        Load the dataset.

        Returns:
            List of FredholmEquation objects.
        """
        if self._equations is not None:
            return self._equations

        path = self._ensure_data_available()
        logger.info(f"Loading Fredholm dataset from {path}")

        # Load CSV
        self._df = pd.read_csv(path)
        logger.info(f"Loaded {len(self._df)} rows from CSV")

        # Convert to FredholmEquation objects with progress bar
        from rich.progress import (BarColumn, Progress, SpinnerColumn,
                                   TextColumn, TimeRemainingColumn)
        
        self._equations = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("  Processing equations...", total=len(self._df))
            
            for _, row in self._df.iterrows():
                try:
                    eq = FredholmEquation.from_csv_row(row.to_dict())
                    self._equations.append(eq)
                except Exception as e:
                    logger.warning(f"Skipping invalid row: {e}")
                progress.update(task, advance=1)

        if self.max_samples:
            self._equations = self._equations[: self.max_samples]

        logger.info(f"Loaded {len(self._equations)} valid equations")
        return self._equations

    def load_as_dicts(self) -> list[dict[str, Any]]:
        """Load dataset as list of dictionaries."""
        return [eq.to_dict() for eq in self.load()]

    def load_as_dataframe(self) -> pd.DataFrame:
        """Load dataset as pandas DataFrame."""
        if self._df is None:
            self.load()
        assert self._df is not None  # Guaranteed after load()
        return self._df

    def iterate(self, batch_size: int = 1) -> Iterator[list[FredholmEquation]]:
        """
        Iterate over equations in batches.

        Args:
            batch_size: Number of equations per batch.

        Yields:
            Batches of FredholmEquation objects.
        """
        equations = self.load()
        for i in range(0, len(equations), batch_size):
            yield equations[i : i + batch_size]

    def filter(
        self,
        u_type: ExpressionType | None = None,
        f_type: ExpressionType | None = None,
        kernel_type: ExpressionType | None = None,
        max_degree: int | None = None,
        **kwargs: Any,
    ) -> list[FredholmEquation]:
        """
        Filter equations by criteria.

        Args:
            u_type: Filter by solution expression type.
            f_type: Filter by f expression type.
            kernel_type: Filter by kernel expression type.
            max_degree: Filter by maximum degree across all expressions.
            **kwargs: Additional metadata filters.

        Returns:
            Filtered list of equations.
        """
        equations = self.load()
        filtered = equations

        if u_type:
            filtered = [eq for eq in filtered if eq.metadata.get("u_type") == u_type]

        if f_type:
            filtered = [eq for eq in filtered if eq.metadata.get("f_type") == f_type]

        if kernel_type:
            filtered = [
                eq for eq in filtered if eq.metadata.get("kernel_type") == kernel_type
            ]

        if max_degree is not None:
            filtered = [
                eq
                for eq in filtered
                if max(
                    eq.metadata.get("u_max_degree", 0),
                    eq.metadata.get("f_max_degree", 0),
                    eq.metadata.get("kernel_max_degree", 0),
                )
                <= max_degree
            ]

        for key, value in kwargs.items():
            filtered = [eq for eq in filtered if eq.metadata.get(key) == value]

        return filtered

    def get_statistics(self) -> dict[str, Any]:
        """
        Compute dataset statistics.

        Returns:
            Dictionary with dataset statistics.
        """
        equations = self.load()

        stats = {
            "total_equations": len(equations),
            "u_types": {},
            "f_types": {},
            "kernel_types": {},
        }

        for expr_type in ExpressionType:
            stats["u_types"][expr_type.value] = len(
                [eq for eq in equations if eq.metadata.get("u_type") == expr_type]
            )
            stats["f_types"][expr_type.value] = len(
                [eq for eq in equations if eq.metadata.get("f_type") == expr_type]
            )
            stats["kernel_types"][expr_type.value] = len(
                [eq for eq in equations if eq.metadata.get("kernel_type") == expr_type]
            )

        return stats

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.load())

    def __getitem__(self, idx: int) -> FredholmEquation:
        """Get equation by index."""
        return self.load()[idx]


def load_fredholm_dataset(
    path: Path | str | None = None,
    variant: str = "sample",
    max_samples: int | None = None,
    auto_download: bool = True,
) -> list[FredholmEquation]:
    """
    Convenience function to load the Fredholm-LLM dataset.

    Args:
        path: Path to dataset CSV. Downloads if None.
        variant: Dataset variant for download ('full' or 'sample').
        max_samples: Maximum samples to load.
        auto_download: Auto-download if not found.

    Returns:
        List of FredholmEquation objects.
    """
    loader = FredholmDatasetLoader(
        data_path=path,
        variant=variant,
        max_samples=max_samples,
        auto_download=auto_download,
    )
    return loader.load()

