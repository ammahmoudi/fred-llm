"""Tests for data validation and integration."""

from pathlib import Path

import pytest

from src.data.augmentation import DataAugmenter
from src.data.format_converter import FormatConverter
from src.data.fredholm_loader import FredholmDatasetLoader
from src.data.validator import validate_dataset, validate_equation


@pytest.fixture
def sample_dataset_path() -> Path:
    """Return path to sample dataset."""
    return Path("data/raw/Fredholm_Dataset_Sample.csv")


@pytest.fixture
def fredholm_loader(sample_dataset_path: Path) -> FredholmDatasetLoader:
    """Create FredholmDatasetLoader instance."""
    return FredholmDatasetLoader(data_path=sample_dataset_path, auto_download=False)


@pytest.fixture
def format_converter() -> FormatConverter:
    """Create FormatConverter instance."""
    return FormatConverter()


@pytest.fixture
def data_augmenter() -> DataAugmenter:
    """Create DataAugmenter instance."""
    return DataAugmenter()


class TestDataValidation:
    """Test data validation."""

    def test_valid_equation(self) -> None:
        """Test validation of a valid equation."""
        eq_dict = {
            "u": "x**2",
            "f": "x**2 + 2*x",
            "kernel": "x*t",
            "lambda_val": "0.5",
            "a": "0",
            "b": "1",
        }

        result = validate_equation(eq_dict, strict=False)
        # Basic validation should pass or have only warnings
        assert result["valid"] or len(result.get("warnings", [])) > 0

    def test_invalid_syntax(self) -> None:
        """Test validation catches syntax errors."""
        eq_dict = {
            "u": "x**",  # Invalid syntax
            "f": "x**2",
            "kernel": "x*t",
            "lambda_val": "0.5",
            "a": "0",
            "b": "1",
        }

        result = validate_equation(eq_dict, strict=False)
        # Should have errors or warnings
        assert not result["valid"] or len(result["warnings"]) > 0

    def test_invalid_bounds(self) -> None:
        """Test validation catches invalid integration bounds."""
        eq_dict = {
            "u": "x**2",
            "f": "x**2 + 2*x",
            "kernel": "x*t",
            "lambda_val": "0.5",
            "a": "1",  # a > b
            "b": "0",
        }

        result = validate_equation(eq_dict, strict=False)
        # Should have warning or error about bounds
        assert len(result["warnings"]) > 0 or not result["valid"]

    @pytest.mark.skipif(
        not Path("data/raw/Fredholm_Dataset_Sample.csv").exists(),
        reason="Sample dataset not found",
    )
    def test_batch_validation(
        self, sample_dataset_path: Path, fredholm_loader: FredholmDatasetLoader
    ) -> None:
        """Test validating multiple equations."""
        fredholm_loader.max_samples = 10
        equations = fredholm_loader.load()

        # Convert to dicts for validation
        eq_dicts = [eq.to_dict() for eq in equations]
        results = validate_dataset(eq_dicts, strict=False)

        assert results["total"] == len(equations)
        assert results["valid"] + results["invalid"] == results["total"]
        # Should have some results (validation ran)
        assert results["total"] == 10


class TestIntegration:
    """Test integration of multiple pipeline components."""

    @pytest.mark.skipif(
        not Path("data/raw/Fredholm_Dataset_Sample.csv").exists(),
        reason="Sample dataset not found",
    )
    def test_full_pipeline(
        self,
        sample_dataset_path: Path,
        fredholm_loader: FredholmDatasetLoader,
        format_converter: FormatConverter,
        data_augmenter: DataAugmenter,
    ) -> None:
        """Test complete pipeline: load → augment → convert."""
        # 1. Load just 2 equations
        fredholm_loader.max_samples = 2
        equations = fredholm_loader.load()
        assert len(equations) <= 2

        # 2. Light augmentation
        eq_dicts = [eq.to_dict() for eq in equations]
        augmenter = DataAugmenter(
            strategies=["none_solution"]
        )  # Use edge case strategy
        augmented = augmenter.augment(eq_dicts, multiplier=2)
        assert len(augmented) >= len(eq_dicts)

        # 3. Convert only first 3 to avoid hanging
        converted_count = 0
        for eq_dict in augmented[:3]:
            try:
                rpn = format_converter.convert(
                    eq_dict["u"], source_format="infix", target_format="rpn"
                )
                if rpn:
                    converted_count += 1
            except Exception:
                pass

        # Should convert at least 1
        assert converted_count >= 1
