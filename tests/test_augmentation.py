"""Tests for data augmentation strategies."""

from pathlib import Path

import pytest

from src.data.augmentation import DataAugmenter
from src.data.augmentations import (
    ApproximateOnlyAugmentation,
    IllPosedAugmentation,
    NoSolutionAugmentation,
)
from src.data.fredholm_loader import FredholmDatasetLoader


@pytest.fixture
def sample_dataset_path() -> Path:
    """Return path to sample dataset."""
    return Path("data/raw/Fredholm_Dataset_Sample.csv")


@pytest.fixture
def fredholm_loader(sample_dataset_path: Path) -> FredholmDatasetLoader:
    """Create FredholmDatasetLoader instance."""
    return FredholmDatasetLoader(data_path=sample_dataset_path, auto_download=False)


@pytest.fixture
def data_augmenter() -> DataAugmenter:
    """Create DataAugmenter instance."""
    return DataAugmenter()


class TestDataAugmentation:
    """Test data augmentation strategies."""

    @pytest.mark.skipif(
        not Path("data/raw/Fredholm_Dataset_Sample.csv").exists(),
        reason="Sample dataset not found",
    )
    def test_substitute_augmentation(
        self,
        sample_dataset_path: Path,
        fredholm_loader: FredholmDatasetLoader,
        data_augmenter: DataAugmenter,
    ) -> None:
        """Test variable substitution augmentation."""
        fredholm_loader.max_samples = 2
        equations = fredholm_loader.load()
        eq_dicts = [eq.to_dict() for eq in equations]
        augmenter = DataAugmenter(strategies=["substitute"])
        augmented = augmenter.augment(eq_dicts, multiplier=3)

        # Should have more equations
        assert len(augmented) > len(eq_dicts)

    @pytest.mark.skipif(
        not Path("data/raw/Fredholm_Dataset_Sample.csv").exists(),
        reason="Sample dataset not found",
    )
    def test_scale_augmentation(
        self,
        sample_dataset_path: Path,
        fredholm_loader: FredholmDatasetLoader,
        data_augmenter: DataAugmenter,
    ) -> None:
        """Test lambda coefficient scaling augmentation."""
        fredholm_loader.max_samples = 2
        equations = fredholm_loader.load()
        eq_dicts = [eq.to_dict() for eq in equations]
        augmenter = DataAugmenter(strategies=["scale"])
        augmented = augmenter.augment(eq_dicts, multiplier=2)

        assert len(augmented) > len(eq_dicts)

    @pytest.mark.skipif(
        not Path("data/raw/Fredholm_Dataset_Sample.csv").exists(),
        reason="Sample dataset not found",
    )
    def test_shift_augmentation(
        self,
        sample_dataset_path: Path,
        fredholm_loader: FredholmDatasetLoader,
        data_augmenter: DataAugmenter,
    ) -> None:
        """Test domain shift augmentation."""
        fredholm_loader.max_samples = 2
        equations = fredholm_loader.load()
        eq_dicts = [eq.to_dict() for eq in equations]
        augmenter = DataAugmenter(strategies=["shift"])
        augmented = augmenter.augment(eq_dicts, multiplier=2)

        assert len(augmented) > len(eq_dicts)

    @pytest.mark.skipif(
        not Path("data/raw/Fredholm_Dataset_Sample.csv").exists(),
        reason="Sample dataset not found",
    )
    def test_compose_augmentation(
        self,
        sample_dataset_path: Path,
        fredholm_loader: FredholmDatasetLoader,
        data_augmenter: DataAugmenter,
    ) -> None:
        """Test kernel composition augmentation."""
        fredholm_loader.max_samples = 2
        equations = fredholm_loader.load()
        eq_dicts = [eq.to_dict() for eq in equations]
        augmenter = DataAugmenter(strategies=["compose"])
        augmented = augmenter.augment(eq_dicts, multiplier=2)

        assert len(augmented) > len(eq_dicts)

    @pytest.mark.skipif(
        not Path("data/raw/Fredholm_Dataset_Sample.csv").exists(),
        reason="Sample dataset not found",
    )
    def test_combined_augmentation(
        self,
        sample_dataset_path: Path,
        fredholm_loader: FredholmDatasetLoader,
        data_augmenter: DataAugmenter,
    ) -> None:
        """Test combining multiple augmentation strategies."""
        fredholm_loader.max_samples = 3
        equations = fredholm_loader.load()
        eq_dicts = [eq.to_dict() for eq in equations]

        # Apply multiple strategies
        augmenter = DataAugmenter(strategies=["substitute", "scale"])
        augmented = augmenter.augment(eq_dicts, multiplier=3)

        # Should have significantly more equations
        expansion_ratio = len(augmented) / len(equations)
        assert expansion_ratio >= 2.0

    @pytest.mark.skipif(
        not Path("data/raw/Fredholm_Dataset_Sample.csv").exists(),
        reason="Sample dataset not found",
    )
    def test_augmentation_preserves_structure(
        self,
        sample_dataset_path: Path,
        fredholm_loader: FredholmDatasetLoader,
    ) -> None:
        """Test that augmentation preserves equation structure."""
        fredholm_loader.max_samples = 1
        equations = fredholm_loader.load()
        eq_dicts = [eq.to_dict() for eq in equations]

        augmenter = DataAugmenter(strategies=["scale"])
        augmented = augmenter.augment(eq_dicts, multiplier=2)

        # Check that augmented equations have required fields
        for aug_eq in augmented:
            assert "u" in aug_eq
            assert "f" in aug_eq
            assert "kernel" in aug_eq
            assert "a" in aug_eq
            assert "b" in aug_eq


class TestEdgeCaseAugmentations:
    """Test edge case augmentation strategies."""

    def test_no_solution_augmentation(self) -> None:
        """Test no-solution (singular) case generation."""
        augmenter = NoSolutionAugmentation()

        # Create a sample equation
        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "1",
            "lambda_val": "1",
            "a": "0",
            "b": "1",
        }

        # Generate no-solution cases
        no_sol_cases = augmenter.augment(sample_eq)

        # Should generate 3 cases (constant, separable, symmetric kernels)
        assert len(no_sol_cases) == 3

        # All should be marked as no solution
        for case in no_sol_cases:
            assert case["has_solution"] is False
            assert case["solution_type"] == "none"
            assert "reason" in case
            assert "Fredholm Alternative" in case["reason"]
            assert case["augmented"] is True
            assert case["augmentation_type"] == "no_solution"
            assert case["edge_case"] == "no_solution"
            # Solution should be "None" for no-solution cases
            assert case["u"] == "None"

    def test_approximate_only_augmentation(self) -> None:
        """Test approximate-only (numerical) case generation."""
        augmenter = ApproximateOnlyAugmentation(num_sample_points=5)

        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "0.5",
            "lambda_val": "0.5",
            "a": "0",
            "b": "1",
        }

        # Generate approximate-only cases
        approx_cases = augmenter.augment(sample_eq)

        # Should generate 3 cases (Gaussian, exponential, sinc kernels)
        assert len(approx_cases) == 3

        # All should require numerical methods
        for case in approx_cases:
            assert case["has_solution"] is True
            assert case["solution_type"] == "numerical"
            assert "numerical_method" in case
            assert "sample_points" in case
            assert "sample_values" in case
            assert len(case["sample_points"]) == 5
            assert len(case["sample_values"]) == 5
            assert case["augmented"] is True
            assert case["augmentation_type"] == "approximate_only"
            assert case["edge_case"] == "approximate_only"
            assert "reason" in case
            # Solution should be "Numerical" for approximate-only cases
            assert case["u"] == "Numerical"

    def test_ill_posed_augmentation(self) -> None:
        """Test ill-posed (1st kind) case generation."""
        augmenter = IllPosedAugmentation(num_sample_points=5, regularization_param=0.01)

        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "1",
            "lambda_val": "1",
            "a": "0",
            "b": "1",
        }

        # Generate ill-posed cases
        ill_posed_cases = augmenter.augment(sample_eq)

        # Should generate 3 cases (simple, exponential, oscillatory)
        assert len(ill_posed_cases) == 3

        # All should be ill-posed first kind
        for case in ill_posed_cases:
            assert case["equation_type"] == "fredholm_first_kind"
            assert case["is_ill_posed"] is True
            assert case["requires_regularization"] is True
            assert "recommended_methods" in case
            assert len(case["recommended_methods"]) > 0
            assert case["solution_type"] == "regularized"
            assert case["augmented"] is True
            assert case["augmentation_type"] == "ill_posed"
            assert case["edge_case"] == "ill_posed"
            assert "reason" in case
            assert "warning" in case
            assert case["regularization_param"] == 0.01
            # Lambda should be N/A or 0 for first kind
            assert case["lambda"] == "N/A"
            assert case["lambda_val"] == "0"

    def test_edge_case_strategies_with_augmenter(self) -> None:
        """Test using edge case strategies through DataAugmenter."""
        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "1",
            "lambda_val": "1",
            "a": "0",
            "b": "1",
        }

        # Test no_solution strategy
        augmenter = DataAugmenter(strategies=["no_solution"])
        augmented = augmenter.augment([sample_eq], multiplier=2)
        assert any(eq.get("edge_case") == "no_solution" for eq in augmented)

        # Test approximate_only strategy
        augmenter = DataAugmenter(strategies=["approximate_only"])
        augmented = augmenter.augment([sample_eq], multiplier=2)
        assert any(eq.get("edge_case") == "approximate_only" for eq in augmented)

        # Test ill_posed strategy
        augmenter = DataAugmenter(strategies=["ill_posed"])
        augmented = augmenter.augment([sample_eq], multiplier=2)
        assert any(eq.get("edge_case") == "ill_posed" for eq in augmented)

    def test_combined_basic_and_edge_strategies(self) -> None:
        """Test combining basic and edge case strategies."""
        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "1",
            "lambda_val": "1",
            "a": "0",
            "b": "1",
        }

        # Combine basic and edge case strategies
        augmenter = DataAugmenter(
            strategies=["substitute", "scale", "no_solution", "approximate_only"]
        )
        # Use higher multiplier to ensure all strategies get applied
        augmented = augmenter.augment([sample_eq], multiplier=15)

        # Should have mix of regular and edge cases
        regular_cases = [eq for eq in augmented if not eq.get("edge_case")]
        edge_cases = [eq for eq in augmented if eq.get("edge_case")]

        assert len(regular_cases) > 0
        assert len(edge_cases) > 0
        assert len(augmented) > 1

    def test_no_solution_eigenvalue_detection(self) -> None:
        """Test that no-solution cases correctly identify eigenvalues."""
        augmenter = NoSolutionAugmentation()

        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "1",
            "lambda": "1",
            "lambda_val": "1",
            "a": "0",
            "b": "1",
        }

        cases = augmenter.augment(sample_eq)

        # First case should be constant kernel with eigenvalue 1/(b-a) = 1
        constant_kernel_case = [
            c
            for c in cases
            if c.get("augmentation_variant") == "constant_kernel_eigenvalue"
        ][0]
        assert constant_kernel_case["kernel"] == "1"
        assert float(constant_kernel_case["lambda"]) == 1.0  # 1/(1-0)

    def test_approximate_only_sample_points(self) -> None:
        """Test that approximate-only cases generate valid sample points."""
        augmenter = ApproximateOnlyAugmentation(num_sample_points=10)

        sample_eq = {
            "u": "x",
            "f": "1",
            "kernel": "x*t",
            "lambda": "0.5",
            "lambda_val": "0.5",
            "a": "0",
            "b": "1",
        }

        cases = augmenter.augment(sample_eq)

        for case in cases:
            # Sample points should be within domain
            assert all(0 <= x <= 1 for x in case["sample_points"])
            # Sample values should be finite numbers
            assert all(abs(val) < 1e6 for val in case["sample_values"])
            # Should have correct number of points
            assert len(case["sample_points"]) == 10
