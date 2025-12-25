"""Tests for data augmentation strategies."""

from pathlib import Path

import pytest

from src.data.augmentation import DataAugmenter
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
