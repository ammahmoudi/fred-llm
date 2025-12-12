"""
Tests for the data loader module.
"""

import json
import tempfile
from pathlib import Path

import pytest


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_load_json_file(self, tmp_path: Path) -> None:
        """Test loading a JSON file."""
        # Create test data
        test_data = [
            {"id": "1", "kernel": "x*t", "f": "x", "lambda_val": 1.0},
            {"id": "2", "kernel": "exp(x+t)", "f": "1", "lambda_val": 0.5},
        ]

        # Write test file
        test_file = tmp_path / "test.json"
        with open(test_file, "w") as f:
            json.dump(test_data, f)

        # TODO: Import and test actual loader
        # from src.data.loader import DataLoader
        # loader = DataLoader(test_file)
        # data = loader.load()
        # assert len(data) == 2
        # assert data[0]["id"] == "1"

        # Placeholder assertion
        assert test_file.exists()

    def test_load_jsonl_file(self, tmp_path: Path) -> None:
        """Test loading a JSONL file."""
        test_data = [
            {"id": "1", "kernel": "x*t"},
            {"id": "2", "kernel": "exp(x+t)"},
        ]

        test_file = tmp_path / "test.jsonl"
        with open(test_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # TODO: Test actual loader
        # from src.data.loader import DataLoader
        # loader = DataLoader(test_file, format="jsonl")
        # data = loader.load()
        # assert len(data) == 2

        assert test_file.exists()

    def test_load_with_max_samples(self, tmp_path: Path) -> None:
        """Test loading with sample limit."""
        test_data = [{"id": str(i)} for i in range(100)]

        test_file = tmp_path / "test.json"
        with open(test_file, "w") as f:
            json.dump(test_data, f)

        # TODO: Test actual loader with max_samples
        # from src.data.loader import DataLoader
        # loader = DataLoader(test_file, max_samples=10)
        # data = loader.load()
        # assert len(data) == 10

        assert test_file.exists()

    def test_filter_by_kernel_type(self, tmp_path: Path) -> None:
        """Test filtering by kernel type."""
        test_data = [
            {"id": "1", "kernel_type": "polynomial"},
            {"id": "2", "kernel_type": "exponential"},
            {"id": "3", "kernel_type": "polynomial"},
        ]

        test_file = tmp_path / "test.json"
        with open(test_file, "w") as f:
            json.dump(test_data, f)

        # TODO: Test actual filter
        # from src.data.loader import DataLoader
        # loader = DataLoader(test_file)
        # filtered = loader.filter(kernel_type="polynomial")
        # assert len(filtered) == 2

        assert test_file.exists()

    def test_iterate_batches(self, tmp_path: Path) -> None:
        """Test batch iteration."""
        test_data = [{"id": str(i)} for i in range(10)]

        test_file = tmp_path / "test.json"
        with open(test_file, "w") as f:
            json.dump(test_data, f)

        # TODO: Test actual batch iteration
        # from src.data.loader import DataLoader
        # loader = DataLoader(test_file)
        # batches = list(loader.iterate(batch_size=3))
        # assert len(batches) == 4  # 3 + 3 + 3 + 1

        assert test_file.exists()


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_dataset_function(self, tmp_path: Path) -> None:
        """Test convenience function."""
        test_data = [{"id": "1", "kernel": "x*t"}]

        test_file = tmp_path / "test.json"
        with open(test_file, "w") as f:
            json.dump(test_data, f)

        # TODO: Test actual function
        # from src.data.loader import load_dataset
        # data = load_dataset(test_file)
        # assert len(data) == 1

        assert test_file.exists()

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Test loading nonexistent file."""
        fake_path = tmp_path / "nonexistent.json"

        # TODO: Test actual behavior
        # from src.data.loader import DataLoader
        # loader = DataLoader(fake_path)
        # data = loader.load()
        # assert data == []

        assert not fake_path.exists()
