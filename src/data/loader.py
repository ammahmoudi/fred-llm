"""
Data loader for Fredholm integral equation datasets.

Supports multiple formats: JSON, JSONL, CSV, Parquet.
"""

import json
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Loader for Fredholm integral equation datasets."""

    def __init__(
        self,
        data_path: Path | str,
        format: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        """
        Initialize the data loader.

        Args:
            data_path: Path to the dataset file or directory.
            format: Data format (json, jsonl, csv, parquet). Auto-detected if None.
            max_samples: Maximum number of samples to load.
        """
        self.data_path = Path(data_path)
        self.format = format or self._detect_format()
        self.max_samples = max_samples

        self._data: list[dict[str, Any]] | None = None

    def _detect_format(self) -> str:
        """Detect data format from file extension."""
        suffix = self.data_path.suffix.lower()
        format_map = {
            ".json": "json",
            ".jsonl": "jsonl",
            ".csv": "csv",
            ".parquet": "parquet",
            ".pq": "parquet",
        }
        return format_map.get(suffix, "json")

    def load(self) -> list[dict[str, Any]]:
        """
        Load the complete dataset.

        Returns:
            List of equation dictionaries.
        """
        if self._data is not None:
            return self._data

        logger.info(f"Loading data from {self.data_path} (format: {self.format})")

        if not self.data_path.exists():
            logger.warning(f"Data file not found: {self.data_path}")
            return []

        if self.format == "json":
            self._data = self._load_json()
        elif self.format == "jsonl":
            self._data = self._load_jsonl()
        elif self.format == "csv":
            self._data = self._load_csv()
        elif self.format == "parquet":
            self._data = self._load_parquet()
        else:
            raise ValueError(f"Unknown format: {self.format}")

        if self.max_samples:
            self._data = self._data[: self.max_samples]

        logger.info(f"Loaded {len(self._data)} samples")
        return self._data

    def _load_json(self) -> list[dict[str, Any]]:
        """Load JSON file."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Handle both list and dict with 'data' key
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return data if isinstance(data, list) else [data]

    def _load_jsonl(self) -> list[dict[str, Any]]:
        """Load JSONL file."""
        data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _load_csv(self) -> list[dict[str, Any]]:
        """Load CSV file."""
        df = pd.read_csv(self.data_path)
        return df.to_dict("records")

    def _load_parquet(self) -> list[dict[str, Any]]:
        """Load Parquet file."""
        df = pd.read_parquet(self.data_path)
        return df.to_dict("records")

    def iterate(self, batch_size: int = 1) -> Iterator[list[dict[str, Any]]]:
        """
        Iterate over the dataset in batches.

        Args:
            batch_size: Number of samples per batch.

        Yields:
            Batches of equation dictionaries.
        """
        data = self.load()
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]

    def get_sample(self, idx: int) -> dict[str, Any]:
        """Get a single sample by index."""
        data = self.load()
        return data[idx]

    def filter(
        self,
        kernel_type: str | None = None,
        has_solution: bool | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Filter the dataset by criteria.

        Args:
            kernel_type: Filter by kernel type.
            has_solution: Filter by whether solution exists.
            **kwargs: Additional filter criteria.

        Returns:
            Filtered list of equations.
        """
        data = self.load()
        filtered = data

        if kernel_type:
            filtered = [d for d in filtered if d.get("kernel_type") == kernel_type]

        if has_solution is not None:
            filtered = [d for d in filtered if ("solution" in d) == has_solution]

        for key, value in kwargs.items():
            filtered = [d for d in filtered if d.get(key) == value]

        return filtered

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.load())

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get sample by index."""
        return self.get_sample(idx)


def load_dataset(
    path: Path | str,
    format: str | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """
    Convenience function to load a dataset.

    Args:
        path: Path to dataset file.
        format: Data format (auto-detected if None).
        max_samples: Maximum samples to load.

    Returns:
        List of equation dictionaries.
    """
    loader = DataLoader(path, format=format, max_samples=max_samples)
    return loader.load()
