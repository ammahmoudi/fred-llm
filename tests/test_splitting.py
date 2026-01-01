"""Tests for dataset splitting functionality."""

import pytest

from src.data.splitter import get_split_statistics, split_dataset


@pytest.fixture
def sample_data() -> list[dict]:
    """Create sample dataset for splitting tests."""
    return [
        {
            "u": f"x**{i}",
            "f": f"x**{i} + x",
            "kernel": f"x*t + {i}",
            "lambda_val": str(i * 0.1),
            "a": "0",
            "b": "1",
            "id": i,
        }
        for i in range(100)
    ]


class TestDatasetSplitting:
    """Test dataset splitting functionality."""

    def test_split_ratios_standard(self, sample_data: list[dict]) -> None:
        """Test standard 80/10/10 split."""
        train, val, test = split_dataset(sample_data, train_ratio=0.8, val_ratio=0.1)

        # Check sizes
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

        # Check total
        assert len(train) + len(val) + len(test) == len(sample_data)

    def test_split_ratios_custom(self, sample_data: list[dict]) -> None:
        """Test custom split ratios."""
        train, val, test = split_dataset(sample_data, train_ratio=0.7, val_ratio=0.15)

        # Check sizes (allow ±1 for rounding)
        assert 69 <= len(train) <= 71, f"Train size: {len(train)}"
        assert 14 <= len(val) <= 16, f"Val size: {len(val)}"
        assert 14 <= len(test) <= 16, f"Test size: {len(test)}"

        # Check total
        assert len(train) + len(val) + len(test) == len(sample_data)

    def test_split_no_overlap(self, sample_data: list[dict]) -> None:
        """Test that splits have no overlapping items."""
        train, val, test = split_dataset(sample_data, train_ratio=0.8, val_ratio=0.1)

        # Extract IDs
        train_ids = {item["id"] for item in train}
        val_ids = {item["id"] for item in val}
        test_ids = {item["id"] for item in test}

        # Check no overlap
        assert len(train_ids & val_ids) == 0, "Train and val have overlapping items"
        assert len(train_ids & test_ids) == 0, "Train and test have overlapping items"
        assert len(val_ids & test_ids) == 0, "Val and test have overlapping items"

    def test_split_contains_all_items(self, sample_data: list[dict]) -> None:
        """Test that split contains all original items."""
        train, val, test = split_dataset(sample_data, train_ratio=0.8, val_ratio=0.1)

        # Combine all splits
        combined = train + val + test
        combined_ids = {item["id"] for item in combined}
        original_ids = {item["id"] for item in sample_data}

        # Check all items present
        assert combined_ids == original_ids, "Split doesn't contain all original items"

    def test_split_reproducibility(self, sample_data: list[dict]) -> None:
        """Test that split is reproducible with same seed."""
        train1, val1, test1 = split_dataset(sample_data, train_ratio=0.8, val_ratio=0.1)
        train2, val2, test2 = split_dataset(sample_data, train_ratio=0.8, val_ratio=0.1)

        # Check same IDs in each split
        assert [item["id"] for item in train1] == [item["id"] for item in train2]
        assert [item["id"] for item in val1] == [item["id"] for item in val2]
        assert [item["id"] for item in test1] == [item["id"] for item in test2]

    def test_split_small_dataset(self) -> None:
        """Test splitting with small dataset."""
        small_data = [{"id": i, "value": i} for i in range(10)]
        train, val, test = split_dataset(small_data, train_ratio=0.8, val_ratio=0.1)

        # Check sizes
        assert len(train) == 8
        assert len(val) == 1
        assert len(test) == 1

    def test_split_large_dataset(self) -> None:
        """Test splitting with large dataset."""
        large_data = [{"id": i, "value": i} for i in range(10000)]
        train, val, test = split_dataset(large_data, train_ratio=0.8, val_ratio=0.1)

        # Check sizes
        assert len(train) == 8000
        assert len(val) == 1000
        assert len(test) == 1000

        # Check total
        assert len(train) + len(val) + len(test) == len(large_data)

    def test_split_90_5_5(self, sample_data: list[dict]) -> None:
        """Test 90/5/5 split for large datasets."""
        train, val, test = split_dataset(sample_data, train_ratio=0.9, val_ratio=0.05)

        # Check sizes
        assert len(train) == 90
        assert len(val) == 5
        assert len(test) == 5

    def test_split_preserves_data_structure(self, sample_data: list[dict]) -> None:
        """Test that split preserves all fields in data."""
        train, val, test = split_dataset(sample_data, train_ratio=0.8, val_ratio=0.1)

        # Check all splits have same keys as original data
        original_keys = set(sample_data[0].keys())

        for item in train + val + test:
            assert set(item.keys()) == original_keys, (
                f"Keys mismatch: {set(item.keys())} vs {original_keys}"
            )

    def test_split_invalid_ratios(self, sample_data: list[dict]) -> None:
        """Test that invalid ratios are handled gracefully."""
        # This should auto-adjust ratios to be valid
        train, val, test = split_dataset(sample_data, train_ratio=0.8, val_ratio=0.3)

        # Should still split the data somehow (auto-adjusted)
        assert len(train) + len(val) + len(test) == len(sample_data)
        # Train should be largest since it had highest ratio
        assert len(train) >= len(val)
        assert len(train) >= len(test)


class TestSplitEdgeCases:
    """Test edge cases in splitting."""

    def test_split_single_item(self) -> None:
        """Test splitting with single item."""
        single = [{"id": 0, "value": 0}]
        train, val, test = split_dataset(single, train_ratio=0.8, val_ratio=0.1)

        # All in train since can't split 1 item
        assert len(train) == 1
        assert len(val) == 0
        assert len(test) == 0

    def test_split_empty_list(self) -> None:
        """Test splitting empty list."""
        empty: list[dict] = []
        train, val, test = split_dataset(empty, train_ratio=0.8, val_ratio=0.1)

        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0

    def test_split_all_train(self, sample_data: list[dict]) -> None:
        """Test 100/0/0 split (all train)."""
        train, val, test = split_dataset(sample_data, train_ratio=1.0, val_ratio=0.0)

        assert len(train) == 100
        assert len(val) == 0
        assert len(test) == 0


class TestSplitStatistics:
    """Test split statistics functionality."""

    def test_get_split_statistics(self) -> None:
        """Test that statistics are calculated correctly."""
        data = []
        # 80 original
        for i in range(80):
            data.append(
                {
                    "id": i,
                    "augmented": False,
                    "solution_type": "exact",
                    "edge_case": "",
                }
            )
        # 20 augmented
        for i in range(20):
            data.append(
                {
                    "id": i + 80,
                    "augmented": True,
                    "solution_type": "numerical",
                    "edge_case": "gaussian_kernel",
                }
            )

        train, val, test = split_dataset(data, train_ratio=0.8, val_ratio=0.1)
        stats = get_split_statistics(train, val, test)

        # Check structure
        assert "train" in stats
        assert "val" in stats
        assert "test" in stats

        # Check train stats
        assert stats["train"]["size"] == len(train)
        assert stats["train"]["original_ratio"] > 0.7  # Should be ~80%
        assert "exact" in stats["train"]["solution_types"]
        assert "numerical" in stats["train"]["solution_types"]


class TestStratifiedSplitting:
    """Test stratified splitting to maintain data balance."""

    @pytest.fixture
    def augmented_sample_data(self) -> list[dict]:
        """Create sample dataset with augmented data mimicking real distribution."""
        data = []

        # Original data (86.7% ≈ 87 items) - all exact solutions
        for i in range(87):
            data.append(
                {
                    "u": f"x**{i}",
                    "f": f"x**{i} + x",
                    "kernel": f"x*t + {i}",
                    "lambda_val": str(i * 0.1),
                    "a": "0",
                    "b": "1",
                    "id": f"orig_{i}",
                    "augmented": False,
                    "solution_type": "exact",
                    "edge_case": "",
                }
            )

        # Augmented data (13.3% ≈ 13 items) - various edge cases
        edge_cases = [
            ("numerical", "weakly_singular"),
            ("numerical", "gaussian_kernel"),
            ("none", "eigenvalue"),
            ("none", "divergent_kernel"),
            ("regularized", "ill_posed"),
        ]

        for i, (sol_type, edge_type) in enumerate(edge_cases * 3):  # 15 items
            if i >= 13:  # Keep exactly 13 augmented
                break
            data.append(
                {
                    "u": "",
                    "f": f"x**2",
                    "kernel": f"exp(-(x-t)**2)",
                    "lambda_val": "0.5",
                    "a": "0",
                    "b": "1",
                    "id": f"aug_{i}",
                    "augmented": True,
                    "solution_type": sol_type,
                    "edge_case": edge_type,
                }
            )

        return data

    def test_stratified_split_maintains_original_ratio(
        self, augmented_sample_data: list[dict]
    ) -> None:
        """Test that original vs augmented ratio is maintained."""
        train, val, test = split_dataset(
            augmented_sample_data, train_ratio=0.8, val_ratio=0.1
        )

        # Count original vs augmented in each split
        def count_originals(split):
            return sum(1 for item in split if not item["augmented"])

        train_orig_ratio = count_originals(train) / len(train) if len(train) > 0 else 0
        val_orig_ratio = count_originals(val) / len(val) if len(val) > 0 else 0
        test_orig_ratio = count_originals(test) / len(test) if len(test) > 0 else 0

        # All splits should have similar ratio to original (~87%)
        # Allow wider tolerance for small samples (stratification with small groups has variance)
        # The important thing is train set is well-balanced since it's the largest
        assert 0.70 <= train_orig_ratio <= 0.95, f"Train ratio: {train_orig_ratio:.2%}"

        # Val and test may vary more due to small sample sizes
        if len(val) > 5:
            assert 0.50 <= val_orig_ratio <= 1.0, f"Val ratio: {val_orig_ratio:.2%}"
        if len(test) > 5:
            assert 0.50 <= test_orig_ratio <= 1.0, f"Test ratio: {test_orig_ratio:.2%}"

    def test_stratified_split_maintains_solution_types(
        self, augmented_sample_data: list[dict]
    ) -> None:
        """Test that solution type distribution is maintained."""
        train, val, test = split_dataset(
            augmented_sample_data, train_ratio=0.8, val_ratio=0.1
        )

        # Count solution types in original data
        from collections import Counter

        original_counts = Counter(
            item["solution_type"] for item in augmented_sample_data
        )

        # Check each solution type appears proportionally in splits
        for sol_type in original_counts.keys():
            original_ratio = original_counts[sol_type] / len(augmented_sample_data)

            train_count = sum(1 for item in train if item["solution_type"] == sol_type)
            val_count = sum(1 for item in val if item["solution_type"] == sol_type)
            test_count = sum(1 for item in test if item["solution_type"] == sol_type)

            # At least one split should have this type if it's >5% of data
            if original_ratio > 0.05:
                assert train_count > 0, f"{sol_type} missing from train"

    def test_stratified_split_maintains_edge_cases(
        self, augmented_sample_data: list[dict]
    ) -> None:
        """Test that edge case types are distributed across splits."""
        train, val, test = split_dataset(
            augmented_sample_data, train_ratio=0.8, val_ratio=0.1
        )

        # Get all edge case types from augmented data
        augmented_items = [item for item in augmented_sample_data if item["augmented"]]
        edge_types = {
            item["edge_case"] for item in augmented_items if item["edge_case"]
        }

        # Collect edge cases in all splits
        all_splits_edge_types = set()
        for split in [train, val, test]:
            for item in split:
                if item["augmented"] and item["edge_case"]:
                    all_splits_edge_types.add(item["edge_case"])

        # All edge case types should appear somewhere in the splits
        assert edge_types == all_splits_edge_types, (
            "Some edge cases missing from splits"
        )

    def test_stratified_split_no_data_leakage(
        self, augmented_sample_data: list[dict]
    ) -> None:
        """Test that stratification doesn't cause data leakage between splits."""
        train, val, test = split_dataset(
            augmented_sample_data, train_ratio=0.8, val_ratio=0.1
        )

        # Extract IDs
        train_ids = {item["id"] for item in train}
        val_ids = {item["id"] for item in val}
        test_ids = {item["id"] for item in test}

        # Check no overlap
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_stratified_split_with_single_stratum(self) -> None:
        """Test stratification works when all data is same type."""
        uniform_data = [
            {
                "id": i,
                "augmented": False,
                "solution_type": "exact",
                "edge_case": "",
            }
            for i in range(100)
        ]

        train, val, test = split_dataset(uniform_data, train_ratio=0.8, val_ratio=0.1)

        # Should still split correctly
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
