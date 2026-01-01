"""
Dataset splitting utilities for Fred-LLM.

Uses scikit-learn's stratified splitting to maintain data balance across:
- Original vs augmented equations
- Solution types (exact, numerical, none, regularized, family)
- Edge case types (for augmented data)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def split_dataset(
    data: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.0,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split data into train/val/test sets with stratified splitting.

    Uses scikit-learn's train_test_split with stratification to maintain balance across:
    - Original vs augmented equations
    - Solution types (exact, numerical, none, regularized, family)
    - Edge case types (for augmented data)

    Args:
        data: List of equation dictionaries to split.
        train_ratio: Proportion of data for training (default: 0.8).
        val_ratio: Proportion of data for validation (default: 0.0, test will be 0.2).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        Tuple of (train, val, test) lists.

    Example:
        >>> data = load_dataset("data.csv")
        >>> train, val, test = split_dataset(data, train_ratio=0.8, val_ratio=0.0)
        >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        Train: 800, Val: 0, Test: 200
    """
    if len(data) == 0:
        logger.info("Split complete: empty dataset")
        return [], [], []

    # Validate ratios
    if train_ratio + val_ratio > 1.0:
        logger.warning(
            f"Invalid ratios: train={train_ratio} + val={val_ratio} = {train_ratio + val_ratio} > 1.0. "
            "Adjusting to fit."
        )
        total = train_ratio + val_ratio
        train_ratio = train_ratio / total * 0.9  # Leave 10% for test
        val_ratio = val_ratio / total * 0.9

    # Handle 100% train case
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        logger.info("All data assigned to train (100%)")
        return data, [], []

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data)

    # Fill missing columns with defaults
    if "augmented" not in df.columns:
        df["augmented"] = False
    if "solution_type" not in df.columns:
        df["solution_type"] = "exact"
    if "edge_case" not in df.columns:
        df["edge_case"] = ""

    # Create stratification key by combining:
    # 1. Whether augmented or original
    # 2. Solution type
    # 3. Edge case type (if augmented)
    df["_strat_key"] = (
        df["augmented"].astype(str)
        + "_"
        + df["solution_type"].astype(str)
        + "_"
        + df["edge_case"].astype(str)
    )

    # Calculate test ratio
    test_ratio = 1.0 - train_ratio - val_ratio

    logger.info(
        f"Splitting {len(df)} items with ratios: "
        f"train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}"
    )

    # Count strata
    strata_counts = df["_strat_key"].value_counts()
    logger.info(f"Found {len(strata_counts)} distinct strata")

    # Check if dataset is too small for splitting
    if len(df) == 1:
        logger.warning("Dataset has only 1 sample. Assigning to train.")
        train_df = df
        val_df = pd.DataFrame()
        test_df = pd.DataFrame()
    # Check if we have small strata that can't be stratified
    # sklearn requires at least 2 samples per stratum for stratification
    elif val_ratio > 0:
        min_stratum_size = strata_counts.min()
        can_stratify = min_stratum_size >= 2

        if not can_stratify:
            logger.warning(
                f"Some strata have only {min_stratum_size} sample(s). "
                "Falling back to non-stratified split for val/test."
            )

        # First split: train vs (val+test)
        temp_test_size = val_ratio + test_ratio

        try:
            train_df, temp_df = train_test_split(
                df,
                test_size=temp_test_size,
                random_state=seed,
                stratify=df["_strat_key"] if can_stratify else None,
            )

            # Second split: val vs test from temp
            val_size_from_temp = val_ratio / temp_test_size

            # Check if temp still has small strata
            temp_min_size = temp_df["_strat_key"].value_counts().min()
            can_stratify_temp = temp_min_size >= 2

            val_df, test_df = train_test_split(
                temp_df,
                test_size=1 - val_size_from_temp,
                random_state=seed,
                stratify=temp_df["_strat_key"] if can_stratify_temp else None,
            )
        except ValueError as e:
            # If stratification still fails, fall back to non-stratified
            logger.warning(f"Stratification failed: {e}. Using non-stratified split.")
            train_df, temp_df = train_test_split(
                df,
                test_size=temp_test_size,
                random_state=seed,
            )
            val_size_from_temp = val_ratio / temp_test_size
            val_df, test_df = train_test_split(
                temp_df,
                test_size=1 - val_size_from_temp,
                random_state=seed,
            )
    else:
        # Simple train/test split (no val)
        try:
            train_df, test_df = train_test_split(
                df,
                test_size=test_ratio,
                random_state=seed,
                stratify=df["_strat_key"],
            )
        except ValueError as e:
            logger.warning(f"Stratification failed: {e}. Using non-stratified split.")
            train_df, test_df = train_test_split(
                df,
                test_size=test_ratio,
                random_state=seed,
            )
        val_df = pd.DataFrame()  # Empty val set

    # Remove the temporary stratification key and added columns
    columns_to_drop = ["_strat_key"]
    # Also remove columns that we added for stratification if they weren't in original data
    original_columns = set(data[0].keys()) if data else set()
    for col in ["augmented", "solution_type", "edge_case"]:
        if col in train_df.columns and col not in original_columns:
            columns_to_drop.append(col)

    train_df = train_df.drop(
        columns=[c for c in columns_to_drop if c in train_df.columns]
    )
    val_df = (
        val_df.drop(columns=[c for c in columns_to_drop if c in val_df.columns])
        if not val_df.empty
        else val_df
    )
    test_df = (
        test_df.drop(columns=[c for c in columns_to_drop if c in test_df.columns])
        if not test_df.empty
        else test_df
    )

    # Convert back to list of dicts
    train = train_df.to_dict("records")
    val = val_df.to_dict("records") if not val_df.empty else []
    test = test_df.to_dict("records")

    logger.info(
        f"Split complete: train={len(train)} ({100 * len(train) / len(data):.1f}%), "
        f"val={len(val)} ({100 * len(val) / len(data):.1f}%), "
        f"test={len(test)} ({100 * len(test) / len(data):.1f}%)"
    )

    return train, val, test


def get_split_statistics(
    train: list[dict], val: list[dict], test: list[dict]
) -> dict[str, dict]:
    """
    Get statistics about dataset splits for validation using pandas.

    Args:
        train: Training set.
        val: Validation set.
        test: Test set.

    Returns:
        Dictionary with statistics for each split.

    Example:
        >>> stats = get_split_statistics(train, val, test)
        >>> print(f"Train original ratio: {stats['train']['original_ratio']:.2%}")
    """

    def analyze_split(split: list[dict], name: str) -> dict:
        """Analyze a single split using pandas."""
        if not split:
            return {
                "name": name,
                "size": 0,
                "original_count": 0,
                "augmented_count": 0,
                "original_ratio": 0.0,
                "solution_types": {},
                "edge_cases": {},
            }

        df = pd.DataFrame(split)

        augmented_count = df.get("augmented", pd.Series([False])).sum()
        original_count = len(df) - augmented_count

        # Count solution types
        solution_types = (
            df.get("solution_type", pd.Series(["exact"] * len(df)))
            .value_counts()
            .to_dict()
        )

        # Count edge cases (only for augmented items)
        augmented_df = df[df.get("augmented", False)]
        edge_cases = {}
        if len(augmented_df) > 0 and "edge_case" in augmented_df.columns:
            edge_cases = (
                augmented_df["edge_case"]
                .replace("", pd.NA)
                .dropna()
                .value_counts()
                .to_dict()
            )

        return {
            "name": name,
            "size": len(split),
            "original_count": int(original_count),
            "augmented_count": int(augmented_count),
            "original_ratio": original_count / len(split) if split else 0.0,
            "solution_types": solution_types,
            "edge_cases": edge_cases,
        }

    return {
        "train": analyze_split(train, "train"),
        "val": analyze_split(val, "val"),
        "test": analyze_split(test, "test"),
    }
