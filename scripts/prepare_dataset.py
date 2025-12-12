#!/usr/bin/env python3
"""
Dataset preparation script for Fred-LLM.

Prepares and preprocesses the FIE-500k dataset for training and evaluation.

Usage:
    python scripts/prepare_dataset.py --input data/raw/fie_500k.json --output data/processed/
"""

import argparse
import json
from pathlib import Path

# TODO: Import from src once package is installed
# from src.data import DataLoader, validate_equation
# from src.utils import get_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare FIE dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/fie_500k.json"),
        help="Input dataset path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/"),
        help="Output directory",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split into train/val/test sets",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate equations before processing",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process",
    )
    return parser.parse_args()


def load_raw_data(input_path: Path) -> list[dict]:
    """Load raw dataset."""
    print(f"Loading data from {input_path}")
    
    if not input_path.exists():
        print(f"Warning: Input file not found: {input_path}")
        return []
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    
    print(f"Loaded {len(data)} samples")
    return data


def validate_data(data: list[dict]) -> list[dict]:
    """Validate and filter data."""
    valid_data = []
    invalid_count = 0
    
    for item in data:
        # TODO: Use proper validation
        # result = validate_equation(item)
        # if result["valid"]:
        #     valid_data.append(item)
        # else:
        #     invalid_count += 1
        
        # Basic validation
        if "kernel" in item and "f" in item:
            valid_data.append(item)
        else:
            invalid_count += 1
    
    print(f"Validation: {len(valid_data)} valid, {invalid_count} invalid")
    return valid_data


def split_data(
    data: list[dict],
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split data into train/val/test sets."""
    import random
    
    random.seed(42)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:val_end]
    test_data = shuffled[val_end:]
    
    print(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    return train_data, val_data, test_data


def save_data(data: list[dict], output_path: Path) -> None:
    """Save processed data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(data)} samples to {output_path}")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Load data
    data = load_raw_data(args.input)
    
    if not data:
        print("No data to process. Creating sample data...")
        # Create sample data for testing
        data = [
            {
                "id": "sample_1",
                "kernel": "x*t",
                "f": "x",
                "lambda_val": 1.0,
                "domain": [0, 1],
                "solution": "3*x/2",
            },
            {
                "id": "sample_2",
                "kernel": "exp(x)*exp(t)",
                "f": "1",
                "lambda_val": 1.0,
                "domain": [0, 1],
                "solution": "1 + 2*(exp(1)-1)/(3-exp(2))*exp(x)",
            },
        ]
    
    # Limit samples if specified
    if args.max_samples:
        data = data[: args.max_samples]
    
    # Validate if requested
    if args.validate:
        data = validate_data(data)
    
    # Process and save
    args.output.mkdir(parents=True, exist_ok=True)
    
    if args.split:
        train_data, val_data, test_data = split_data(
            data, args.train_ratio, args.val_ratio
        )
        save_data(train_data, args.output / "train.json")
        save_data(val_data, args.output / "val.json")
        save_data(test_data, args.output / "test.json")
    else:
        save_data(data, args.output / "processed.json")
    
    print("Dataset preparation complete!")


if __name__ == "__main__":
    main()
