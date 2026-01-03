#!/usr/bin/env python3
"""
Run the adaptive pipeline with a configuration file.

Examples:
    # Full automation (raw data)
    python scripts/run_adaptive_pipeline.py configs/examples/full_automation.yaml

    # Partial automation (pre-split data)
    python scripts/run_adaptive_pipeline.py configs/examples/partial_automation.yaml

    # Manual control (pre-generated prompts)
    python scripts/run_adaptive_pipeline.py configs/examples/manual_control.yaml

    # Dry run (show what would happen)
    python scripts/run_adaptive_pipeline.py config.yaml --dry-run
"""

import argparse
from pathlib import Path

from src.adaptive_pipeline import AdaptivePipeline, load_adaptive_config
from src.utils.logging_utils import setup_logging


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run adaptive pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show execution plan without running",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    # Load and validate config
    print(f"Loading config: {args.config}")
    config = load_adaptive_config(args.config)

    # Create and run pipeline
    pipeline = AdaptivePipeline(config)
    results = pipeline.run(dry_run=args.dry_run)

    if not args.dry_run:
        print("\n✅ Pipeline complete!")
        print(f"Results saved to: {config.output.dir}")


if __name__ == "__main__":
    main()
