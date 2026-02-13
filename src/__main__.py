"""
Entry point for running Fred-LLM as a module.

Usage:
    python -m src run --config config.yaml
    python -m src evaluate --input results.json
"""

import multiprocessing
import sys


def _setup_multiprocessing():
    """Configure multiprocessing for Windows compatibility."""
    if sys.platform == "win32":
        # On Windows, use 'spawn' method to avoid handle inheritance issues
        # This prevents "The handle is invalid" errors
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            # Already set, ignore
            pass


if __name__ == "__main__":
    _setup_multiprocessing()
    from src.cli import main

    main()
