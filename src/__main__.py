"""
Entry point for running Fred-LLM as a module.

Usage:
    python -m src run --config config.yaml
    python -m src evaluate --input results.json
"""

if __name__ == "__main__":
    from src.cli import main

    main()
