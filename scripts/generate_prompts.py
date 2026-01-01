#!/usr/bin/env python3
"""
DEPRECATED: This script has been replaced by run_prompt_generation.py

For batch prompt generation, use the new tools listed below.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Create deprecation panel
console.print()
panel = Panel.fit(
    "[yellow]⚠️  This script is deprecated![/yellow]\n\n"
    "Please use the new prompt generation tools below.",
    title="⚠️  Deprecation Notice",
    border_style="yellow",
)
console.print(panel)

# Create alternatives table
table = Table(title="🚀 New Tools", show_header=True, header_style="bold cyan")
table.add_column("Option", style="cyan", width=30)
table.add_column("Command", style="white")

table.add_row(
    "1️⃣  Batch Generation Script",
    "python scripts/run_prompt_generation.py --input <path> --styles all"
)
table.add_row(
    "2️⃣  Python API",
    "from src.prompts import create_prompt_style, EquationData"
)
table.add_row(
    "3️⃣  Example Scripts",
    "python examples/prompts/simple_demo.py"
)
table.add_row(
    "4️⃣  CLI Command",
    "uv run python -m src.cli prompt generate <file.csv>"
)

console.print(table)
console.print()

sys.exit(1)
