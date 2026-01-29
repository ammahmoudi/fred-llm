"""
Cost tracking utilities for LLM API calls.

Tracks and aggregates costs for OpenAI and OpenRouter API usage.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class CallCost:
    """Cost information for a single API call."""

    timestamp: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    total_tokens: int
    prompt_cost_usd: Decimal
    completion_cost_usd: Decimal
    cached_cost_usd: Decimal
    total_cost_usd: Decimal
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with Decimal as string."""
        d = asdict(self)
        # Convert Decimal to string for JSON serialization
        for key in [
            "prompt_cost_usd",
            "completion_cost_usd",
            "cached_cost_usd",
            "total_cost_usd",
        ]:
            d[key] = str(d[key])
        return d


@dataclass
class RunCostSummary:
    """Aggregated cost summary for a run."""

    run_id: str
    start_time: str
    end_time: str | None
    total_requests: int
    total_tokens: int
    total_cost_usd: Decimal
    provider_breakdown: dict[str, dict[str, Any]]
    model_breakdown: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with Decimal as string."""
        d = asdict(self)
        d["total_cost_usd"] = str(d["total_cost_usd"])
        # Convert Decimal in nested dicts
        for provider in d["provider_breakdown"].values():
            provider["total_cost_usd"] = str(provider["total_cost_usd"])
        for model in d["model_breakdown"].values():
            model["total_cost_usd"] = str(model["total_cost_usd"])
        return d


class CostTracker:
    """Track and aggregate API call costs."""

    def __init__(self, run_id: str | None = None):
        """
        Initialize cost tracker.

        Args:
            run_id: Unique identifier for this run. Auto-generated if not provided.
        """
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now().isoformat()
        self.calls: list[CallCost] = []

    def add_call(self, cost: CallCost) -> None:
        """Add a call cost to the tracker."""
        self.calls.append(cost)
        logger.debug(
            f"Tracked {cost.provider}/{cost.model}: "
            f"{cost.total_tokens} tokens, ${cost.total_cost_usd}"
        )

    def get_summary(self) -> RunCostSummary:
        """Get aggregated cost summary."""
        if not self.calls:
            return RunCostSummary(
                run_id=self.run_id,
                start_time=self.start_time,
                end_time=datetime.now().isoformat(),
                total_requests=0,
                total_tokens=0,
                total_cost_usd=Decimal("0"),
                provider_breakdown={},
                model_breakdown={},
            )

        # Aggregate by provider
        provider_breakdown: dict[str, dict[str, Any]] = {}
        for call in self.calls:
            if call.provider not in provider_breakdown:
                provider_breakdown[call.provider] = {
                    "requests": 0,
                    "tokens": 0,
                    "total_cost_usd": Decimal("0"),
                }
            provider_breakdown[call.provider]["requests"] += 1
            provider_breakdown[call.provider]["tokens"] += call.total_tokens
            provider_breakdown[call.provider]["total_cost_usd"] += call.total_cost_usd

        # Aggregate by model
        model_breakdown: dict[str, dict[str, Any]] = {}
        for call in self.calls:
            if call.model not in model_breakdown:
                model_breakdown[call.model] = {
                    "provider": call.provider,
                    "requests": 0,
                    "tokens": 0,
                    "total_cost_usd": Decimal("0"),
                }
            model_breakdown[call.model]["requests"] += 1
            model_breakdown[call.model]["tokens"] += call.total_tokens
            model_breakdown[call.model]["total_cost_usd"] += call.total_cost_usd

        return RunCostSummary(
            run_id=self.run_id,
            start_time=self.start_time,
            end_time=datetime.now().isoformat(),
            total_requests=len(self.calls),
            total_tokens=sum(c.total_tokens for c in self.calls),
            total_cost_usd=sum((c.total_cost_usd for c in self.calls), Decimal("0")),
            provider_breakdown=provider_breakdown,
            model_breakdown=model_breakdown,
        )

    def save_detailed_log(self, filepath: Path) -> None:
        """Save detailed cost log for each call."""
        with open(filepath, "w") as f:
            for call in self.calls:
                f.write(json.dumps(call.to_dict()) + "\n")
        logger.info(f"Saved detailed cost log to {filepath}")

    def save_summary(self, filepath: Path) -> None:
        """Save cost summary."""
        summary = self.get_summary()
        with open(filepath, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        logger.info(f"Saved cost summary to {filepath}")

    def print_summary(self) -> None:
        """Print cost summary to console."""
        from rich.console import Console
        from rich.table import Table

        summary = self.get_summary()
        console = Console()

        console.print("\n[bold cyan]Cost Summary[/bold cyan]")
        console.print(f"Run ID: {summary.run_id}")
        console.print(f"Duration: {summary.start_time} → {summary.end_time}")
        console.print(f"Total Requests: {summary.total_requests}")
        console.print(f"Total Tokens: {summary.total_tokens:,}")
        console.print(
            f"[bold green]Total Cost: ${summary.total_cost_usd:.8f}[/bold green]"
        )

        # Provider breakdown
        if summary.provider_breakdown:
            console.print("\n[bold]By Provider:[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Provider")
            table.add_column("Requests", justify="right")
            table.add_column("Tokens", justify="right")
            table.add_column("Cost (USD)", justify="right")

            for provider, stats in summary.provider_breakdown.items():
                table.add_row(
                    provider,
                    str(stats["requests"]),
                    f"{stats['tokens']:,}",
                    f"${Decimal(stats['total_cost_usd']):.8f}",
                )
            console.print(table)

        # Model breakdown
        if summary.model_breakdown:
            console.print("\n[bold]By Model:[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Model")
            table.add_column("Provider")
            table.add_column("Requests", justify="right")
            table.add_column("Tokens", justify="right")
            table.add_column("Cost (USD)", justify="right")

            for model, stats in summary.model_breakdown.items():
                table.add_row(
                    model,
                    stats["provider"],
                    str(stats["requests"]),
                    f"{stats['tokens']:,}",
                    f"${Decimal(stats['total_cost_usd']):.8f}",
                )
            console.print(table)
