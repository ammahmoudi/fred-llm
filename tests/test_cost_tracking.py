"""Tests for cost tracking functionality."""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.llm.cost_calculator import calculate_openai_cost, calculate_openrouter_cost
from src.llm.cost_tracker import CallCost, CostTracker
from src.llm.model_runner import OpenAIModelRunner, OpenRouterModelRunner


def make_call_cost(
    prompt_tokens: int,
    completion_tokens: int,
    total_cost_usd: Decimal,
    model: str,
    provider: str,
) -> CallCost:
    """Helper to create a CallCost with sensible defaults."""
    return CallCost(
        timestamp=datetime.now().isoformat(),
        model=model,
        provider=provider,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=0,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_cost_usd=total_cost_usd * Decimal("0.6"),  # Rough approximation
        completion_cost_usd=total_cost_usd * Decimal("0.4"),
        cached_cost_usd=Decimal("0"),
        total_cost_usd=total_cost_usd,
    )


class TestCostCalculations:
    """Test cost calculation functions."""

    @patch("openai_cost_calculator.estimate_cost_typed")
    def test_calculate_openai_cost(self, mock_estimate):
        """Test OpenAI cost calculation."""
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_response.usage.prompt_tokens_details.cached_tokens = 10

        # Mock the cost breakdown returned by openai-cost-calculator
        mock_breakdown = MagicMock()
        mock_breakdown.total_cost = Decimal("0.0045")
        mock_breakdown.prompt_cost_uncached = Decimal("0.0020")
        mock_breakdown.prompt_cost_cached = Decimal("0.0005")
        mock_breakdown.completion_cost = Decimal("0.0020")
        mock_estimate.return_value = mock_breakdown

        cost = calculate_openai_cost(mock_response, "gpt-4")

        assert cost.prompt_tokens == 100
        assert cost.completion_tokens == 50
        assert cost.total_tokens == 150
        assert cost.cached_tokens == 10
        assert cost.total_cost_usd == Decimal("0.0045")
        assert cost.model == "gpt-4"
        assert cost.provider == "openai"

    def test_calculate_openrouter_cost(self):
        """Test OpenRouter cost calculation."""
        mock_response = MagicMock()
        mock_response.id = "test-id-123"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_response.usage.cost = 0.0025
        # Need to configure hasattr behavior
        mock_response.usage.prompt_tokens_details = None  # No cached tokens

        cost = calculate_openrouter_cost(mock_response, "anthropic/claude-3-opus")

        assert cost.prompt_tokens == 100
        assert cost.completion_tokens == 50
        assert cost.total_tokens == 150
        assert cost.total_cost_usd == Decimal("0.0025")
        assert cost.model == "anthropic/claude-3-opus"
        assert cost.provider == "openrouter"

    def test_calculate_openrouter_cost_no_cost_field(self):
        """Test OpenRouter cost calculation when cost field is missing."""
        mock_response = MagicMock()
        mock_response.id = "test-id-456"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_response.usage.cost = None  # Missing cost
        mock_response.usage.prompt_tokens_details = None

        cost = calculate_openrouter_cost(mock_response, "openai/gpt-3.5-turbo")

        assert cost.prompt_tokens == 100
        assert cost.completion_tokens == 50
        assert cost.total_tokens == 150
        assert cost.total_cost_usd == Decimal("0")  # Should default to 0
        assert cost.model == "openai/gpt-3.5-turbo"
        assert cost.provider == "openrouter"


class TestCostTracker:
    """Test CostTracker functionality."""

    def test_add_call(self):
        """Test adding a call to tracker."""
        tracker = CostTracker()

        cost = make_call_cost(100, 50, Decimal("0.0045"), "gpt-4", "openai")

        tracker.add_call(cost)

        assert len(tracker.calls) == 1
        assert tracker.calls[0] == cost

    def test_get_summary(self):
        """Test getting summary of costs."""
        tracker = CostTracker()

        # Add multiple calls
        tracker.add_call(make_call_cost(100, 50, Decimal("0.0045"), "gpt-4", "openai"))
        tracker.add_call(make_call_cost(200, 100, Decimal("0.0090"), "gpt-4", "openai"))
        tracker.add_call(
            make_call_cost(150, 75, Decimal("0.0025"), "claude-3-opus", "openrouter")
        )

        summary = tracker.get_summary()

        assert summary.total_requests == 3
        assert summary.total_tokens == 450 + 225  # prompt + completion for all
        assert summary.total_cost_usd == Decimal("0.0160")

        # Check by provider
        assert "openai" in summary.provider_breakdown
        assert "openrouter" in summary.provider_breakdown
        assert summary.provider_breakdown["openai"]["requests"] == 2
        assert summary.provider_breakdown["openai"]["total_cost_usd"] == Decimal(
            "0.0135"
        )
        assert summary.provider_breakdown["openrouter"]["requests"] == 1
        assert summary.provider_breakdown["openrouter"]["total_cost_usd"] == Decimal(
            "0.0025"
        )

        # Check by model
        assert "gpt-4" in summary.model_breakdown
        assert "claude-3-opus" in summary.model_breakdown
        assert summary.model_breakdown["gpt-4"]["requests"] == 2
        assert summary.model_breakdown["claude-3-opus"]["requests"] == 1

    def test_save_summary(self, tmp_path):
        """Test saving summary to file."""
        tracker = CostTracker()
        tracker.add_call(make_call_cost(100, 50, Decimal("0.0045"), "gpt-4", "openai"))

        output_file = tmp_path / "cost_summary.json"
        tracker.save_summary(output_file)

        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert data["total_requests"] == 1
        assert data["total_cost_usd"] == "0.0045"

    def test_save_detailed_log(self, tmp_path):
        """Test saving detailed log to file."""
        tracker = CostTracker()
        tracker.add_call(make_call_cost(100, 50, Decimal("0.0045"), "gpt-4", "openai"))
        tracker.add_call(make_call_cost(200, 100, Decimal("0.0090"), "gpt-4", "openai"))

        output_file = tmp_path / "cost_details.jsonl"
        tracker.save_detailed_log(output_file)

        assert output_file.exists()

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2

        call1 = json.loads(lines[0])
        assert call1["prompt_tokens"] == 100
        assert call1["total_cost_usd"] == "0.0045"


class TestModelRunnerCostTracking:
    """Test cost tracking integration with model runners."""

    @patch("src.llm.model_runner.OpenAI")
    def test_openai_runner_tracks_costs(self, mock_openai_class):
        """Test that OpenAI runner tracks costs."""
        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock the response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.model = "gpt-4"
        mock_client.chat.completions.create.return_value = mock_response

        # Create runner with cost tracker
        runner = OpenAIModelRunner(api_key="test-key", model_name="gpt-4")
        tracker = CostTracker()
        runner.set_cost_tracker(tracker)

        # Mock the cost calculation
        with patch("src.llm.model_runner.calculate_openai_cost") as mock_calc:
            mock_calc.return_value = make_call_cost(
                100, 50, Decimal("0.0045"), "gpt-4", "openai"
            )

            # Generate a response
            result = runner.generate("Test prompt")

            assert result == "Test response"
            assert len(tracker.calls) == 1
            assert tracker.calls[0].total_cost_usd == Decimal("0.0045")

    @patch("src.llm.model_runner.OpenAI")
    def test_openrouter_runner_tracks_costs(self, mock_openai_class):
        """Test that OpenRouter runner tracks costs."""
        # Mock the OpenAI client (OpenRouter uses OpenAI SDK)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock the response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.cost = 0.0025
        mock_response.model = "anthropic/claude-3-opus"
        mock_client.chat.completions.create.return_value = mock_response

        # Create runner with cost tracker
        runner = OpenRouterModelRunner(
            api_key="test-key", model_name="anthropic/claude-3-opus"
        )
        tracker = CostTracker()
        runner.set_cost_tracker(tracker)

        # Mock the cost calculation
        with patch("src.llm.model_runner.calculate_openrouter_cost") as mock_calc:
            mock_calc.return_value = make_call_cost(
                100, 50, Decimal("0.0025"), "anthropic/claude-3-opus", "openrouter"
            )

            # Generate a response
            result = runner.generate("Test prompt")

            assert result == "Test response"
            assert len(tracker.calls) == 1
            assert tracker.calls[0].total_cost_usd == Decimal("0.0025")

    @patch("src.llm.model_runner.OpenAI")
    def test_runner_without_cost_tracker(self, mock_openai_class):
        """Test that runner works without cost tracker."""
        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock the response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response

        # Create runner WITHOUT cost tracker
        runner = OpenAIModelRunner(api_key="test-key", model_name="gpt-4")

        # Generate should still work
        result = runner.generate("Test prompt")
        assert result == "Test response"
