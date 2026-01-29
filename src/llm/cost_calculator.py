"""
Cost calculation helpers for OpenAI and OpenRouter responses.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from src.llm.cost_tracker import CallCost
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def calculate_openai_cost(response: Any, model: str) -> CallCost:
    """
    Calculate cost from OpenAI API response using openai-cost-calculator.

    Args:
        response: OpenAI ChatCompletion response object
        model: Model name

    Returns:
        CallCost object with detailed cost breakdown
    """
    try:
        from openai_cost_calculator import estimate_cost_typed
    except ImportError:
        logger.warning(
            "openai-cost-calculator not installed. Install with: pip install openai-cost-calculator"
        )
        # Fallback to basic calculation
        return _fallback_cost_calculation(response, model, "openai")

    try:
        cost_breakdown = estimate_cost_typed(response)

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        # Get cached tokens if available
        cached_tokens = 0
        if hasattr(usage, "prompt_tokens_details"):
            details = usage.prompt_tokens_details
            if hasattr(details, "cached_tokens"):
                cached_tokens = details.cached_tokens or 0

        return CallCost(
            timestamp=datetime.now().isoformat(),
            model=model,
            provider="openai",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
            prompt_cost_usd=cost_breakdown.prompt_cost_uncached,
            completion_cost_usd=cost_breakdown.completion_cost,
            cached_cost_usd=cost_breakdown.prompt_cost_cached,
            total_cost_usd=cost_breakdown.total_cost,
            request_id=response.id if hasattr(response, "id") else None,
        )
    except Exception as e:
        logger.warning(f"Cost calculation failed: {e}. Using fallback.")
        return _fallback_cost_calculation(response, model, "openai")


def calculate_openrouter_cost(response: Any, model: str) -> CallCost:
    """
    Calculate cost from OpenRouter API response.

    OpenRouter automatically includes usage and cost information in responses.

    Args:
        response: OpenRouter ChatCompletion response object
        model: Model name

    Returns:
        CallCost object with detailed cost breakdown
    """
    try:
        usage = response.usage

        # OpenRouter provides cost directly in the response
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        # Get cached tokens from details
        cached_tokens = 0
        if hasattr(usage, "prompt_tokens_details"):
            details = usage.prompt_tokens_details
            if hasattr(details, "cached_tokens"):
                cached_tokens = details.cached_tokens or 0

        # OpenRouter provides cost in credits (already USD)
        # Cost field is in the usage object
        total_cost_usd = (
            Decimal(str(usage.cost)) if hasattr(usage, "cost") else Decimal("0")
        )

        # Rough breakdown based on token counts
        # (OpenRouter doesn't provide separate prompt/completion costs)
        if total_tokens > 0:
            prompt_ratio = Decimal(str(prompt_tokens / total_tokens))
            completion_ratio = Decimal(str(completion_tokens / total_tokens))
            cached_ratio = (
                Decimal(str(cached_tokens / total_tokens))
                if cached_tokens > 0
                else Decimal("0")
            )

            prompt_cost_usd = total_cost_usd * prompt_ratio
            completion_cost_usd = total_cost_usd * completion_ratio
            cached_cost_usd = total_cost_usd * cached_ratio
        else:
            prompt_cost_usd = Decimal("0")
            completion_cost_usd = Decimal("0")
            cached_cost_usd = Decimal("0")

        return CallCost(
            timestamp=datetime.now().isoformat(),
            model=model,
            provider="openrouter",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
            prompt_cost_usd=prompt_cost_usd,
            completion_cost_usd=completion_cost_usd,
            cached_cost_usd=cached_cost_usd,
            total_cost_usd=total_cost_usd,
            request_id=response.id if hasattr(response, "id") else None,
        )
    except Exception as e:
        logger.warning(f"OpenRouter cost calculation failed: {e}. Using fallback.")
        return _fallback_cost_calculation(response, model, "openrouter")


def _fallback_cost_calculation(response: Any, model: str, provider: str) -> CallCost:
    """
    Fallback cost calculation when detailed calculation fails.

    Returns basic token counts with zero costs.
    """
    try:
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if hasattr(usage, "prompt_tokens") else 0
        completion_tokens = (
            usage.completion_tokens if hasattr(usage, "completion_tokens") else 0
        )
        total_tokens = usage.total_tokens if hasattr(usage, "total_tokens") else 0
    except:
        prompt_tokens = completion_tokens = total_tokens = 0

    return CallCost(
        timestamp=datetime.now().isoformat(),
        model=model,
        provider=provider,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=0,
        total_tokens=total_tokens,
        prompt_cost_usd=Decimal("0"),
        completion_cost_usd=Decimal("0"),
        cached_cost_usd=Decimal("0"),
        total_cost_usd=Decimal("0"),
        request_id=None,
    )
