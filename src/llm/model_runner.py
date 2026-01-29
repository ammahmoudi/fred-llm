"""
Model runner for LLM inference.

Supports multiple providers: OpenAI API, OpenRouter, local models, HuggingFace.
"""

import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any

from openai import OpenAI
from rich.progress import Progress, SpinnerColumn, TextColumn
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.llm.cost_calculator import calculate_openai_cost, calculate_openrouter_cost
from src.llm.cost_tracker import CallCost, CostTracker
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseModelRunner(ABC):
    """Abstract base class for model runners."""

    def __init__(self):
        """Initialize base runner with cost tracker."""
        self.cost_tracker: CostTracker | None = None

    def set_cost_tracker(self, tracker: CostTracker) -> None:
        """Set the cost tracker for this runner."""
        self.cost_tracker = tracker

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the model."""
        pass

    @abstractmethod
    def batch_generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """Generate responses for multiple prompts."""
        pass


class OpenAIModelRunner(BaseModelRunner):
    """Model runner for OpenAI API."""

    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int = 60,
    ) -> None:
        """
        Initialize OpenAI model runner.

        Args:
            model_name: Name of the model to use.
            api_key: API key (if not provided, uses OPENAI_API_KEY env var).
            base_url: Optional custom base URL.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
        """
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        self._client = None

    def _get_client(self) -> OpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY env var or pass api_key."
                )
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a response from OpenAI API.

        Args:
            prompt: The input prompt.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text response.
        """
        logger.debug(f"Generating response with {self.model_name}")

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        # Track cost if tracker is available
        if self.cost_tracker:
            try:
                cost = calculate_openai_cost(response, self.model_name)
                self.cost_tracker.add_call(cost)
            except Exception as e:
                logger.warning(f"Failed to track cost: {e}")

        content = response.choices[0].message.content
        return content if content else ""

    def batch_generate(
        self,
        prompts: list[str],
        rate_limit_delay: float = 0.5,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        """
        Generate responses for multiple prompts with rate limiting.

        Args:
            prompts: List of input prompts.
            rate_limit_delay: Delay in seconds between requests.
            show_progress: Whether to show progress bar.
            **kwargs: Additional generation parameters.

        Returns:
            List of generated responses.
        """
        logger.info(f"Batch generating {len(prompts)} responses with {self.model_name}")
        results: list[str] = []
        errors: list[tuple[int, str]] = []

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"Generating with {self.model_name}...", total=len(prompts)
                )
                for i, prompt in enumerate(prompts):
                    try:
                        result = self.generate(prompt, **kwargs)
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Failed to generate for prompt {i}: {e}")
                        errors.append((i, str(e)))
                        results.append("")
                    progress.advance(task)
                    if i < len(prompts) - 1:
                        time.sleep(rate_limit_delay)
        else:
            for i, prompt in enumerate(prompts):
                try:
                    result = self.generate(prompt, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to generate for prompt {i}: {e}")
                    errors.append((i, str(e)))
                    results.append("")
                if i < len(prompts) - 1:
                    time.sleep(rate_limit_delay)

        if errors:
            logger.warning(f"Batch generation completed with {len(errors)} errors")

        return results


class LocalModelRunner(BaseModelRunner):
    """Model runner for local models (e.g., via transformers or vLLM)."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        """
        Initialize local model runner.

        Args:
            model_path: Path to the model or HuggingFace model ID.
            device: Device to run on (cuda, cpu, mps).
            **kwargs: Additional model configuration.
        """
        self.model_path = model_path
        self.device = device
        self.config = kwargs

        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        # TODO: Implement model loading
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # self._model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device)
        pass

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from local model."""
        raise NotImplementedError(
            "Local model inference is not yet implemented. "
            "Use 'openai' or 'openrouter' provider instead."
        )

    def batch_generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """Generate responses for multiple prompts."""
        raise NotImplementedError(
            "Local model inference is not yet implemented. "
            "Use 'openai' or 'openrouter' provider instead."
        )


class OpenRouterModelRunner(BaseModelRunner):
    """Model runner for OpenRouter API (access multiple models via single API)."""

    # OpenRouter base URL
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model_name: str = "anthropic/claude-3.5-sonnet",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 120,
        app_name: str = "fred-llm",
        site_url: str | None = None,
    ) -> None:
        """
        Initialize OpenRouter model runner.

        Args:
            model_name: Model identifier (e.g., 'anthropic/claude-3.5-sonnet').
            api_key: API key (if not provided, uses OPENROUTER_API_KEY env var).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            app_name: App name for OpenRouter analytics.
            site_url: Site URL for OpenRouter rankings.
        """
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.app_name = app_name
        self.site_url = site_url

        self._client = None

    def _get_client(self) -> OpenAI:
        """Get or create OpenAI-compatible client for OpenRouter."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "OpenRouter API key not found. Set OPENROUTER_API_KEY env var or pass api_key."
                )
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.BASE_URL,
                timeout=self.timeout,
                default_headers={
                    "HTTP-Referer": self.site_url or "",
                    "X-Title": self.app_name,
                },
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a response from OpenRouter API.

        Args:
            prompt: The input prompt.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text response.
        """
        logger.debug(f"Generating response with OpenRouter model: {self.model_name}")

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        # Track cost if tracker is available (OpenRouter provides cost in response)
        if self.cost_tracker:
            try:
                cost = calculate_openrouter_cost(response, self.model_name)
                self.cost_tracker.add_call(cost)
            except Exception as e:
                logger.warning(f"Failed to track cost: {e}")

        content = response.choices[0].message.content
        return content if content else ""

    def batch_generate(
        self,
        prompts: list[str],
        rate_limit_delay: float = 1.0,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        """
        Generate responses for multiple prompts with rate limiting.

        Args:
            prompts: List of input prompts.
            rate_limit_delay: Delay in seconds between requests (default 1.0 for OpenRouter).
            show_progress: Whether to show progress bar.
            **kwargs: Additional generation parameters.

        Returns:
            List of generated responses.
        """
        logger.info(f"Batch generating {len(prompts)} responses via OpenRouter")
        results: list[str] = []
        errors: list[tuple[int, str]] = []

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"Generating with {self.model_name}...", total=len(prompts)
                )
                for i, prompt in enumerate(prompts):
                    try:
                        result = self.generate(prompt, **kwargs)
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Failed to generate for prompt {i}: {e}")
                        errors.append((i, str(e)))
                        results.append("")
                    progress.advance(task)
                    if i < len(prompts) - 1:
                        time.sleep(rate_limit_delay)
        else:
            for i, prompt in enumerate(prompts):
                try:
                    result = self.generate(prompt, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to generate for prompt {i}: {e}")
                    errors.append((i, str(e)))
                    results.append("")
                if i < len(prompts) - 1:
                    time.sleep(rate_limit_delay)

        if errors:
            logger.warning(f"Batch generation completed with {len(errors)} errors")

        return results


class ModelRunner:
    """Factory class to create appropriate model runner."""

    def __init__(
        self,
        provider: str = "openai",
        **kwargs: Any,
    ) -> None:
        """
        Initialize model runner.

        Args:
            provider: Model provider (openai, openrouter, local, huggingface).
            **kwargs: Provider-specific configuration.
        """
        self.provider = provider

        if provider == "openai":
            self._runner = OpenAIModelRunner(**kwargs)
        elif provider == "openrouter":
            self._runner = OpenRouterModelRunner(**kwargs)
        elif provider in ("local", "huggingface"):
            self._runner = LocalModelRunner(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response."""
        return self._runner.generate(prompt, **kwargs)

    def batch_generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """Generate responses for multiple prompts."""
        return self._runner.batch_generate(prompts, **kwargs)

    def set_cost_tracker(self, tracker: "CostTracker") -> None:
        """Set the cost tracker."""
        self._runner.set_cost_tracker(tracker)
