"""
Tests for model runner module.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.llm.model_runner import (
    ModelRunner,
    OpenAIModelRunner,
    OpenRouterModelRunner,
    LocalModelRunner,
)


class TestOpenAIModelRunner:
    """Tests for OpenAI model runner."""

    def test_initialization(self) -> None:
        """Test runner initialization."""
        runner = OpenAIModelRunner(
            model_name="gpt-4",
            api_key="test-key",
            temperature=0.5,
        )

        assert runner.model_name == "gpt-4"
        assert runner.temperature == 0.5
        assert runner.api_key == "test-key"

    def test_api_key_from_env(self) -> None:
        """Test loading API key from environment."""
        os.environ["OPENAI_API_KEY"] = "env-test-key"
        try:
            runner = OpenAIModelRunner()
            assert runner.api_key == "env-test-key"
        finally:
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

    def test_missing_api_key_raises_error(self) -> None:
        """Test that missing API key raises error on client creation."""
        # Remove the env var if it exists
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            runner = OpenAIModelRunner(api_key=None)

            with pytest.raises(ValueError, match="API key not found"):
                runner._get_client()
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_generate_with_mock(self) -> None:
        """Test generate with mocked OpenAI client."""
        runner = OpenAIModelRunner(model_name="gpt-4", api_key="test-key")

        # Create mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "u(x) = x^2"

        with patch.object(runner, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = runner.generate("Solve the equation")

            assert result == "u(x) = x^2"
            mock_client.chat.completions.create.assert_called_once()

    def test_batch_generate_returns_list(self) -> None:
        """Test that batch_generate returns a list of responses."""
        runner = OpenAIModelRunner(model_name="gpt-4", api_key="test-key")

        # Mock single generate
        with patch.object(runner, "generate", side_effect=["resp1", "resp2", "resp3"]):
            results = runner.batch_generate(
                ["p1", "p2", "p3"], show_progress=False, rate_limit_delay=0
            )

            assert isinstance(results, list)
            assert len(results) == 3
            assert results == ["resp1", "resp2", "resp3"]


class TestOpenRouterModelRunner:
    """Tests for OpenRouter model runner."""

    def test_initialization(self) -> None:
        """Test OpenRouter runner initialization."""
        runner = OpenRouterModelRunner(
            model_name="anthropic/claude-3.5-sonnet",
            api_key="test-key",
            app_name="test-app",
        )

        assert runner.model_name == "anthropic/claude-3.5-sonnet"
        assert runner.app_name == "test-app"
        assert runner.api_key == "test-key"

    def test_base_url_is_openrouter(self) -> None:
        """Test that base URL is OpenRouter."""
        assert OpenRouterModelRunner.BASE_URL == "https://openrouter.ai/api/v1"

    def test_api_key_from_env(self) -> None:
        """Test loading API key from environment."""
        os.environ["OPENROUTER_API_KEY"] = "env-test-key"
        try:
            runner = OpenRouterModelRunner()
            assert runner.api_key == "env-test-key"
        finally:
            del os.environ["OPENROUTER_API_KEY"]

    def test_missing_api_key_raises_error(self) -> None:
        """Test that missing API key raises error on client creation."""
        # Remove the env var if it exists
        old_key = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            runner = OpenRouterModelRunner(api_key=None)

            with pytest.raises(ValueError, match="API key not found"):
                runner._get_client()
        finally:
            if old_key is not None:
                os.environ["OPENROUTER_API_KEY"] = old_key

    def test_generate_with_mock(self) -> None:
        """Test generate with mocked client."""
        runner = OpenRouterModelRunner(
            model_name="anthropic/claude-3.5-sonnet", api_key="test-key"
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "u(x) = sin(x)"

        with patch.object(runner, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = runner.generate("Solve the equation")

            assert result == "u(x) = sin(x)"


class TestLocalModelRunner:
    """Tests for local model runner."""

    def test_initialization(self) -> None:
        """Test local runner initialization."""
        runner = LocalModelRunner(
            model_path="meta-llama/Llama-2-7b",
            device="cpu",
        )

        assert runner.model_path == "meta-llama/Llama-2-7b"
        assert runner.device == "cpu"

    def test_generate_raises_not_implemented(self) -> None:
        """Test that generate raises NotImplementedError."""
        runner = LocalModelRunner(model_path="test/model")

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            runner.generate("test prompt")

    def test_batch_generate_raises_not_implemented(self) -> None:
        """Test that batch_generate raises NotImplementedError."""
        runner = LocalModelRunner(model_path="test/model")

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            runner.batch_generate(["p1", "p2"])


class TestModelRunnerFactory:
    """Tests for ModelRunner factory class."""

    def test_create_openai_runner(self) -> None:
        """Test creating OpenAI runner via factory."""
        runner = ModelRunner(
            provider="openai",
            model_name="gpt-4",
            api_key="test",
        )

        assert runner.provider == "openai"
        assert isinstance(runner._runner, OpenAIModelRunner)

    def test_create_openrouter_runner(self) -> None:
        """Test creating OpenRouter runner via factory."""
        runner = ModelRunner(
            provider="openrouter",
            model_name="anthropic/claude-3.5-sonnet",
            api_key="test",
        )

        assert runner.provider == "openrouter"
        assert isinstance(runner._runner, OpenRouterModelRunner)

    def test_create_local_runner(self) -> None:
        """Test creating local runner via factory."""
        runner = ModelRunner(
            provider="local",
            model_path="path/to/model",
        )

        assert runner.provider == "local"
        assert isinstance(runner._runner, LocalModelRunner)

    def test_huggingface_alias(self) -> None:
        """Test that huggingface maps to local runner."""
        runner = ModelRunner(
            provider="huggingface",
            model_path="path/to/model",
        )

        assert isinstance(runner._runner, LocalModelRunner)

    def test_unknown_provider_raises_error(self) -> None:
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            ModelRunner(provider="unknown")

    def test_factory_delegates_generate(self) -> None:
        """Test that factory delegates generate to runner."""
        runner = ModelRunner(provider="openai", api_key="test")

        with patch.object(runner._runner, "generate", return_value="response"):
            result = runner.generate("test prompt")
            assert result == "response"

    def test_factory_delegates_batch_generate(self) -> None:
        """Test that factory delegates batch_generate to runner."""
        runner = ModelRunner(provider="openai", api_key="test")

        with patch.object(
            runner._runner, "batch_generate", return_value=["r1", "r2"]
        ):
            results = runner.batch_generate(["p1", "p2"])
            assert results == ["r1", "r2"]


class TestBatchGeneration:
    """Tests for batch generation functionality."""

    def test_batch_handles_errors_gracefully(self) -> None:
        """Test that batch generation handles individual errors."""
        runner = OpenAIModelRunner(model_name="gpt-4", api_key="test-key")

        def mock_generate(prompt, **kwargs):
            if prompt == "fail":
                raise Exception("API Error")
            return f"response for {prompt}"

        with patch.object(runner, "generate", side_effect=mock_generate):
            results = runner.batch_generate(
                ["ok1", "fail", "ok2"], show_progress=False, rate_limit_delay=0
            )

            assert len(results) == 3
            assert results[0] == "response for ok1"
            assert results[1] == ""  # Failed request returns empty string
            assert results[2] == "response for ok2"

    def test_batch_preserves_order(self) -> None:
        """Test that batch results maintain order."""
        runner = OpenAIModelRunner(model_name="gpt-4", api_key="test-key")

        responses = ["first", "second", "third"]
        with patch.object(runner, "generate", side_effect=responses):
            results = runner.batch_generate(
                ["p1", "p2", "p3"], show_progress=False, rate_limit_delay=0
            )

            assert results == responses
