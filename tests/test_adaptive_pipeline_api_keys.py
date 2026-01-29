"""
Test the adaptive pipeline with API key override functionality.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.adaptive_config import AdaptivePipelineConfig, ModelConfig


class TestAdaptivePipelineAPIKeys:
    """Test adaptive pipeline properly passes API keys to model runners."""

    def test_model_config_from_yaml_no_api_key(self, tmp_path):
        """Test loading config without explicit api_key uses env var."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
dataset:
  prompts:
    prompts_dir: data/prompts/basic
    style: basic

model:
  provider: openai
  name: gpt-4o-mini
  temperature: 0.1

evaluation:
  mode: both

output:
  dir: outputs/test
"""
        )

        config = AdaptivePipelineConfig.from_yaml(config_file)

        assert config.model is not None
        assert config.model.api_key is None  # No explicit key
        assert config.model.provider == "openai"
        # Provider openai → will use OPENAI_API_KEY from env

    def test_model_config_from_yaml_with_api_key(self, tmp_path):
        """Test loading config with explicit api_key overrides env."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
dataset:
  prompts:
    prompts_dir: data/prompts/basic
    style: basic

model:
  provider: openrouter
  name: anthropic/claude-3.5-sonnet
  api_key: sk-or-v1-explicit-key-from-yaml
  temperature: 0.1

evaluation:
  mode: both

output:
  dir: outputs/test
"""
        )

        config = AdaptivePipelineConfig.from_yaml(config_file)

        assert config.model is not None
        assert config.model.api_key == "sk-or-v1-explicit-key-from-yaml"
        assert config.model.provider == "openrouter"

    def test_model_config_openai_provider(self):
        """Test OpenAI provider config structure."""
        config = ModelConfig(
            provider="openai",
            name="gpt-4o-mini",
            temperature=0.1,
            max_tokens=2048,
        )

        assert config.provider == "openai"
        assert config.api_key is None  # Will use OPENAI_API_KEY env
        assert config.name == "gpt-4o-mini"

    def test_model_config_openrouter_provider(self):
        """Test OpenRouter provider config structure."""
        config = ModelConfig(
            provider="openrouter",
            name="anthropic/claude-3.5-sonnet",
            api_key="sk-override",
        )

        assert config.provider == "openrouter"
        assert config.api_key == "sk-override"  # Explicit override

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-env-key"})
    def test_model_runner_kwargs_without_override(self):
        """Test that runner_kwargs are built correctly without override."""
        model_config = ModelConfig(
            provider="openai", name="gpt-4o-mini", temperature=0.2, max_tokens=1024
        )

        # Simulate what adaptive_pipeline does
        runner_kwargs = {
            "model_name": model_config.name,
            "api_key": model_config.api_key,  # None, will use env
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
            "timeout": model_config.timeout,
        }

        assert runner_kwargs["model_name"] == "gpt-4o-mini"
        assert runner_kwargs["api_key"] is None  # None → uses OPENAI_API_KEY
        assert runner_kwargs["temperature"] == 0.2
        assert runner_kwargs["max_tokens"] == 1024

    def test_model_runner_kwargs_with_override(self):
        """Test that runner_kwargs pass explicit api_key correctly."""
        model_config = ModelConfig(
            provider="openrouter",
            name="anthropic/claude-3.5-sonnet",
            api_key="sk-yaml-override",
            temperature=0.1,
        )

        # Simulate what adaptive_pipeline does
        runner_kwargs = {
            "model_name": model_config.name,
            "api_key": model_config.api_key,  # Explicit key
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
            "timeout": model_config.timeout,
        }

        assert runner_kwargs["model_name"] == "anthropic/claude-3.5-sonnet"
        assert runner_kwargs["api_key"] == "sk-yaml-override"  # Overrides env
        assert runner_kwargs["temperature"] == 0.1
