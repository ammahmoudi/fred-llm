"""
Test API key override mechanism.

Verify that:
1. API keys are loaded from environment variables based on provider
2. API keys in YAML config override environment variables
"""

import os
from unittest.mock import patch

import pytest

from src.adaptive_config import ModelConfig
from src.llm.model_runner import ModelRunner, OpenAIModelRunner, OpenRouterModelRunner


class TestAPIKeyOverride:
    """Test API key loading and override behavior."""

    def test_model_config_no_direct_key(self):
        """Test that ModelConfig works without direct key."""
        config = ModelConfig(provider="openai", name="gpt-4o-mini")
        
        assert config.api_key is None  # No direct key
        assert config.provider == "openai"

    def test_model_config_direct_key_override(self):
        """Test that direct api_key in config is stored."""
        config = ModelConfig(
            provider="openai",
            name="gpt-4o-mini",
            api_key="sk-direct-key-123"
        )
        
        assert config.api_key == "sk-direct-key-123"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key-456"})
    def test_openai_runner_uses_env_var(self):
        """Test OpenAI runner loads key from OPENAI_API_KEY env."""
        runner = OpenAIModelRunner(
            model_name="gpt-4o-mini",
            api_key=None  # Not provided
        )
        
        assert runner.api_key == "env-key-456"

    def test_openai_runner_direct_key_overrides_env(self):
        """Test direct key overrides OPENAI_API_KEY environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key-789"}):
            runner = OpenAIModelRunner(
                model_name="gpt-4o-mini",
                api_key="direct-key-override"  # Direct key provided
            )
            
            assert runner.api_key == "direct-key-override"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-router-key"})
    def test_openrouter_runner_uses_env_var(self):
        """Test OpenRouter runner loads key from OPENROUTER_API_KEY env."""
        runner = OpenRouterModelRunner(
            model_name="anthropic/claude-3.5-sonnet",
            api_key=None
        )
        
        assert runner.api_key == "env-router-key"

    def test_openrouter_runner_direct_key_overrides_env(self):
        """Test direct key overrides OPENROUTER_API_KEY environment."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-router-key"}):
            runner = OpenRouterModelRunner(
                model_name="anthropic/claude-3.5-sonnet",
                api_key="direct-router-key"
            )
            
            assert runner.api_key == "direct-router-key"

    def test_model_runner_factory_passes_keys(self):
        """Test ModelRunner factory passes API keys correctly."""
        runner = ModelRunner(
            provider="openai",
            model_name="gpt-4o-mini",
            api_key="factory-key"
        )
        
        assert runner._runner.api_key == "factory-key"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "openai-env"})
    def test_provider_determines_env_var_openai(self):
        """Test that openai provider uses OPENAI_API_KEY."""
        runner = OpenAIModelRunner(model_name="gpt-4o-mini")
        assert runner.api_key == "openai-env"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "router-env"})
    def test_provider_determines_env_var_openrouter(self):
        """Test that openrouter provider uses OPENROUTER_API_KEY."""
        runner = OpenRouterModelRunner(model_name="anthropic/claude-3.5-sonnet")
        assert runner.api_key == "router-env"
