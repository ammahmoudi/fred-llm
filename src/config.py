"""
Configuration loader for Fred-LLM.

Loads and validates configuration from YAML files.
"""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    name: str = "FIE-500k"
    path: str = "data/raw/fie_500k.json"
    processed_path: str = "data/processed/"
    format: str = "json"
    max_samples: Optional[int] = None


class ModelConfig(BaseModel):
    """LLM model configuration."""

    provider: str = "openai"
    name: str = "gpt-4"
    api_key: Optional[str] = (
        None  # Direct API key (overrides env var based on provider)
    )
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 60


class PromptConfig(BaseModel):
    """Prompting configuration."""

    style: str = "chain-of-thought"
    template_dir: str = "data/prompts/"
    include_examples: bool = True
    num_examples: int = 3


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    mode: str = "both"
    symbolic_tolerance: float = 1e-10
    numeric_tolerance: float = 1e-6
    test_points: int = 100
    integration_method: str = "quad"


class OutputConfig(BaseModel):
    """Output configuration."""

    format: str = "latex"
    save_intermediate: bool = True
    results_dir: str = "data/processed/results/"
    log_level: str = "INFO"


class DomainConfig(BaseModel):
    """Integration domain configuration."""

    a: float = 0.0
    b: float = 1.0


class LambdaRangeConfig(BaseModel):
    """Lambda parameter range."""

    min: float = 0.1
    max: float = 2.0


class EquationConfig(BaseModel):
    """Equation-specific configuration."""

    kernel_types: list[str] = Field(
        default_factory=lambda: [
            "polynomial",
            "exponential",
            "trigonometric",
            "separable",
        ]
    )
    domain: DomainConfig = Field(default_factory=DomainConfig)
    lambda_range: LambdaRangeConfig = Field(default_factory=LambdaRangeConfig)


class Config(BaseModel):
    """Main configuration container."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    prompting: PromptConfig = Field(default_factory=PromptConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    equation: EquationConfig = Field(default_factory=EquationConfig)


def load_config(config_path: Path | str) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Config object with all settings.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raw_config = {}

    return Config(**raw_config)


def save_config(config: Config, config_path: Path | str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration object to save.
        config_path: Path to save the configuration file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
