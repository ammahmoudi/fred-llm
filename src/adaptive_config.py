"""
Adaptive pipeline configuration schema.

Supports multiple levels of automation:
1. Full automation: raw data → augment → split → convert → prompts → inference
2. Partial automation: pre-split data → prompts → inference
3. Manual control: pre-generated prompts → inference

Smart defaults: Pipeline automatically chains outputs to next stage inputs.
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class RawDatasetConfig(BaseModel):
    """Configuration for raw dataset that needs preparation."""

    path: Path
    """Path to raw dataset (CSV/JSON)"""

    output_dir: Optional[Path] = None
    """Output directory for prepared data. If None, uses data/processed/<timestamp>"""

    max_samples: Optional[int] = None
    """Limit number of samples to process (None = all)"""

    # Augmentation settings
    augment: bool = False
    """Whether to apply augmentation"""

    augment_multiplier: float = 1.15
    """Target dataset size multiplier"""

    augment_strategies: Optional[list[str]] = None
    """Edge case strategies to apply. If None, uses default (substitute, scale, shift)"""

    include_edge_metadata: bool = False
    """Include detailed edge case metadata in output"""

    # Validation settings
    validate_data: bool = True
    """Whether to validate equations"""

    # Splitting settings
    split: bool = True
    """Whether to split into train/val/test"""

    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    """Train/val/test split ratios"""

    seed: int = 42
    """Random seed for reproducible splits"""

    # Format conversion
    convert_formats: list[Literal["infix", "latex", "rpn", "tokenized", "python"]] = [
        "infix"
    ]
    """Which formats to generate"""

    convert_limit: Optional[int] = None
    """Limit number of equations to convert (None = all)"""

    output_format: Literal["json", "csv", "both"] = "both"
    """Output file format (json, csv, or both)"""


class PreparedDatasetConfig(BaseModel):
    """Configuration for pre-prepared dataset files."""

    train_path: Path
    """Path to training data CSV"""

    val_path: Optional[Path] = None
    """Path to validation data CSV"""

    test_path: Optional[Path] = None
    """Path to test data CSV"""

    format: Optional[Literal["infix", "latex", "rpn"]] = None
    """Data format. If None, auto-detected."""

    max_samples: Optional[int] = None
    """Limit number of samples (for testing)"""


class PromptGenerationConfig(BaseModel):
    """Configuration for prompt generation."""

    output_dir: Optional[Path] = None
    """Output directory for generated prompts. If None, uses data/prompts/<style>"""

    input_dir: Optional[Path] = None
    """Input directory (for prepared data). If None, uses output from previous stage."""

    style: Literal["basic", "chain-of-thought", "few-shot", "tool-assisted"] = (
        "chain-of-thought"
    )
    """Prompt style to use"""

    edge_case_mode: Literal["none", "guardrails", "hints"] = "none"
    """How to handle edge cases in prompts"""

    num_examples: int = 2
    """Number of examples for few-shot prompts"""

    include_ground_truth: bool = True
    """Whether to include solutions in prompts"""

    include_examples: bool = True
    """Whether to include few-shot examples (only for few-shot style)"""

    format: Optional[Literal["infix", "latex", "rpn"]] = None
    """Force specific format (None = auto-detect from data)"""


class PreparedPromptsConfig(BaseModel):
    """Configuration for pre-generated prompts."""

    prompts_dir: Path
    """Directory containing pre-generated JSONL prompt files"""

    style: str
    """Prompt style name (must match directory name)"""


class AdaptiveDatasetConfig(BaseModel):
    """Adaptive dataset configuration - supports multiple automation levels."""

    # Option 1: Raw dataset (full automation)
    raw: Optional[RawDatasetConfig] = None
    """Raw dataset configuration. Pipeline will prepare it."""

    # Option 2: Pre-prepared dataset (partial automation)
    prepared: Optional[PreparedDatasetConfig] = None
    """Pre-prepared dataset paths. Pipeline will generate prompts."""

    # Prompting configuration
    prompting: Optional[PromptGenerationConfig] = None
    """Prompt generation settings. If None and no pre-generated prompts, uses defaults."""

    # Option 3: Pre-generated prompts (manual control)
    prompts: Optional[PreparedPromptsConfig] = None
    """Pre-generated prompts. Pipeline will use directly."""

    @model_validator(mode="after")
    def validate_dataset_config(self):
        """Validate dataset configuration for conflicts and requirements."""
        # Check: At least one input method
        if not any([self.raw, self.prepared, self.prompts]):
            raise ValueError(
                "Must specify at least one of: dataset.raw, dataset.prepared, or dataset.prompts"
            )

        # Check: Conflict if both prepared data path and raw output are same
        if self.raw and self.prepared:
            if self.raw.output_dir and self.prepared.train_path:
                # Check if they conflict
                if self.raw.output_dir == self.prepared.train_path.parent:
                    raise ValueError(
                        f"Conflict: raw.output_dir ({self.raw.output_dir}) overlaps with "
                        f"prepared data location ({self.prepared.train_path.parent}). "
                        "Specify only one dataset source."
                    )

        # Check: Conflict if both prompting input and output specified inconsistently
        if self.prompting and self.prompting.input_dir and self.prompting.output_dir:
            if self.prompting.input_dir == self.prompting.output_dir:
                raise ValueError(
                    f"Conflict: prompting.input_dir and output_dir cannot be the same: "
                    f"{self.prompting.input_dir}"
                )

        # Check: Can't have both prompting and prompts
        if self.prompting and self.prompts:
            raise ValueError(
                "Conflict: Cannot specify both 'prompting' (generate prompts) and "
                "'prompts' (use pre-generated prompts). Choose one."
            )

        return self


class ModelConfig(BaseModel):
    """LLM model configuration."""

    provider: Literal["openai", "openrouter", "local"] = "openai"
    name: str = "gpt-4o-mini"
    api_key: Optional[str] = (
        None  # Direct API key (overrides env var based on provider)
    )
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 60


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    mode: Literal["symbolic", "numeric", "both"] = "both"
    symbolic_tolerance: float = 1e-10
    numeric_tolerance: float = 1e-6
    num_test_points: int = 100
    use_math_verify: bool = True
    """Use Math-Verify library for parsing and comparison when available."""
    type_tolerances: dict[str, float] = Field(
        default_factory=lambda: {
            "series": 1e-2,
            "approx_coef": 1e-3,
            "regularized": 1e-3,
        }
    )


class OutputConfig(BaseModel):
    """Output configuration."""

    dir: Path = Path("outputs")
    save_predictions: bool = True
    save_metrics: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class AdaptivePipelineConfig(BaseModel):
    """Complete adaptive pipeline configuration."""

    dataset: AdaptiveDatasetConfig
    model: Optional[ModelConfig] = None
    evaluation: Optional[EvaluationConfig] = None
    output: OutputConfig = Field(default_factory=OutputConfig)

    def get_automation_level(self) -> Literal["full", "partial", "manual"]:
        """Determine the automation level based on configuration."""
        if self.dataset.prompts:
            return "manual"  # Pre-generated prompts
        elif self.dataset.prepared:
            return "partial"  # Pre-split data
        elif self.dataset.raw:
            return "full"  # Raw data
        else:
            raise ValueError("Invalid dataset configuration")

    def resolve_paths(self) -> dict:
        """
        Resolve all paths with smart defaults, chaining outputs to inputs.

        Returns dict with resolved paths for each stage.
        """
        from datetime import datetime

        paths = {}

        # Stage 1: Data preparation output
        if self.dataset.raw:
            if self.dataset.raw.output_dir:
                paths["prepared_data"] = self.dataset.raw.output_dir
            else:
                # Default: data/processed/run_<timestamp>
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                paths["prepared_data"] = Path(f"data/processed/run_{timestamp}")

        # Stage 2: Prompt generation
        if self.dataset.prompting:
            # Input: Use previous stage output or explicit input_dir
            if self.dataset.prompting.input_dir:
                paths["prompting_input"] = self.dataset.prompting.input_dir
            elif self.dataset.prepared:
                # Use prepared data location
                paths["prompting_input"] = self.dataset.prepared.train_path.parent
            elif "prepared_data" in paths:
                # Use output from data preparation
                paths["prompting_input"] = paths["prepared_data"]

            # Output: Use explicit or generate default
            if self.dataset.prompting.output_dir:
                paths["prompts"] = self.dataset.prompting.output_dir
            else:
                # Default: data/prompts/<style>
                style = self.dataset.prompting.style
                paths["prompts"] = Path(f"data/prompts/{style}")

        # Stage 3: Pre-generated prompts
        if self.dataset.prompts:
            paths["prompts"] = self.dataset.prompts.prompts_dir

        return paths

    @classmethod
    def from_yaml(cls, path: Path) -> "AdaptivePipelineConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)
