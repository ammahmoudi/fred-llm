"""
Tests for adaptive pipeline configuration.

Tests validation, defaults, path resolution, and conflict detection.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.adaptive_config import (
    AdaptiveDatasetConfig,
    AdaptivePipelineConfig,
    PreparedDatasetConfig,
    PreparedPromptsConfig,
    PromptGenerationConfig,
    RawDatasetConfig,
)


class TestConfigValidation:
    """Test basic config validation."""

    def test_requires_at_least_one_dataset_source(self):
        """Config must have raw, prepared, or prompts."""
        with pytest.raises(ValidationError, match="Must specify at least one"):
            AdaptivePipelineConfig(dataset=AdaptiveDatasetConfig())

    def test_raw_dataset_minimal_config(self):
        """Raw dataset with minimal config."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                raw=RawDatasetConfig(path=Path("data/raw/dataset.csv"))
            )
        )
        assert config.get_automation_level() == "full"

    def test_prepared_dataset_minimal_config(self):
        """Prepared dataset with minimal config."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                prepared=PreparedDatasetConfig(train_path=Path("data/train.csv"))
            )
        )
        assert config.get_automation_level() == "partial"

    def test_prompts_dataset_minimal_config(self):
        """Pre-generated prompts with minimal config."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                prompts=PreparedPromptsConfig(
                    prompts_dir=Path("data/prompts/basic"), style="basic"
                )
            )
        )
        assert config.get_automation_level() == "manual"


class TestConflictDetection:
    """Test conflict detection between config sections."""

    def test_cannot_have_both_prompting_and_prompts(self):
        """Error if both prompting (generate) and prompts (use existing) specified."""
        with pytest.raises(
            ValidationError, match="Cannot specify both 'prompting'.*and 'prompts'"
        ):
            AdaptivePipelineConfig(
                dataset=AdaptiveDatasetConfig(
                    prepared=PreparedDatasetConfig(train_path=Path("data/train.csv")),
                    prompting=PromptGenerationConfig(style="basic"),
                    prompts=PreparedPromptsConfig(
                        prompts_dir=Path("data/prompts/basic"), style="basic"
                    ),
                )
            )

    def test_cannot_have_prompting_input_equals_output(self):
        """Error if prompting input_dir and output_dir are the same."""
        with pytest.raises(
            ValidationError, match="input_dir and output_dir cannot be the same"
        ):
            AdaptivePipelineConfig(
                dataset=AdaptiveDatasetConfig(
                    prepared=PreparedDatasetConfig(train_path=Path("data/train.csv")),
                    prompting=PromptGenerationConfig(
                        input_dir=Path("data/prompts"),
                        output_dir=Path("data/prompts"),
                        style="basic",
                    ),
                )
            )


class TestPathResolution:
    """Test smart path resolution and defaults."""

    def test_raw_dataset_default_output_path(self):
        """Raw dataset without output_dir gets timestamped default."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                raw=RawDatasetConfig(path=Path("data/raw/dataset.csv"))
            )
        )
        paths = config.resolve_paths()

        assert "prepared_data" in paths
        assert paths["prepared_data"].parts[0] == "data"
        assert paths["prepared_data"].parts[1] == "processed"
        assert "run_" in str(paths["prepared_data"])

    def test_raw_dataset_explicit_output_path(self):
        """Raw dataset with explicit output_dir uses it."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                raw=RawDatasetConfig(
                    path=Path("data/raw/dataset.csv"),
                    output_dir=Path("data/processed/my_prep"),
                )
            )
        )
        paths = config.resolve_paths()

        assert paths["prepared_data"] == Path("data/processed/my_prep")

    def test_prompting_default_output_path(self):
        """Prompting without output_dir gets data/prompts/<style>."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                prepared=PreparedDatasetConfig(train_path=Path("data/train.csv")),
                prompting=PromptGenerationConfig(style="chain-of-thought"),
            )
        )
        paths = config.resolve_paths()

        assert paths["prompts"] == Path("data/prompts/chain-of-thought")

    def test_prompting_explicit_output_path(self):
        """Prompting with explicit output_dir uses it."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                prepared=PreparedDatasetConfig(train_path=Path("data/train.csv")),
                prompting=PromptGenerationConfig(
                    style="few-shot", output_dir=Path("data/my_prompts")
                ),
            )
        )
        paths = config.resolve_paths()

        assert paths["prompts"] == Path("data/my_prompts")

    def test_prompting_input_chains_from_prepared(self):
        """Prompting input_dir defaults to prepared data location."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                prepared=PreparedDatasetConfig(
                    train_path=Path("data/processed/exp1/train.csv")
                ),
                prompting=PromptGenerationConfig(style="basic"),
            )
        )
        paths = config.resolve_paths()

        assert paths["prompting_input"] == Path("data/processed/exp1")

    def test_prompting_input_chains_from_raw_output(self):
        """Prompting input_dir defaults to raw preparation output."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                raw=RawDatasetConfig(
                    path=Path("data/raw/dataset.csv"),
                    output_dir=Path("data/processed/prep1"),
                ),
                prompting=PromptGenerationConfig(style="basic"),
            )
        )
        paths = config.resolve_paths()

        # Raw output becomes prompting input
        assert paths["prepared_data"] == Path("data/processed/prep1")
        assert paths["prompting_input"] == Path("data/processed/prep1")

    def test_explicit_prompting_input_overrides_default(self):
        """Explicit prompting input_dir overrides chaining."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                prepared=PreparedDatasetConfig(
                    train_path=Path("data/processed/exp1/train.csv")
                ),
                prompting=PromptGenerationConfig(
                    style="basic", input_dir=Path("data/processed/exp2")
                ),
            )
        )
        paths = config.resolve_paths()

        # Should use explicit, not prepared location
        assert paths["prompting_input"] == Path("data/processed/exp2")


class TestAutomationLevels:
    """Test automation level detection."""

    def test_full_automation_level(self):
        """Raw dataset → full automation."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                raw=RawDatasetConfig(path=Path("data/raw/dataset.csv"))
            )
        )
        assert config.get_automation_level() == "full"

    def test_partial_automation_level(self):
        """Prepared dataset → partial automation."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                prepared=PreparedDatasetConfig(train_path=Path("data/train.csv"))
            )
        )
        assert config.get_automation_level() == "partial"

    def test_manual_automation_level(self):
        """Pre-generated prompts → manual control."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                prompts=PreparedPromptsConfig(
                    prompts_dir=Path("data/prompts/basic"), style="basic"
                )
            )
        )
        assert config.get_automation_level() == "manual"

    def test_prompts_override_prepared_for_level(self):
        """If prompts specified, level is manual even if prepared also exists."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                prompts=PreparedPromptsConfig(
                    prompts_dir=Path("data/prompts/basic"), style="basic"
                )
            )
        )
        # Should be manual because prompts take priority
        assert config.get_automation_level() == "manual"


class TestCompleteWorkflows:
    """Test complete workflow configurations."""

    def test_preparation_only_workflow(self):
        """Raw → prepare → STOP (no prompts, no inference)."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                raw=RawDatasetConfig(
                    path=Path("data/raw/dataset.csv"),
                    output_dir=Path("data/processed/prep"),
                )
                # No prompting section = preparation only
            )
        )
        paths = config.resolve_paths()

        assert "prepared_data" in paths
        assert "prompts" not in paths  # No prompts

    def test_prompts_generation_workflow(self):
        """Prepared → generate prompts → STOP (no inference)."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                prepared=PreparedDatasetConfig(
                    train_path=Path("data/processed/train.csv")
                ),
                prompting=PromptGenerationConfig(
                    style="few-shot", output_dir=Path("data/prompts/exp1")
                ),
            )
        )
        paths = config.resolve_paths()

        assert paths["prompts"] == Path("data/prompts/exp1")

    def test_full_pipeline_workflow(self):
        """Raw → prepare → generate prompts → inference."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                raw=RawDatasetConfig(
                    path=Path("data/raw/dataset.csv"),
                    output_dir=Path("data/processed/full"),
                ),
                prompting=PromptGenerationConfig(
                    style="chain-of-thought", output_dir=Path("data/prompts/full")
                ),
            )
        )
        paths = config.resolve_paths()

        # All stages present
        assert paths["prepared_data"] == Path("data/processed/full")
        assert paths["prompting_input"] == Path("data/processed/full")
        assert paths["prompts"] == Path("data/prompts/full")


class TestConfigDefaults:
    """Test default values."""

    def test_model_defaults(self):
        """Model config is optional - None when not specified."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                raw=RawDatasetConfig(path=Path("data/raw/dataset.csv"))
            )
        )
        # Model is now optional - should be None if not explicitly set
        assert config.model is None

    def test_evaluation_defaults(self):
        """Evaluation config is optional - None when not specified."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                raw=RawDatasetConfig(path=Path("data/raw/dataset.csv"))
            )
        )
        # Evaluation is now optional - should be None if not explicitly set
        assert config.evaluation is None

    def test_output_defaults(self):
        """Output config has sensible defaults."""
        config = AdaptivePipelineConfig(
            dataset=AdaptiveDatasetConfig(
                raw=RawDatasetConfig(path=Path("data/raw/dataset.csv"))
            )
        )
        assert config.output.dir == Path("outputs")
        assert config.output.save_predictions is True
        assert config.output.save_predictions is True
