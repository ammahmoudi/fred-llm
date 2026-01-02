"""Tests for prompt generation modules."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.prompts import (
    BasicPromptStyle,
    BatchPromptProcessor,
    ChainOfThoughtPromptStyle,
    EdgeCaseMode,
    EquationData,
    FewShotPromptStyle,
    GeneratedPrompt,
    ToolAssistedPromptStyle,
    create_edge_case_mode,
    create_processor,
    create_prompt_style,
)


class TestEquationData:
    """Test EquationData dataclass."""

    def test_equation_data_creation(self) -> None:
        """Test creating EquationData object."""
        eq = EquationData(
            u="x",
            f="x",
            kernel="x*t",
            lambda_val=1.0,
            a=0.0,
            b=1.0,
            equation_id="eq_1",
        )

        assert eq.u == "x"
        assert eq.f == "x"
        assert eq.kernel == "x*t"
        assert eq.lambda_val == 1.0
        assert eq.a == 0.0
        assert eq.b == 1.0
        assert eq.equation_id == "eq_1"

    def test_equation_data_optional_id(self) -> None:
        """Test EquationData with optional ID."""
        eq = EquationData(
            u="x",
            f="x",
            kernel="x*t",
            lambda_val=1.0,
            a=0.0,
            b=1.0,
        )

        assert eq.equation_id is None


class TestPromptStyles:
    """Test prompt style classes."""

    def test_init_valid_styles(self) -> None:
        """Test initialization with valid styles."""
        styles = {
            "basic": BasicPromptStyle,
            "chain-of-thought": ChainOfThoughtPromptStyle,
            "few-shot": FewShotPromptStyle,
            "tool-assisted": ToolAssistedPromptStyle,
        }

        for style_name, style_class in styles.items():
            style = style_class()
            assert style.style_name == style_name

    def test_factory_invalid_style(self) -> None:
        """Test factory with invalid style."""
        with pytest.raises(ValueError, match="Unknown prompt style"):
            create_prompt_style(style="invalid_style")

    def test_generate_basic_prompt(self) -> None:
        """Test generating basic prompt."""
        style = create_prompt_style("basic")

        eq = EquationData(
            u="3*x/2",
            f="x",
            kernel="x*t",
            lambda_val=1.0,
            a=0.0,
            b=1.0,
            equation_id="eq_test",
        )

        prompt = style.generate(eq, include_ground_truth=True)

        assert prompt.equation_id == "eq_test"
        assert prompt.style == "basic"
        assert "x*t" in prompt.prompt
        assert "1.0" in prompt.prompt or "1" in prompt.prompt
        assert prompt.ground_truth == "3*x/2"
        assert prompt.metadata["kernel"] == "x*t"

    def test_generate_chain_of_thought_prompt(self) -> None:
        """Test generating chain-of-thought prompt."""
        style = create_prompt_style("chain-of-thought")

        eq = EquationData(
            u="sin(x)",
            f="sin(x)",
            kernel="cos(x*t)",
            lambda_val=0.5,
            a=-1.0,
            b=1.0,
        )

        prompt = style.generate(eq)

        assert prompt.style == "chain-of-thought"
        assert "step" in prompt.prompt.lower()
        assert "cos(x*t)" in prompt.prompt
        assert "0.5" in prompt.prompt

    def test_generate_few_shot_prompt(self) -> None:
        """Test generating few-shot prompt with examples."""
        style = create_prompt_style("few-shot", include_examples=True, num_examples=2)

        eq = EquationData(
            u="exp(x)",
            f="exp(x)",
            kernel="exp(x+t)",
            lambda_val=1.0,
            a=0.0,
            b=1.0,
        )

        prompt = style.generate(eq)

        assert prompt.style == "few-shot"
        assert "Example" in prompt.prompt
        assert "exp(x+t)" in prompt.prompt

    def test_generate_without_ground_truth(self) -> None:
        """Test generating prompt without ground truth."""
        style = create_prompt_style("basic")

        eq = EquationData(
            u="x**2",
            f="x**2",
            kernel="x*t",
            lambda_val=1.0,
            a=0.0,
            b=1.0,
        )

        prompt = style.generate(eq, include_ground_truth=False)

        assert prompt.ground_truth is None

    def test_generate_batch(self) -> None:
        """Test batch generation."""
        style = create_prompt_style("basic")

        equations = [
            EquationData(
                u=f"x**{i}",
                f=f"x**{i}",
                kernel="x*t",
                lambda_val=1.0,
                a=0.0,
                b=1.0,
                equation_id=f"eq_{i}",
            )
            for i in range(5)
        ]

        prompts = style.generate_batch(equations)

        assert len(prompts) == 5
        for i, prompt in enumerate(prompts):
            assert prompt.equation_id == f"eq_{i}"
            assert prompt.style == "basic"


class TestBatchPromptProcessor:
    """Test BatchPromptProcessor class."""

    def test_init(self, tmp_path: Path) -> None:
        """Test processor initialization."""
        style = create_prompt_style("basic")
        processor = BatchPromptProcessor(
            prompt_style=style,
            output_dir=tmp_path,
            include_ground_truth=True,
        )

        assert processor.output_dir == tmp_path
        assert processor.include_ground_truth is True
        assert tmp_path.exists()

    def test_load_equations_from_csv(self, tmp_path: Path) -> None:
        """Test loading equations from CSV."""
        # Create test CSV
        csv_path = tmp_path / "test_equations.csv"
        df = pd.DataFrame(
            {
                "u": ["x", "x**2", "sin(x)"],
                "f": ["x", "x**2", "sin(x)"],
                "kernel": ["x*t", "x*t", "cos(x*t)"],
                "lambda_val": [1.0, 0.5, 2.0],
                "a": [0.0, -1.0, 0.0],
                "b": [1.0, 1.0, 3.14],
            }
        )
        df.to_csv(csv_path, index=False)

        style = create_prompt_style("basic")
        processor = BatchPromptProcessor(
            prompt_style=style,
            output_dir=tmp_path,
        )

        equations = processor.load_equations_from_csv(csv_path)

        assert len(equations) == 3
        assert equations[0].u == "x"
        assert equations[1].lambda_val == 0.5
        assert equations[2].kernel == "cos(x*t)"

    def test_load_equations_missing_columns(self, tmp_path: Path) -> None:
        """Test loading CSV with missing columns."""
        csv_path = tmp_path / "bad_equations.csv"
        df = pd.DataFrame(
            {
                "u": ["x"],
                "f": ["x"],
                # Missing kernel, lambda_val, a, b
            }
        )
        df.to_csv(csv_path, index=False)

        style = create_prompt_style("basic")
        processor = BatchPromptProcessor(
            prompt_style=style,
            output_dir=tmp_path,
        )

        with pytest.raises(ValueError, match="missing required columns"):
            processor.load_equations_from_csv(csv_path)

    def test_save_prompts_jsonl(self, tmp_path: Path) -> None:
        """Test saving prompts to JSONL."""
        style = create_prompt_style("basic")
        processor = BatchPromptProcessor(
            prompt_style=style,
            output_dir=tmp_path,
        )

        # Create test prompts
        eq = EquationData(
            u="x",
            f="x",
            kernel="x*t",
            lambda_val=1.0,
            a=0.0,
            b=1.0,
            equation_id="eq_1",
        )
        prompts = [style.generate(eq, include_ground_truth=True)]

        output_file = tmp_path / "test_prompts.jsonl"
        processor.save_prompts_jsonl(prompts, output_file)

        assert output_file.exists()

        # Read and verify
        with open(output_file, encoding="utf-8") as f:
            line = f.readline()
            data = json.loads(line)

            assert data["equation_id"] == "eq_1"
            assert data["style"] == "basic"
            assert data["format_type"] == "infix"
            assert "prompt" in data
            assert "ground_truth" in data
            assert "metadata" in data

    def test_process_dataset(self, tmp_path: Path) -> None:
        """Test processing complete dataset."""
        # Create test CSV
        csv_path = tmp_path / "test_data.csv"
        df = pd.DataFrame(
            {
                "u": ["x", "x**2"],
                "f": ["x", "x**2"],
                "kernel": ["x*t", "x*t"],
                "lambda_val": [1.0, 1.0],
                "a": [0.0, 0.0],
                "b": [1.0, 1.0],
            }
        )
        df.to_csv(csv_path, index=False)

        style = create_prompt_style("basic")
        processor = BatchPromptProcessor(
            prompt_style=style,
            output_dir=tmp_path,
        )

        output_file = processor.process_dataset(
            input_csv=csv_path,
            format_type="infix",
            show_progress=False,
        )

        assert output_file.exists()
        assert output_file.suffix == ".jsonl"

        # Verify content
        with open(output_file, encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2

    def test_infer_format_types(self, tmp_path: Path) -> None:
        """Test format type inference from filenames."""
        style = create_prompt_style("basic")
        processor = BatchPromptProcessor(
            prompt_style=style,
            output_dir=tmp_path,
        )

        files = [
            "train_infix.csv",
            "train_latex.csv",
            "train_rpn.csv",
            "test.csv",
        ]

        formats = processor._infer_format_types(files)

        assert formats == ["infix", "latex", "rpn", "infix"]


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_processor(self, tmp_path: Path) -> None:
        """Test create_processor factory."""
        processor = create_processor(
            style="chain-of-thought",
            output_dir=tmp_path,
            include_ground_truth=False,
            include_examples=True,
            num_examples=3,
        )

        assert isinstance(processor, BatchPromptProcessor)
        assert processor.prompt_style.style_name == "chain-of-thought"
        assert processor.include_ground_truth is False

    def test_create_prompt_style_basic(self) -> None:
        """Test creating basic style."""
        style = create_prompt_style("basic")
        assert isinstance(style, BasicPromptStyle)
        assert style.style_name == "basic"

    def test_create_prompt_style_few_shot(self) -> None:
        """Test creating few-shot style with parameters."""
        style = create_prompt_style("few-shot", include_examples=True, num_examples=3)
        assert isinstance(style, FewShotPromptStyle)
        assert style.include_examples is True
        assert style.num_examples == 3


class TestEdgeCaseMode:
    """Test edge case mode functionality."""

    def test_edge_case_mode_none(self) -> None:
        """Test mode='none' adds no guardrails."""
        mode = EdgeCaseMode(mode="none")
        assert mode.mode == "none"
        assert mode.hint_fields is None

    def test_edge_case_mode_guardrails(self) -> None:
        """Test mode='guardrails' is valid."""
        mode = EdgeCaseMode(mode="guardrails")
        assert mode.mode == "guardrails"

    def test_edge_case_mode_hints_default_fields(self) -> None:
        """Test mode='hints' with default fields."""
        mode = EdgeCaseMode(mode="hints")
        assert mode.mode == "hints"
        # Should have all available fields (only 2 now)
        assert "has_solution" in mode.hint_fields
        assert "solution_type" in mode.hint_fields
        assert len(mode.hint_fields) == 2

    def test_edge_case_mode_hints_custom_fields(self) -> None:
        """Test mode='hints' with custom fields."""
        mode = EdgeCaseMode(mode="hints", hint_fields=["has_solution"])
        assert mode.hint_fields == ["has_solution"]

    def test_edge_case_mode_invalid_mode(self) -> None:
        """Test invalid mode raises error."""
        with pytest.raises(ValueError, match="Invalid mode"):
            EdgeCaseMode(mode="invalid")

    def test_edge_case_mode_invalid_fields(self) -> None:
        """Test invalid hint fields raise error."""
        with pytest.raises(ValueError, match="Invalid hint fields"):
            EdgeCaseMode(mode="hints", hint_fields=["invalid_field"])

    def test_create_edge_case_mode_factory(self) -> None:
        """Test create_edge_case_mode factory function."""
        mode = create_edge_case_mode(mode="guardrails")
        assert isinstance(mode, EdgeCaseMode)
        assert mode.mode == "guardrails"

    def test_prompt_style_with_edge_case_mode(self) -> None:
        """Test creating prompt style with edge case mode."""
        style = create_prompt_style("basic", edge_case_mode="guardrails")
        assert style.edge_case_mode.mode == "guardrails"

    def test_generate_prompt_with_guardrails(self) -> None:
        """Test prompt generation includes guardrails text."""
        style = create_prompt_style("basic", edge_case_mode="guardrails")
        eq = EquationData(
            u="x",
            f="x",
            kernel="x*t",
            lambda_val=1.0,
            a=0.0,
            b=1.0,
        )
        prompt = style.generate(eq)
        assert (
            "no closed-form solution" in prompt.prompt.lower()
            or "no solution" in prompt.prompt.lower()
        )

    def test_equation_data_with_edge_case_fields(self) -> None:
        """Test EquationData with edge case fields."""
        eq = EquationData(
            u="x",
            f="x",
            kernel="x*t",
            lambda_val=1.0,
            a=0.0,
            b=1.0,
            has_solution=False,
            solution_type="none",
        )
        assert eq.has_solution is False
        assert eq.solution_type == "none"

    def test_generate_prompt_with_hints(self) -> None:
        """Test prompt generation includes hints from equation data."""
        style = create_prompt_style(
            "basic",
            edge_case_mode="hints",
            hint_fields=["has_solution", "solution_type"],
        )
        eq = EquationData(
            u="x",
            f="x",
            kernel="x*t",
            lambda_val=1.0,
            a=0.0,
            b=1.0,
            has_solution=False,
            solution_type="none",
        )
        prompt = style.generate(eq)
        assert "Has solution: No" in prompt.prompt
        assert "none" in prompt.prompt

    def test_metadata_includes_edge_case_info(self) -> None:
        """Test metadata includes edge case information."""
        style = create_prompt_style("basic")
        eq = EquationData(
            u="x",
            f="x",
            kernel="x*t",
            lambda_val=1.0,
            a=0.0,
            b=1.0,
            has_solution=False,
            solution_type="numerical",
        )
        prompt = style.generate(eq)
        assert prompt.metadata["has_solution"] is False
        assert prompt.metadata["solution_type"] == "numerical"
