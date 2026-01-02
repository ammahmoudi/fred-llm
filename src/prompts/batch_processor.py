"""
Batch prompt processor for generating prompts from datasets.

Handles reading CSV files, generating prompts, and saving to JSONL format.
"""

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.prompts.base import EquationData, GeneratedPrompt, PromptStyle
from src.prompts.factory import create_prompt_style
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BatchPromptProcessor:
    """Process datasets and generate prompts in batch."""

    def __init__(
        self,
        prompt_style: PromptStyle,
        output_dir: Path | str,
        include_ground_truth: bool = True,
    ):
        """
        Initialize batch processor.

        Args:
            prompt_style: Configured PromptStyle instance.
            output_dir: Directory to save generated prompts.
            include_ground_truth: Whether to include solutions in output.
        """
        self.prompt_style = prompt_style
        self.output_dir = Path(output_dir)
        self.include_ground_truth = include_ground_truth

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized BatchPromptProcessor, output: {self.output_dir}")

    def load_equations_from_csv(self, csv_path: Path | str) -> list[EquationData]:
        """
        Load equations from CSV file.

        Args:
            csv_path: Path to CSV file with columns: u, f, kernel, lambda_val, a, b.
                     Optional edge case columns: has_solution, solution_type.

        Returns:
            List of EquationData objects.
        """
        csv_path = Path(csv_path)
        logger.info(f"Loading equations from {csv_path}")

        df = pd.read_csv(csv_path)

        # Validate required columns
        required_cols = ["u", "f", "kernel", "lambda_val", "a", "b"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        # Optional edge case columns (only essential ones)
        edge_case_cols = ["has_solution", "solution_type"]

        # Convert to EquationData objects
        equations = []
        for idx, row in df.iterrows():
            # Build base equation data
            eq_kwargs = {
                "u": str(row["u"]),
                "f": str(row["f"]),
                "kernel": str(row["kernel"]),
                "lambda_val": float(row["lambda_val"]),
                "a": float(row["a"]),
                "b": float(row["b"]),
                "equation_id": f"eq_{idx}",
            }

            # Add optional edge case fields if present
            for col in edge_case_cols:
                if col in df.columns and pd.notna(row[col]):
                    value = row[col]
                    if col == "has_solution":
                        eq_kwargs[col] = bool(value) if isinstance(value, (bool, int)) else str(value).lower() == "true"
                    else:
                        eq_kwargs[col] = str(value)

            eq = EquationData(**eq_kwargs)
            equations.append(eq)

        logger.info(f"Loaded {len(equations)} equations")
        return equations

    def save_prompts_jsonl(
        self,
        prompts: list[GeneratedPrompt],
        output_file: Path | str,
    ) -> None:
        """
        Save generated prompts to JSONL file.

        Args:
            prompts: List of generated prompts.
            output_file: Output JSONL file path.
        """
        output_file = Path(output_file)
        logger.info(f"Saving {len(prompts)} prompts to {output_file}")

        with open(output_file, "w", encoding="utf-8") as f:
            for prompt in prompts:
                record = {
                    "equation_id": prompt.equation_id,
                    "prompt": prompt.prompt,
                    "style": prompt.style,
                    "format_type": prompt.format_type,
                }

                if prompt.ground_truth is not None:
                    record["ground_truth"] = prompt.ground_truth

                if prompt.metadata:
                    record["metadata"] = prompt.metadata

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Saved prompts to {output_file}")

    def process_dataset(
        self,
        input_csv: Path | str,
        output_name: str | None = None,
        format_type: str = "infix",
        show_progress: bool = True,
    ) -> Path:
        """
        Process a complete dataset: load CSV → generate prompts → save JSONL.

        Args:
            input_csv: Input CSV file path.
            output_name: Output filename (without extension). Auto-generated if None.
            format_type: Format type (infix, latex, rpn).
            show_progress: Whether to show progress bar.

        Returns:
            Path to saved JSONL file.
        """
        input_csv = Path(input_csv)

        # Auto-generate output name
        if output_name is None:
            base_name = input_csv.stem
            output_name = f"{base_name}_{self.prompt_style.style_name}"

        output_file = self.output_dir / f"{output_name}.jsonl"

        # Load equations
        equations = self.load_equations_from_csv(input_csv)

        # Generate prompts with progress bar
        logger.info(f"Generating prompts in {self.prompt_style.style_name} style")
        prompts = []

        iterator = (
            tqdm(equations, desc="Generating prompts") if show_progress else equations
        )

        for eq in iterator:
            prompt = self.prompt_style.generate(
                eq,
                format_type=format_type,
                include_ground_truth=self.include_ground_truth,
            )
            prompts.append(prompt)

        # Save to JSONL
        self.save_prompts_jsonl(prompts, output_file)

        return output_file

    def process_multiple_datasets(
        self,
        input_files: list[Path | str],
        format_types: list[str] | None = None,
        show_progress: bool = True,
    ) -> list[Path]:
        """
        Process multiple datasets in batch.

        Args:
            input_files: List of input CSV files.
            format_types: Format type for each file (infix, latex, rpn).
                         If None, infers from filename.
            show_progress: Whether to show progress bar.

        Returns:
            List of paths to saved JSONL files.
        """
        if format_types is None:
            format_types = self._infer_format_types(input_files)

        if len(format_types) != len(input_files):
            raise ValueError("format_types must match length of input_files")

        output_files = []
        for input_file, format_type in zip(input_files, format_types):
            logger.info(f"Processing {input_file} ({format_type} format)")
            output_file = self.process_dataset(
                input_file,
                format_type=format_type,
                show_progress=show_progress,
            )
            output_files.append(output_file)

        return output_files

    def _infer_format_types(self, input_files: list[Path | str]) -> list[str]:
        """
        Infer format types from filenames.

        Args:
            input_files: List of input file paths.

        Returns:
            List of format types (infix, latex, rpn).
        """
        format_types = []
        for file_path in input_files:
            file_str = str(file_path).lower()
            if "latex" in file_str:
                format_types.append("latex")
            elif "rpn" in file_str:
                format_types.append("rpn")
            else:
                format_types.append("infix")

        return format_types


def create_processor(
    style: str = "chain-of-thought",
    output_dir: Path | str = "data/prompts",
    include_ground_truth: bool = True,
    include_examples: bool = True,
    num_examples: int = 2,
    edge_case_mode: str = "none",
) -> BatchPromptProcessor:
    """
    Factory function to create a batch prompt processor.

    Args:
        style: Prompting style.
        output_dir: Output directory for prompts.
        include_ground_truth: Whether to include solutions.
        include_examples: Whether to include few-shot examples.
        num_examples: Number of few-shot examples.
        edge_case_mode: Edge case handling mode (none, guardrails, hints).

    Returns:
        Configured BatchPromptProcessor instance.
    """
    prompt_style = create_prompt_style(
        style=style,
        include_examples=include_examples,
        num_examples=num_examples,
        edge_case_mode=edge_case_mode,
    )

    return BatchPromptProcessor(
        prompt_style=prompt_style,
        output_dir=output_dir,
        include_ground_truth=include_ground_truth,
    )
