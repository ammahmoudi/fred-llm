"""
Adaptive pipeline orchestrator.

Smart pipeline that automatically determines what preparation steps are needed
based on the configuration and available files.
"""

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.adaptive_config import AdaptivePipelineConfig
from src.utils.logging_utils import get_logger

console = Console()
logger = get_logger(__name__)


class AdaptivePipeline:
    """
    Adaptive pipeline that fills in missing preparation steps automatically.

    Supports three automation levels:
    1. Full: raw data → augment → split → convert → prompts → inference
    2. Partial: pre-split data → prompts → inference
    3. Manual: pre-generated prompts → inference
    """

    def __init__(self, config: AdaptivePipelineConfig):
        """Initialize adaptive pipeline."""
        self.config = config
        self.automation_level = config.get_automation_level()
        self.paths = config.resolve_paths()  # Resolve all paths upfront

        console.print(f"\n[bold blue]Adaptive Pipeline[/bold blue]")
        console.print(f"Automation Level: [cyan]{self.automation_level}[/cyan]")
        console.print(f"Resolved Paths: [dim]{self.paths}[/dim]\n")

    def run(self, dry_run: bool = False) -> dict[str, Any]:
        """
        Run the adaptive pipeline.

        Args:
            dry_run: If True, show what would be done without executing

        Returns:
            Pipeline results
        """
        if dry_run:
            return self._show_execution_plan()

        results = {}

        # Step 1: Prepare dataset (if needed)
        if self.automation_level == "full":
            console.print(
                Panel("[bold]Step 1: Dataset Preparation[/bold]", style="blue")
            )
            prepared_paths = self._prepare_dataset()
            results["prepared_data"] = prepared_paths
        elif self.automation_level == "partial":
            console.print(
                Panel("[bold]Step 1: Loading Pre-split Data[/bold]", style="blue")
            )
            prepared_paths = self._load_prepared_data()
            results["prepared_data"] = prepared_paths
        else:  # manual
            console.print(
                Panel(
                    "[bold]Step 1: Loading Pre-generated Prompts[/bold]", style="blue"
                )
            )
            prompts_path = self._load_prompts()
            results["prompts"] = prompts_path
            prepared_paths = None

        # Step 2: Generate or load prompts (if configured)
        if self.config.dataset.prompting and prepared_paths:
            console.print(Panel("[bold]Step 2: Prompt Generation[/bold]", style="blue"))
            prompts = self._generate_prompts(prepared_paths)
            results["prompts"] = prompts
        elif not self.config.dataset.prompts and prepared_paths:
            console.print(
                Panel("[bold]Step 2: Skipping Prompt Generation[/bold]", style="yellow")
            )
            console.print("[cyan]Preparation complete. Prompts not configured.[/cyan]")
            return results

        # Step 3: Run LLM inference (if prompts available AND model configured)
        if results.get("prompts") and self.config.model:
            console.print(Panel("[bold]Step 3: LLM Inference[/bold]", style="blue"))
            predictions = self._run_inference()
            results["predictions"] = predictions

            # Step 4: Evaluate results
            console.print(Panel("[bold]Step 4: Evaluation[/bold]", style="blue"))
            metrics = self._evaluate(predictions)
            results["metrics"] = metrics
        else:
            if not self.config.model:
                console.print(
                    Panel(
                        "[bold]Pipeline Complete[/bold] (No model configured - stopping before inference)",
                        style="green",
                    )
                )
            else:
                console.print(Panel("[bold]Pipeline Complete[/bold]", style="green"))

        return results

    def _show_execution_plan(self) -> dict[str, Any]:
        """Show what the pipeline would do without executing."""
        console.print("\n[bold yellow]📋 Execution Plan (Dry Run)[/bold yellow]\n")

        plan = []

        if self.automation_level == "full":
            raw_config = self.config.dataset.raw
            output_dir = self.paths.get("prepared_data", "auto-generated")
            plan.extend(
                [
                    f"1. Load raw dataset: {raw_config.path}",
                    f"   • Output: {output_dir}",
                    f"   • Augment: {'Yes' if raw_config.augment else 'No'}",
                    f"   • Strategies: {raw_config.augment_strategies or 'default'}",
                    f"   • Split: {raw_config.split_ratios}",
                    f"   • Convert to: {raw_config.convert_formats}",
                ]
            )

            # Only show prompting if configured
            if self.config.dataset.prompting:
                prompts_output = self.paths.get("prompts", "auto-generated")
                prompts_input = self.paths.get("prompting_input", output_dir)
                plan.extend(
                    [
                        "2. Generate prompts on-the-fly",
                        f"   • Input: {prompts_input}",
                        f"   • Output: {prompts_output}",
                        f"   • Style: {self.config.dataset.prompting.style}",
                        f"   • Edge case mode: {self.config.dataset.prompting.edge_case_mode}",
                    ]
                )
            else:
                plan.append("2. Skip prompt generation (preparation only)")

        elif self.automation_level == "partial":
            prep_config = self.config.dataset.prepared
            plan.extend(
                [
                    f"1. Load pre-split data:",
                    f"   • Train: {prep_config.train_path}",
                    f"   • Val: {prep_config.val_path}",
                    f"   • Test: {prep_config.test_path}",
                    f"   • Format: {prep_config.format or 'auto-detect'}",
                ]
            )

            if self.config.dataset.prompting:
                prompts_output = self.paths.get("prompts", "auto-generated")
                prompts_input = self.paths.get(
                    "prompting_input", prep_config.train_path.parent
                )
                plan.extend(
                    [
                        "2. Generate prompts on-the-fly",
                        f"   • Input: {prompts_input}",
                        f"   • Output: {prompts_output}",
                        f"   • Style: {self.config.dataset.prompting.style}",
                    ]
                )
            else:
                plan.append("2. Skip prompt generation (prompts only)")

        else:  # manual
            prompts_config = self.config.dataset.prompts
            plan.extend(
                [
                    f"1. Load pre-generated prompts:",
                    f"   • Directory: {prompts_config.prompts_dir}",
                    f"   • Style: {prompts_config.style}",
                ]
            )

        # Only show inference/evaluation if not preparation-only AND model is configured
        if (
            self.config.dataset.prompting or self.config.dataset.prompts
        ) and self.config.model:
            plan.extend(
                [
                    "3. Run LLM inference",
                    f"   • Provider: {self.config.model.provider}",
                    f"   • Model: {self.config.model.name}",
                ]
            )
            if self.config.evaluation:
                plan.extend(
                    [
                        "4. Evaluate results",
                        f"   • Mode: {self.config.evaluation.mode}",
                        f"   • Output: {self.config.output.dir}",
                    ]
                )
        else:
            if not self.config.model:
                plan.append(
                    "3. Skip inference and evaluation (no model configured - data/prompts only)"
                )
            else:
                plan.append("3. Skip inference and evaluation (preparation only)")

        for step in plan:
            console.print(f"  {step}")

        console.print("\n[yellow]Run without --dry-run to execute[/yellow]\n")

        return {"plan": plan}

    def _prepare_dataset(self) -> dict[str, Path]:
        """Prepare dataset from raw data using the tested prepare_dataset script."""
        raw_config = self.config.dataset.raw
        output_dir = self.paths["prepared_data"]

        console.print(f"[cyan]> Loading raw dataset: {raw_config.path}[/cyan]")
        console.print(f"[cyan]> Output directory: {output_dir}[/cyan]")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Import the tested runner function
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "prepare_dataset", "scripts/prepare_dataset.py"
        )
        prepare_module = importlib.util.module_from_spec(spec)
        sys.modules["prepare_dataset"] = prepare_module
        spec.loader.exec_module(prepare_module)

        # Build arguments for the runner
        runner_args = [
            "--input",
            str(raw_config.path),
            "--output",
            str(output_dir),
        ]

        if raw_config.max_samples:
            runner_args.extend(["--max-samples", str(raw_config.max_samples)])

        if raw_config.augment:
            runner_args.append("--augment")
            runner_args.extend(
                ["--augment-multiplier", str(raw_config.augment_multiplier)]
            )
            if raw_config.augment_strategies:
                runner_args.extend(
                    ["--augment-strategies"] + raw_config.augment_strategies
                )

        if raw_config.convert_formats:
            if not raw_config.convert_formats or raw_config.convert_formats == []:
                runner_args.append("--no-convert")
            else:
                runner_args.extend(["--convert-formats"] + raw_config.convert_formats)

        if raw_config.convert_limit:
            runner_args.extend(["--convert-limit", str(raw_config.convert_limit)])

        if raw_config.validate_data:
            runner_args.append("--validate")

        if raw_config.include_edge_metadata:
            runner_args.append("--include-edge-metadata")

        if raw_config.split:
            runner_args.append("--split")
            train_ratio, val_ratio, _ = raw_config.split_ratios
            runner_args.extend(["--train-ratio", str(train_ratio)])
            runner_args.extend(["--val-ratio", str(val_ratio)])

        runner_args.extend(["--output-format", raw_config.output_format])

        # Call the runner with mocked sys.argv
        original_argv = sys.argv
        try:
            sys.argv = ["prepare_dataset.py"] + runner_args
            prepare_module.main()
        finally:
            sys.argv = original_argv

        # Return paths using first format
        base_format = (
            raw_config.convert_formats[0] if raw_config.convert_formats else "infix"
        )
        return {
            "train": output_dir / f"train_{base_format}.csv",
            "val": output_dir / f"val_{base_format}.csv" if raw_config.split else None,
            "test": output_dir / f"test_{base_format}.csv"
            if raw_config.split
            else None,
        }

    def _load_prepared_data(self) -> dict[str, Path]:
        """Load pre-split dataset."""
        prep_config = self.config.dataset.prepared

        console.print(f"[cyan]> Train: {prep_config.train_path}[/cyan]")
        console.print(f"[cyan]> Val: {prep_config.val_path}[/cyan]")
        console.print(f"[cyan]> Test: {prep_config.test_path}[/cyan]")

        # Auto-detect format
        if not prep_config.format:
            from src.utils.format_detection import auto_detect_format

            detected_format = auto_detect_format(prep_config.train_path)
            console.print(f"[green]OK[/green] Auto-detected format: {detected_format}")

        return {
            "train": prep_config.train_path,
            "val": prep_config.val_path,
            "test": prep_config.test_path,
        }

    def _load_prompts(self) -> Path:
        """Load pre-generated prompts."""
        prompts_config = self.config.dataset.prompts

        console.print(f"[cyan]> Prompts: {prompts_config.prompts_dir}[/cyan]")
        console.print(f"[cyan]> Style: {prompts_config.style}[/cyan]")

        if not prompts_config.prompts_dir.exists():
            console.print(
                f"[red]✗ Prompts directory not found: {prompts_config.prompts_dir}[/red]"
            )
            raise FileNotFoundError(f"Prompts not found: {prompts_config.prompts_dir}")

        console.print("[green]OK[/green] Prompts loaded")
        return prompts_config.prompts_dir

    def _generate_prompts(self, prepared_paths: dict[str, Path]) -> Path:
        """Generate prompts from prepared data using the tested run_prompt_generation script."""
        prompting_config = self.config.dataset.prompting
        output_dir = self.paths["prompts"]

        # Determine input directory
        if prompting_config.input_dir:
            input_dir = prompting_config.input_dir
        else:
            # Use prepared paths
            input_dir = (
                prepared_paths["train"].parent if prepared_paths.get("train") else None
            )

        console.print(f"\n[cyan]> Generating {prompting_config.style} prompts[/cyan]")
        console.print(f"[cyan]> Format: {prompting_config.format}[/cyan]")
        console.print(f"[cyan]> Input: {input_dir}[/cyan]")
        console.print(f"[cyan]> Output: {output_dir}[/cyan]")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Import the tested runner function
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "run_prompt_generation", "scripts/run_prompt_generation.py"
        )
        prompt_module = importlib.util.module_from_spec(spec)
        sys.modules["run_prompt_generation"] = prompt_module
        spec.loader.exec_module(prompt_module)

        # Map our style names to script's style names
        style_mapping = {
            "direct": "basic",
            "few_shot": "few-shot",
            "chain_of_thought": "chain-of-thought",
            "tool_use": "tool-assisted",
        }
        script_style = style_mapping.get(prompting_config.style, prompting_config.style)

        # Build arguments for the runner
        runner_args = [
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--styles",
            script_style,
        ]

        if not prompting_config.include_ground_truth:
            runner_args.append("--no-ground-truth")

        if not prompting_config.include_examples:
            runner_args.append("--no-examples")

        if prompting_config.num_examples:
            runner_args.extend(["--num-examples", str(prompting_config.num_examples)])

        if prompting_config.format and prompting_config.format != "auto":
            runner_args.extend(["--format", prompting_config.format])

        if (
            prompting_config.edge_case_mode
            and prompting_config.edge_case_mode != "none"
        ):
            runner_args.extend(["--edge-case-mode", prompting_config.edge_case_mode])

        # Call the runner with mocked sys.argv
        original_argv = sys.argv
        try:
            sys.argv = ["run_prompt_generation.py"] + runner_args
            prompt_module.main()
        finally:
            sys.argv = original_argv

        console.print(f"\n[bold green]OK - Prompt generation complete![/bold green]")
        console.print(f"[cyan]Output: {output_dir}[/cyan]\n")

        return output_dir

    def _run_inference(self) -> list:
        """Run LLM inference."""
        console.print(f"[cyan]> Running {self.config.model.name}...[/cyan]")

        # TODO: Implement actual inference
        console.print("[yellow]⚠️  LLM inference not yet implemented[/yellow]")

        return []

    def _evaluate(self, predictions: list) -> dict:
        """Evaluate predictions."""
        console.print(
            f"[cyan]→ Evaluating ({self.config.evaluation.mode} mode)...[/cyan]"
        )

        # TODO: Implement actual evaluation
        console.print("[yellow]⚠️  Evaluation not yet implemented[/yellow]")

        return {}


def load_adaptive_config(config_path: Path) -> AdaptivePipelineConfig:
    """Load and validate adaptive configuration."""
    import yaml

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return AdaptivePipelineConfig(**config_dict)
