"""
Adaptive pipeline orchestrator.

Smart pipeline that automatically determines what preparation steps are needed
based on the configuration and available files.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import sympy as sp
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.adaptive_config import AdaptivePipelineConfig
from src.llm.evaluate import SolutionEvaluator
from src.llm.math_verify_adapter import parse_latex_to_sympy
from src.llm.model_runner import ModelRunner
from src.llm.postprocess import parse_llm_output
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

    def _run_inference(self) -> list[dict[str, Any]]:
        """
        Run LLM inference on prompts.

        Returns:
            List of prediction dictionaries with parsed LLM outputs.
        """
        console.print(f"[cyan]> Running {self.config.model.name}...[/cyan]")

        # Load prompts from JSONL files
        prompts_dir = self.paths.get("prompts")
        if not prompts_dir:
            console.print("[red]✗ No prompts directory found[/red]")
            return []

        # Find all JSONL files in the prompts directory (including subdirectories)
        jsonl_files = list(Path(prompts_dir).glob("**/*.jsonl"))
        if not jsonl_files:
            console.print(f"[red]✗ No JSONL files found in {prompts_dir}[/red]")
            return []

        console.print(f"[cyan]> Found {len(jsonl_files)} prompt files[/cyan]")

        # Load all prompts
        all_prompts: list[dict[str, Any]] = []
        for jsonl_file in jsonl_files:
            console.print(f"[dim]  Loading {jsonl_file.name}...[/dim]")
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        prompt_data = json.loads(line)
                        all_prompts.append(prompt_data)

        console.print(f"[green]OK[/green] Loaded {len(all_prompts)} prompts")

        # Initialize cost tracker
        from src.llm.cost_tracker import CostTracker

        run_id = (
            f"{self.config.output.dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        cost_tracker = CostTracker(run_id=run_id)

        # Initialize model runner
        model_config = self.config.model
        runner_kwargs = {
            "model_name": model_config.name,
            "api_key": model_config.api_key,  # None if not set, provider determines env var
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
            "timeout": model_config.timeout,
        }
        # Only add base_url for OpenAI (not OpenRouter, which uses class constant)
        if model_config.provider == "openai" and model_config.base_url:
            runner_kwargs["base_url"] = model_config.base_url
        # Pass reasoning config for reasoning models (o1, o3, GPT-5.x)
        if model_config.reasoning:
            runner_kwargs["reasoning"] = model_config.reasoning.model_dump()

        runner = ModelRunner(provider=model_config.provider, **runner_kwargs)

        # Set cost tracker on the runner
        runner.set_cost_tracker(cost_tracker)

        # Extract prompt texts
        prompt_texts = [p.get("prompt", "") for p in all_prompts]

        console.print(
            f"\n[cyan]> Generating responses for {len(prompt_texts)} prompts...[/cyan]"
        )

        # Run batch generation
        responses = runner.batch_generate(
            prompt_texts,
            rate_limit_delay=0.5 if model_config.provider == "openai" else 1.0,
            show_progress=True,
        )

        # Save raw responses immediately (before any parsing that could crash)
        output_dir = Path(self.config.output.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        raw_responses_file = output_dir / f"raw_responses_{timestamp}.jsonl"
        with open(raw_responses_file, "w") as f:
            for i, (prompt_data, response) in enumerate(zip(all_prompts, responses)):
                metadata = prompt_data.get("metadata", {})
                raw_entry = {
                    "equation_id": prompt_data.get("equation_id", f"eq_{i}"),
                    "prompt": prompt_data.get("prompt", ""),
                    "ground_truth": prompt_data.get("ground_truth"),
                    "metadata": metadata,
                    "raw_response": response,
                    "api_error": response == "",
                }
                f.write(json.dumps(raw_entry) + "\n")
        console.print(
            f"[cyan]> Saved raw responses to {raw_responses_file}[/cyan]"
        )

        # Parse responses and write predictions incrementally
        predictions_file = output_dir / f"predictions_{timestamp}.jsonl"
        predictions: list[dict[str, Any]] = []
        parse_failures = 0

        with open(predictions_file, "w") as f:
            for i, (prompt_data, response) in enumerate(
                zip(all_prompts, responses)
            ):
                try:
                    parsed = parse_llm_output(response)
                except Exception as e:
                    logger.warning(
                        f"Failed to parse response {i}: {e}"
                    )
                    parsed = {
                        "solution_str": None,
                        "solution_sympy": None,
                        "has_solution": None,
                        "solution_type": None,
                        "reasoning": None,
                        "confidence": 0.0,
                    }
                    parse_failures += 1

                metadata = prompt_data.get("metadata", {})
                prediction = {
                    "equation_id": prompt_data.get("equation_id", f"eq_{i}"),
                    "prompt": prompt_data.get("prompt", ""),
                    "ground_truth": prompt_data.get("ground_truth"),
                    "ground_truth_has_solution": metadata.get("has_solution"),
                    "ground_truth_solution_type": metadata.get("solution_type"),
                    "ground_truth_domain": metadata.get("domain"),
                    "raw_response": response,
                    "api_error": response == "",
                    "solution_str": parsed.get("solution_str"),
                    "solution_sympy": str(parsed.get("solution_sympy"))
                    if parsed.get("solution_sympy") is not None
                    else None,
                    "has_solution": parsed.get("has_solution"),
                    "solution_type": parsed.get("solution_type"),
                    "reasoning": parsed.get("reasoning"),
                    "confidence": parsed.get("confidence", 0.0),
                }
                predictions.append(prediction)
                f.write(json.dumps(prediction) + "\n")

        if parse_failures:
            console.print(
                f"\n[yellow]WARNING[/yellow] {parse_failures}/{len(responses)} "
                f"responses failed to parse"
            )
        console.print(f"\n[green]OK[/green] Generated {len(predictions)} predictions")
        console.print(f"[cyan]> Saved predictions to {predictions_file}[/cyan]")

        # Save cost tracking
        cost_summary_file = output_dir / f"cost_summary_{timestamp}.json"
        cost_details_file = output_dir / f"cost_details_{timestamp}.jsonl"

        cost_tracker.save_summary(cost_summary_file)
        cost_tracker.save_detailed_log(cost_details_file)
        cost_tracker.print_summary()

        return predictions

    def _evaluate(self, predictions: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Evaluate predictions against ground truth.

        Args:
            predictions: List of prediction dictionaries.

        Returns:
            Dictionary with evaluation metrics.
        """
        eval_config = self.config.evaluation
        mode = eval_config.mode if eval_config else "both"

        console.print(f"[cyan]> Evaluating ({mode} mode)...[/cyan]")

        if not predictions:
            console.print("[yellow]⚠️  No predictions to evaluate[/yellow]")
            return {"total": 0, "error": "No predictions"}

        # Initialize evaluator
        evaluator = SolutionEvaluator(
            symbolic_tolerance=eval_config.symbolic_tolerance if eval_config else 1e-10,
            numeric_tolerance=eval_config.numeric_tolerance if eval_config else 1e-6,
            n_test_points=eval_config.num_test_points if eval_config else 100,
        )

        # Track edge case metrics
        has_solution_correct = 0
        has_solution_total = 0
        solution_type_correct = 0
        solution_type_total = 0
        evaluated_count = 0
        api_error_count = 0
        errors: list[str] = []

        # None-type detection: TP/FP/FN for precision/recall/F1
        none_tp = 0
        none_fp = 0
        none_fn = 0

        # Read per-type tolerances from config
        type_tolerances = {}
        if eval_config and hasattr(eval_config, "type_tolerances"):
            type_tolerances = eval_config.type_tolerances

        for i, pred in enumerate(predictions):
            # Skip API errors (empty responses from failed API calls)
            if pred.get("api_error"):
                api_error_count += 1
                errors.append(f"Equation {pred.get('equation_id', i)}: API error (empty response)")
                continue

            # Evaluate edge case metrics
            gt_has_solution = pred.get("ground_truth_has_solution")
            pred_has_solution = pred.get("has_solution")
            if gt_has_solution is not None and pred_has_solution is not None:
                has_solution_total += 1
                if gt_has_solution == pred_has_solution:
                    has_solution_correct += 1

            gt_solution_type = pred.get("ground_truth_solution_type")
            pred_solution_type = pred.get("solution_type")
            if gt_solution_type and pred_solution_type:
                solution_type_total += 1
                if gt_solution_type == pred_solution_type:
                    solution_type_correct += 1

            # None-type detection tracking
            if gt_solution_type == "none":
                if pred_has_solution is False:
                    none_tp += 1
                else:
                    none_fn += 1
            elif gt_solution_type is not None and pred_has_solution is False:
                none_fp += 1

            # Extract domain from metadata
            domain = tuple(pred.get("ground_truth_domain") or [0, 1])

            # Branch: "none" type - evaluate by has_solution detection
            if gt_solution_type == "none":
                evaluator.evaluate_none_type(pred_has_solution)
                evaluated_count += 1
                continue

            # Evaluate solution accuracy
            ground_truth_str = pred.get("ground_truth")
            solution_str = pred.get("solution_str")

            if not ground_truth_str or not solution_str:
                continue

            try:
                gt_expr = parse_latex_to_sympy(ground_truth_str)
                pred_expr = parse_latex_to_sympy(solution_str)

                # Branch: "family" type - use family comparison
                if gt_solution_type == "family":
                    evaluator.evaluate_family(pred_expr, gt_expr, domain=domain)
                    evaluated_count += 1
                    continue

                # Standard evaluation with per-type tolerance override
                tol_override = type_tolerances.get(gt_solution_type) if gt_solution_type else None
                evaluator.evaluate(
                    pred_expr, gt_expr, domain=domain,
                    solution_type=gt_solution_type,
                    numeric_tolerance_override=tol_override,
                    pred_str=solution_str,
                    gt_str=ground_truth_str,
                )
                evaluated_count += 1

            except Exception as e:
                errors.append(f"Equation {pred.get('equation_id', i)}: {str(e)}")
                logger.debug(f"Failed to evaluate prediction {i}: {e}")

        # Get summary
        summary = evaluator.summary()

        # Add edge case metrics
        metrics = {
            **summary,
            "evaluated_count": evaluated_count,
            "total_predictions": len(predictions),
            "api_errors": api_error_count,
            "parse_errors": len(errors) - api_error_count,
        }

        if has_solution_total > 0:
            metrics["has_solution_accuracy"] = has_solution_correct / has_solution_total
            metrics["has_solution_total"] = has_solution_total

        if solution_type_total > 0:
            metrics["solution_type_accuracy"] = (
                solution_type_correct / solution_type_total
            )
            metrics["solution_type_total"] = solution_type_total

        # None-type detection precision / recall / F1
        if none_tp + none_fp + none_fn > 0:
            none_prec = none_tp / (none_tp + none_fp) if (none_tp + none_fp) > 0 else 0.0
            none_rec = none_tp / (none_tp + none_fn) if (none_tp + none_fn) > 0 else 0.0
            none_f1 = (
                2 * none_prec * none_rec / (none_prec + none_rec)
                if (none_prec + none_rec) > 0 else 0.0
            )
            metrics["none_detection"] = {
                "precision": none_prec,
                "recall": none_rec,
                "f1": none_f1,
                "tp": none_tp,
                "fp": none_fp,
                "fn": none_fn,
            }

        # Display results
        console.print(f"\n[bold]Evaluation Results:[/bold]")
        console.print(f"  Total predictions: {len(predictions)}")
        console.print(f"  Evaluated: {evaluated_count}")
        if summary.get("total", 0) > 0:
            console.print(f"  Accuracy: {summary.get('accuracy', 0):.2%}")
            console.print(
                f"  Symbolic accuracy: {summary.get('symbolic_accuracy', 0):.2%}"
            )
            console.print(
                f"  Numeric accuracy: {summary.get('numeric_accuracy', 0):.2%}"
            )
        if "mean_operator_f1" in summary:
            console.print(
                f"  Operator F1: {summary['mean_operator_f1']:.2%}"
            )
            console.print(
                f"  Operator Precision: {summary['mean_operator_precision']:.2%}"
            )
            console.print(
                f"  Operator Recall: {summary['mean_operator_recall']:.2%}"
            )
        if "mean_rel_l2" in summary:
            console.print(
                f"  Relative L2: {summary['mean_rel_l2']:.6f}"
            )
        if "mean_bleu" in summary:
            console.print(f"  BLEU: {summary['mean_bleu']:.4f}")
        if "none_detection" in metrics:
            nd = metrics["none_detection"]
            console.print(
                f"  None detection P/R/F1: "
                f"{nd['precision']:.2%} / {nd['recall']:.2%} / {nd['f1']:.2%}"
            )
        if has_solution_total > 0:
            console.print(
                f"  Has solution accuracy: {metrics['has_solution_accuracy']:.2%}"
            )
        if solution_type_total > 0:
            console.print(
                f"  Solution type accuracy: {metrics['solution_type_accuracy']:.2%}"
            )
        # Display per-type breakdown
        per_type = summary.get("per_type", {})
        if per_type:
            console.print(f"\n  [bold]Per-type breakdown:[/bold]")
            for stype, counts in sorted(per_type.items()):
                console.print(
                    f"    {stype}: {counts['correct']}/{counts['total']} "
                    f"({counts['accuracy']:.0%})"
                )

        if api_error_count > 0:
            console.print(f"  [red]API errors: {api_error_count}[/red]")
        if len(errors) - api_error_count > 0:
            console.print(f"  [yellow]Parse errors: {len(errors) - api_error_count}[/yellow]")

        # Save metrics to file
        output_dir = Path(self.config.output.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = output_dir / f"metrics_{timestamp}.json"

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        console.print(f"\n[cyan]> Saved metrics to {metrics_file}[/cyan]")

        return metrics


def load_adaptive_config(config_path: Path) -> AdaptivePipelineConfig:
    """Load and validate adaptive configuration."""
    import yaml

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return AdaptivePipelineConfig(**config_dict)
