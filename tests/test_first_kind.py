"""Tests for first-kind (regularized) equation handling.

The ill_posed augmentation stores first-kind equations with lambda_val=0 as a
sentinel. These tests cover: prompt rendering (first-kind form, not the
trivial second-kind template), type-classification scoring for regularized
items, and the agentic verifier skipping first-kind equations.
"""

from src.evaluation.core import SolutionEvaluator
from src.llm.agentic_runner import AgenticModelRunner
from src.llm.model_runner import BaseModelRunner
from src.prompts.base import EquationData
from src.prompts.factory import create_prompt_style
from src.prompts.templates import format_equation_line


def make_equation(lambda_val):
    return EquationData(
        u="",
        f="exp(x)",
        kernel="exp(x*t)",
        lambda_val=lambda_val,
        a=0.0,
        b=1.0,
        equation_id="eq_fk",
        has_solution=True,
        solution_type="regularized",
    )


def test_first_kind_renders_without_ux_term():
    line = format_equation_line(make_equation(0.0), "infix")
    assert line.startswith("∫_")
    assert "u(x)" not in line
    assert "0" not in line.split("∫")[0]  # no lambda prefix


def test_second_kind_renders_with_ux_term():
    line = format_equation_line(make_equation(0.5), "infix")
    assert line.startswith("u(x) - 0.5 *")
    assert "∫_" in line


def test_all_styles_render_first_kind():
    for style_name in ("basic", "chain-of-thought", "few-shot", "tool-assisted"):
        style = create_prompt_style(style_name)
        prompt = style.get_user_prompt(make_equation(0.0), "infix")
        assert "u(x) - 0" not in prompt, style_name
        assert "∫_" in prompt, style_name


def test_regularized_scored_by_type_classification():
    ev = SolutionEvaluator()
    assert ev.evaluate_regularized_type("regularized")["correct"] is True
    assert ev.evaluate_regularized_type("exact_symbolic")["correct"] is False
    assert ev.evaluate_regularized_type(None)["correct"] is False


def test_agentic_verifier_skips_first_kind():
    class Dummy(BaseModelRunner):
        def generate(self, p, **k):
            return ""

        def batch_generate(self, p, **k):
            return []

    runner = AgenticModelRunner(Dummy(), use_math_verify=False)
    eq = {"id": "eq_fk", "kernel": "exp(x*t)", "f": "exp(x)", "lambda_val": 0.0}
    assert runner._parse_equation(eq) is None
    eq["lambda_val"] = 0.5
    assert runner._parse_equation(eq) is not None
