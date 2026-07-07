"""Tests for the agentic multi-method solver."""

import pytest

from src.adaptive_config import AgenticConfig, ModelConfig
from src.llm.agentic_runner import METHOD_DIRECTIVES, AgenticModelRunner
from src.llm.model_runner import BaseModelRunner

# u(x) - 0.5 * int_0^1 x*t*u(t) dt = x  =>  u(x) = 6x/5
EQUATION = {
    "id": "eq_test",
    "kernel": "x*t",
    "f": "x",
    "lambda_val": 0.5,
    "domain": [0, 1],
}

CORRECT = (
    "REASONING: separable kernel, solved the linear system.\n"
    "SOLUTION: u(x) = 6*x/5\n"
    "HAS_SOLUTION: yes\n"
    "SOLUTION_TYPE: exact_symbolic\n"
)
WRONG = (
    "REASONING: quick guess.\n"
    "SOLUTION: u(x) = x\n"
    "HAS_SOLUTION: yes\n"
    "SOLUTION_TYPE: exact_symbolic\n"
)
NONE_CLAIM = (
    "REASONING: lambda is a characteristic value and f is not orthogonal.\n"
    "SOLUTION: none\n"
    "HAS_SOLUTION: no\n"
    "SOLUTION_TYPE: none\n"
)


class FakeRunner(BaseModelRunner):
    """Returns canned responses per method; repair responses after round 0."""

    def __init__(self, responses: dict[str, str], repair: dict[str, str] | None = None):
        super().__init__()
        self.responses = responses
        self.repair = repair or {}
        self.calls: list[str] = []

    def _method_of(self, prompt: str) -> str:
        for name, directive in METHOD_DIRECTIVES.items():
            if prompt.startswith(directive):
                return name
        raise AssertionError("prompt missing method directive")

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        method = self._method_of(prompt)
        if "PREVIOUS ATTEMPT" in prompt:
            return self.repair.get(method, self.responses[method])
        return self.responses[method]

    def batch_generate(self, prompts: list[str], **kwargs) -> list[str]:
        return [self.generate(p) for p in prompts]


def make_runner(fake: FakeRunner, methods: list[str], **kwargs) -> AgenticModelRunner:
    return AgenticModelRunner(
        base_runner=fake, methods=methods, use_math_verify=False, **kwargs
    )


def test_verified_candidate_wins():
    fake = FakeRunner(
        {"degenerate_kernel": WRONG, "adomian": CORRECT, "neumann": WRONG}
    )
    runner = make_runner(fake, ["degenerate_kernel", "adomian", "neumann"])

    result = runner.generate("Solve it.", equation=EQUATION)

    assert "6*x/5" in result
    trace = runner.traces[0]
    assert trace["winner_method"] == "adomian"
    assert trace["selection_reason"] == "verified"
    assert trace["llm_calls"] == 3


def test_majority_vote_without_equation():
    fake = FakeRunner(
        {"degenerate_kernel": NONE_CLAIM, "adomian": WRONG, "neumann": NONE_CLAIM}
    )
    runner = make_runner(fake, ["degenerate_kernel", "adomian", "neumann"])

    result = runner.generate("Solve it.", equation=None)

    assert "HAS_SOLUTION: no" in result
    assert runner.traces[0]["selection_reason"] == "majority_vote"


def test_repair_round_fixes_failed_candidate():
    fake = FakeRunner(
        {"degenerate_kernel": WRONG}, repair={"degenerate_kernel": CORRECT}
    )
    runner = make_runner(fake, ["degenerate_kernel"], max_repair_rounds=1)

    result = runner.generate("Solve it.", equation=EQUATION)

    assert "6*x/5" in result
    assert len(fake.calls) == 2
    assert "PREVIOUS ATTEMPT" in fake.calls[1]
    trace = runner.traces[0]
    assert trace["llm_calls"] == 2
    assert trace["selection_reason"] == "verified"


def test_no_repair_when_verified_first_round():
    fake = FakeRunner({"degenerate_kernel": CORRECT})
    runner = make_runner(fake, ["degenerate_kernel"], max_repair_rounds=3)

    runner.generate("Solve it.", equation=EQUATION)

    assert len(fake.calls) == 1


def test_all_calls_failed_returns_empty():
    fake = FakeRunner({"degenerate_kernel": "", "adomian": ""})
    runner = make_runner(fake, ["degenerate_kernel", "adomian"])

    result = runner.generate("Solve it.", equation=EQUATION)

    assert result == ""
    assert runner.traces[0]["selection_reason"] == "all_calls_failed"


def test_batch_generate_aligned_with_prompts():
    fake = FakeRunner({"degenerate_kernel": CORRECT})
    runner = make_runner(fake, ["degenerate_kernel"])

    results = runner.batch_generate(
        ["p1", "p2"],
        equations=[EQUATION, EQUATION],
        rate_limit_delay=0,
        show_progress=False,
    )

    assert len(results) == 2
    assert len(runner.traces) == 2


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="Unknown agentic methods"):
        AgenticModelRunner(base_runner=FakeRunner({}), methods=["banach"])


def test_model_config_accepts_agentic_section():
    config = ModelConfig(
        provider="openrouter",
        name="openai/gpt-4o-mini",
        agentic={"max_repair_rounds": 2},
    )
    assert config.agentic is not None
    assert config.agentic.max_repair_rounds == 2
    assert len(config.agentic.methods) == 5  # defaults to all
    assert AgenticConfig().verify_tolerance == 1e-6
