"""Tests for the program-of-thought code-exec runner (no API calls)."""

from src.llm.code_exec_runner import CodeExecModelRunner, extract_code
from src.llm.model_runner import BaseModelRunner


class FakeRunner(BaseModelRunner):
    """Replays canned responses; records repair prompts."""

    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)
        self.prompts_seen = []

    def generate(self, prompt, **kwargs):
        self.prompts_seen.append(prompt)
        return self.responses.pop(0)

    def batch_generate(self, prompts, **kwargs):
        return [self.generate(p) for p in prompts]


GOOD = "```python\nprint('SOLUTION: u(x) = x')\nprint('HAS_SOLUTION: yes')\nprint('SOLUTION_TYPE: exact_symbolic')\n```"
BROKEN = "```python\nraise ValueError('boom')\n```"


def test_extract_code_takes_last_block():
    text = "```python\nfirst\n```\ntext\n```python\nsecond\n```"
    assert extract_code(text) == "second"
    assert extract_code("no code here") is None


def test_exec_ok_returns_stdout():
    runner = CodeExecModelRunner(FakeRunner([GOOD]), exec_timeout=15)
    out = runner.batch_generate(["solve it"])
    assert "SOLUTION: u(x) = x" in out[0]
    assert runner.traces[0]["outcome"] == "exec_ok"


def test_repair_round_recovers_from_error():
    base = FakeRunner([BROKEN, GOOD])
    runner = CodeExecModelRunner(base, exec_timeout=15, max_repair_rounds=1)
    out = runner.batch_generate(["solve it"])
    assert "SOLUTION: u(x) = x" in out[0]
    assert len(runner.traces[0]["rounds"]) == 2
    assert "boom" in base.prompts_seen[-1]  # error fed back to the model


def test_no_code_block_falls_back_to_raw():
    runner = CodeExecModelRunner(FakeRunner(["SOLUTION: u(x) = 42"]))
    out = runner.batch_generate(["solve it"])
    assert out[0] == "SOLUTION: u(x) = 42"
    assert runner.traces[0]["outcome"] == "no_code"


def test_timeout_falls_back_to_raw():
    hang = "```python\nimport time\ntime.sleep(30)\n```"
    runner = CodeExecModelRunner(
        FakeRunner([hang]), exec_timeout=2, max_repair_rounds=0
    )
    out = runner.batch_generate(["solve it"])
    assert out[0] == hang
    assert runner.traces[0]["rounds"][0]["error"].startswith("timed out")
