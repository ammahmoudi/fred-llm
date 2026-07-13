"""
Program-of-thought runner: the model writes a SymPy script, we execute it.

Wraps a base model runner. Each response is expected to contain a single
```python code block; the block runs in an isolated subprocess with a timeout.
When the script's stdout contains the standard ``SOLUTION:`` format, that
stdout replaces the raw response, so the unchanged postprocess/evaluate
pipeline applies. On failure one feedback repair round re-prompts with the
error; the final fallback is the raw model text (the model may have answered
in prose despite instructions).
"""

import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.llm.model_runner import BaseModelRunner
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_MAX_STDOUT_CHARS = 20_000


def extract_code(text: str) -> str | None:
    """Return the last fenced python block, or None if there is none."""
    blocks = _CODE_BLOCK_RE.findall(text or "")
    return blocks[-1].strip() if blocks else None


class CodeExecModelRunner(BaseModelRunner):
    """Executes model-written SymPy scripts and returns their printed output."""

    def __init__(
        self,
        base_runner: BaseModelRunner,
        exec_timeout: int = 60,
        max_repair_rounds: int = 1,
        exec_workers: int = 8,
    ):
        super().__init__()
        self.base = base_runner
        self.exec_timeout = exec_timeout
        self.max_repair_rounds = max_repair_rounds
        self.exec_workers = exec_workers
        self.traces: list[dict[str, Any]] = []

    def set_cost_tracker(self, tracker: Any) -> None:
        self.base.set_cost_tracker(tracker)

    def _run_script(self, code: str) -> tuple[bool, str, str]:
        """Execute code in an isolated subprocess. Returns (ok, stdout, error)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "solve.py"
            script.write_text(code)
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(script)],
                    capture_output=True,
                    text=True,
                    timeout=self.exec_timeout,
                    cwd=tmpdir,
                )
            except subprocess.TimeoutExpired:
                return False, "", f"timed out after {self.exec_timeout}s"
        stdout = proc.stdout[-_MAX_STDOUT_CHARS:]
        if proc.returncode != 0:
            return False, stdout, proc.stderr[-2000:]
        if "SOLUTION:" not in stdout:
            return False, stdout, "script printed no SOLUTION: line"
        return True, stdout, ""

    def _resolve(self, prompt: str, response: str) -> tuple[str, dict[str, Any]]:
        """Execute (and optionally repair) one response; return (text, trace)."""
        trace: dict[str, Any] = {"rounds": [], "outcome": "fallback_raw"}
        for round_i in range(self.max_repair_rounds + 1):
            code = extract_code(response)
            if code is None:
                trace["outcome"] = "no_code"
                break
            ok, stdout, err = self._run_script(code)
            trace["rounds"].append(
                {"code": code, "ok": ok, "stdout_tail": stdout[-500:], "error": err}
            )
            if ok:
                trace["outcome"] = "exec_ok"
                return stdout, trace
            if round_i < self.max_repair_rounds:
                repair_prompt = (
                    f"{prompt}\n\nYour previous script was:\n```python\n{code}\n```\n"
                    f"It failed:\n{err or stdout[-1000:]}\n\n"
                    "Fix it. Reply with ONLY a single ```python code block."
                )
                try:
                    response = self.base.generate(repair_prompt)
                except Exception as e:
                    logger.warning(f"Repair call failed: {e}")
                    break
        return response, trace

    def generate(self, prompt: str, **kwargs: Any) -> str:
        text, trace = self._resolve(prompt, self.base.generate(prompt, **kwargs))
        self.traces.append(trace)
        return text

    def batch_generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        responses = self.base.batch_generate(prompts, **kwargs)
        results: list[str] = [""] * len(prompts)
        traces: list[dict[str, Any]] = [{} for _ in prompts]

        # Script execution is subprocess-bound and repairs are independent API
        # calls, so per-item resolution parallelizes safely.
        def work(i: int) -> None:
            results[i], traces[i] = self._resolve(prompts[i], responses[i])
            traces[i]["index"] = i
            traces[i]["raw_response"] = responses[i]

        with ThreadPoolExecutor(max_workers=self.exec_workers) as ex:
            list(ex.map(work, range(len(prompts))))

        outcomes = [t.get("outcome") for t in traces]
        logger.info(
            "code-exec outcomes: "
            + ", ".join(f"{o}={outcomes.count(o)}" for o in sorted(set(outcomes)))
        )
        self.traces.extend(traces)
        return results
