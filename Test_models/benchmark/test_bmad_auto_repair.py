#!/usr/bin/env python3
"""Checks for Step 7 BMAD automatic QA repair loop.

This benchmark does not call a real LLM. It injects a fake broken code provider
so QA fails, then verifies that the BMAD repair loop rewrites staging files,
runs QA again, and publishes only after the repaired output passes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.bmad_coding_team import BMADCodingTeam
from agents.bmad_llm_codegen import CodeGenerationResult, GeneratedCodeFile
from tools.file_tools import WORKSPACE_DIR, clear_staging, clear_workspace


class CheckRunner:
    def __init__(self) -> None:
        self.total = 0
        self.passed = 0

    def check(self, condition: bool, message: str) -> None:
        self.total += 1
        if condition:
            self.passed += 1
            print(f"[PASS] {message}")
        else:
            print(f"[FAIL] {message}")

    def finish(self) -> int:
        print(f"\nBMAD automatic repair checks: {self.passed}/{self.total} passed")
        return 0 if self.passed == self.total else 1


class FakeBrokenLLMGenerator:
    """Return unsafe quality code through the normal safe path.

    Paths are safe, but app.py has a syntax error. This simulates a real LLM
    output that passes path validation but fails QA.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, spec: Any) -> CodeGenerationResult:
        self.calls.append(spec.original_request)
        return CodeGenerationResult(
            success=True,
            provider="fake_broken_llm",
            summary="Generated intentionally broken files for repair testing.",
            files=[
                GeneratedCodeFile("app.py", "def broken(:\n    pass\n", "broken implementation"),
                GeneratedCodeFile("README.md", "# Broken output\n\n## Run\n\n```bash\npython app.py\n```\n", "broken docs"),
                GeneratedCodeFile("test_app.py", "import unittest\n\nclass TestBroken(unittest.TestCase):\n    def test_true(self):\n        self.assertTrue(True)\n", "basic tests"),
            ],
        )


def _workspace_text(path: str) -> str:
    return (WORKSPACE_DIR / path).read_text(encoding="utf-8")


def main() -> int:
    checks = CheckRunner()
    clear_staging()
    clear_workspace()

    broken_generator = FakeBrokenLLMGenerator()
    repair_team = BMADCodingTeam(code_generator=broken_generator, max_revision_cycles=1)
    repaired = repair_team.run(
        "Create a temperature converter CLI with README and tests",
        run_docker=False,
        clean_workspace=True,
    )

    checks.check(broken_generator.calls == ["Create a temperature converter CLI with README and tests"], "fake LLM is called once for initial generation")
    checks.check(repaired.revision_attempts == 1, "workflow reports one revision attempt")
    checks.check(repaired.approved and repaired.published, "repaired workflow is approved and published")
    checks.check(repaired.qa_report.passed, "final QA report passes after repair")
    checks.check(repaired.qa_report.attempt == 1, "final QA report records retry attempt number")
    checks.check("app.py" in repaired.workspace_files and "test_app.py" in repaired.workspace_files, "repaired artifacts are published")
    checks.check("def celsius_to_fahrenheit" in _workspace_text("app.py"), "repair replaced broken code with temperature converter implementation")
    checks.check(any("repair_with_deterministic_template" in step.actions for step in repaired.steps), "Dev repair step records deterministic repair")
    checks.check(any(step.agent == "QA" and step.status == "failed" for step in repaired.steps), "workflow keeps the initial failed QA step")
    checks.check(any(step.agent == "QA" and step.status == "passed" for step in repaired.steps), "workflow keeps the final passed QA step")
    checks.check("Revision attempts: 1" in repaired.final_message, "final message reports revision attempts")
    checks.check("revision_attempts=1" in (repaired.memory_candidate or ""), "memory candidate records revision attempts")

    clear_staging()
    clear_workspace()
    no_repair_team = BMADCodingTeam(code_generator=FakeBrokenLLMGenerator(), max_revision_cycles=0)
    rejected = no_repair_team.run(
        "Create a temperature converter CLI with README and tests",
        run_docker=False,
        clean_workspace=True,
    )

    checks.check(rejected.revision_attempts == 0, "max_revision_cycles=0 disables the repair loop")
    checks.check(not rejected.approved and not rejected.published, "broken workflow is rejected when repair is disabled")
    checks.check(any("syntax error" in failure.lower() for failure in rejected.qa_report.failures), "rejected workflow exposes syntax failure")

    clear_staging()
    clear_workspace()
    return checks.finish()


if __name__ == "__main__":
    raise SystemExit(main())
