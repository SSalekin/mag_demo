#!/usr/bin/env python3
"""Checks that BMAD repair receives useful QA/Docker diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.bmad_coding_team import BMADCodingTeam, QAReport, _qa_repair_messages
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
        print(f"\nBMAD repair diagnostics checks: {self.passed}/{self.total} passed")
        return 0 if self.passed == self.total else 1


class FakeRepairAwareGenerator:
    def __init__(self) -> None:
        self.repair_messages: list[str] = []
        self.repair_attempts: list[int] = []

    def generate(self, spec: Any) -> CodeGenerationResult:
        return CodeGenerationResult(
            success=True,
            provider="fake_broken_generator",
            summary="Generated syntactically broken app for diagnostics test.",
            files=[
                GeneratedCodeFile("app.py", "def broken(:\n    pass\n", "broken app"),
                GeneratedCodeFile("README.md", "# Broken\n", "docs"),
                GeneratedCodeFile("test_app.py", "import unittest\n\nclass TestX(unittest.TestCase):\n    def test_x(self):\n        self.assertTrue(True)\n", "tests"),
            ],
        )

    def repair(self, *, spec: Any, qa_failures: Sequence[str], current_files: Sequence[GeneratedCodeFile], attempt: int = 1) -> CodeGenerationResult:
        self.repair_messages = list(qa_failures)
        self.repair_attempts.append(attempt)
        return CodeGenerationResult(
            success=True,
            provider="fake_repair_generator",
            summary="Returned fixed temperature converter.",
            files=[
                GeneratedCodeFile(
                    "app.py",
                    """from __future__ import annotations\n\nimport argparse\n\ndef celsius_to_fahrenheit(celsius: float) -> float:\n    return (celsius * 9 / 5) + 32\n\ndef fahrenheit_to_celsius(fahrenheit: float) -> float:\n    return (fahrenheit - 32) * 5 / 9\n\ndef main(argv: list[str] | None = None) -> int:\n    parser = argparse.ArgumentParser()\n    group = parser.add_mutually_exclusive_group(required=True)\n    group.add_argument('--celsius', type=float)\n    group.add_argument('--fahrenheit', type=float)\n    args = parser.parse_args(argv)\n    if args.celsius is not None:\n        print(f'{celsius_to_fahrenheit(args.celsius):.2f} F')\n    else:\n        print(f'{fahrenheit_to_celsius(args.fahrenheit):.2f} C')\n    return 0\n\nif __name__ == '__main__':\n    raise SystemExit(main())\n""",
                    "fixed app",
                ),
                GeneratedCodeFile("README.md", "# Temperature converter\n\nRun with `python app.py`.\n", "docs"),
                GeneratedCodeFile(
                    "test_app.py",
                    """import unittest\nfrom app import celsius_to_fahrenheit, fahrenheit_to_celsius\n\nclass TestTemperature(unittest.TestCase):\n    def test_celsius(self):\n        self.assertEqual(celsius_to_fahrenheit(20), 68)\n    def test_fahrenheit(self):\n        self.assertEqual(fahrenheit_to_celsius(68), 20)\n\nif __name__ == '__main__':\n    unittest.main()\n""",
                    "tests",
                ),
            ],
        )


def main() -> int:
    checks = CheckRunner()

    report = QAReport(
        passed=False,
        checks=["Found staging/app.py.", "Found staging/README.md."],
        failures=["app.py syntax error: invalid syntax"],
        docker_output="docker build ok\ndocker check failed with traceback",
        attempt=2,
    )
    messages = _qa_repair_messages(report)
    joined = "\n".join(messages)
    checks.check("QA attempt 2 failed" in joined, "repair context includes QA attempt number")
    checks.check("app.py syntax error" in joined, "repair context includes blocking failure")
    checks.check("Found staging/app.py" in joined, "repair context includes passed checks")
    checks.check("Docker diagnostic output" in joined and "traceback" in joined, "repair context includes Docker diagnostics")

    clear_staging()
    clear_workspace()
    generator = FakeRepairAwareGenerator()
    team = BMADCodingTeam(code_generator=generator, max_revision_cycles=1)
    result = team.run("Create a temperature converter CLI with README and tests", clean_workspace=True)

    repair_joined = "\n".join(generator.repair_messages)
    checks.check(result.approved and result.published, "workflow is approved after fake repair")
    checks.check(generator.repair_attempts == [1], "repair generator receives attempt number")
    checks.check("QA attempt 0 failed" in repair_joined, "LLM repair receives QA attempt diagnostics")
    checks.check("syntax error" in repair_joined.lower(), "LLM repair receives syntax failure details")
    checks.check("Checks that already passed" in repair_joined, "LLM repair receives prior successful checks")
    checks.check("app.py" in result.workspace_files and (WORKSPACE_DIR / "app.py").exists(), "fixed app is published")

    clear_staging()
    clear_workspace()
    return checks.finish()


if __name__ == "__main__":
    raise SystemExit(main())
