#!/usr/bin/env python3
"""Checks for Step 5 useful deterministic code-generation templates.

This benchmark still avoids Agno and Ollama. It verifies that BMAD no longer
always produces the same generic app: keyword-based task routing now generates
small but functional CLI programs with local unittest validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.bmad_coding_team import run_bmad_task
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
        print(f"\nBMAD useful generation checks: {self.passed}/{self.total} passed")
        return 0 if self.passed == self.total else 1


def _workspace_text(path: str) -> str:
    return (WORKSPACE_DIR / path).read_text(encoding="utf-8")


def _run_case(task: str) -> tuple[str, str, object]:
    clear_staging()
    clear_workspace()
    result = run_bmad_task(task, run_docker=False, clean_workspace=True)
    return _workspace_text("app.py"), _workspace_text("README.md"), result


def main() -> int:
    checks = CheckRunner()

    app_text, readme_text, result = _run_case(
        "Create a temperature converter CLI in Python with Celsius to Fahrenheit, README, and basic tests"
    )
    checks.check(result.approved and result.published, "temperature converter workflow is approved and published")
    checks.check("def celsius_to_fahrenheit" in app_text, "temperature app has Celsius to Fahrenheit function")
    checks.check("def fahrenheit_to_celsius" in app_text, "temperature app has Fahrenheit to Celsius function")
    checks.check("python app.py --celsius 20" in readme_text, "temperature README has useful CLI example")
    checks.check(any("Generated unit tests passed locally" in item for item in result.qa_report.checks), "temperature generated tests pass locally")

    app_text, readme_text, result = _run_case(
        "Create a todo list CLI in Python with add, list, and clear features, plus README and tests"
    )
    checks.check(result.approved and result.published, "todo workflow is approved and published")
    checks.check("def add_item" in app_text and "def clear_items" in app_text, "todo app has add and clear functions")
    checks.check("subparsers.add_parser(\"add\"" in app_text, "todo app has add subcommand")
    checks.check("python app.py add" in readme_text, "todo README has add example")
    checks.check(any("Generated unit tests passed locally" in item for item in result.qa_report.checks), "todo generated tests pass locally")

    app_text, readme_text, result = _run_case(
        "Create a password strength checker CLI with README and unit tests"
    )
    checks.check(result.approved and result.published, "password checker workflow is approved and published")
    checks.check("def analyze_password" in app_text, "password app has analyze_password function")
    checks.check("length_at_least_8" in app_text and "has_symbol" in app_text, "password app checks length and symbols")
    checks.check("StrongerPass123" in readme_text, "password README has realistic CLI example")
    checks.check(any("Generated unit tests passed locally" in item for item in result.qa_report.checks), "password generated tests pass locally")

    app_text, readme_text, result = _run_case(
        "Create a simple expense tracker CLI with add, list, total, README, and tests"
    )
    checks.check(result.approved and result.published, "expense tracker workflow is approved and published")
    checks.check("def add_expense" in app_text and "def total_expenses" in app_text, "expense app has add and total functions")
    checks.check("expenses.json" in app_text, "expense app has JSON persistence")
    checks.check("python app.py add 'Coffee' 2.50" in readme_text, "expense README has add example")
    checks.check(any("Generated unit tests passed locally" in item for item in result.qa_report.checks), "expense generated tests pass locally")

    module_source = (ROOT / "agents" / "bmad_coding_team.py").read_text(encoding="utf-8")
    checks.check("_detect_project_kind" in module_source, "BMAD has deterministic task template routing")
    checks.check("temperature_converter" in module_source, "BMAD includes temperature converter template")
    checks.check("todo_cli" in module_source, "BMAD includes todo CLI template")
    checks.check("password_strength" in module_source, "BMAD includes password strength template")
    checks.check("expense_tracker" in module_source, "BMAD includes expense tracker template")

    clear_staging()
    clear_workspace()
    return checks.finish()


if __name__ == "__main__":
    raise SystemExit(main())
