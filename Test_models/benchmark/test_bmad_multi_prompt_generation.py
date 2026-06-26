#!/usr/bin/env python3
"""Checks for Step 10 BMAD multi-prompt generation.

The full workflow is executed on representative prompts, while all Step 10
prompt types are checked for deterministic routing, specialized templates and
runtime behavior checks. This keeps the benchmark fast but still protects the
main regression: BMAD must not approve a generic describe_project() placeholder
for a specific coding request.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.bmad_coding_team import _detect_project_kind, _kind_behavior_checks, run_bmad_task
from agents.bmad_templates import build_app_template
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
        print(f"\nBMAD multi-prompt generation checks: {self.passed}/{self.total} passed")
        return 0 if self.passed == self.total else 1


def _workspace_text(path: str) -> str:
    return (WORKSPACE_DIR / path).read_text(encoding="utf-8")


class FakeSpec:
    def __init__(self, request: str) -> None:
        self.original_request = request


def main() -> int:
    checks = CheckRunner()

    all_cases = [
        ("Create a Python program to solve quadratic equations with README and tests", "quadratic_solver", "solve_quadratic"),
        ("Create a Python program to manage a todo list from the command line with README and tests", "todo_cli", "add_item"),
        ("Create a Python program that stores and retrieves user memories with README and tests", "memory_store", "store_memory"),
        ("Create a Python program to read a CSV file and calculate statistics with README and tests", "csv_statistics", "calculate_statistics"),
        ("Create a Python program that simulates a simple multi-agent system with README and tests", "multi_agent_simulation", "run_simulation"),
        ("Create a Python program that validates email addresses and phone numbers with README and tests", "email_phone_validator", "is_valid_email"),
        ("Create a Python program that organizes files in a folder by extension with README and tests", "file_organizer", "organize_by_extension"),
        ("Create a Python program that encrypts and decrypts messages using Caesar cipher with README and tests", "caesar_cipher", "caesar_encrypt"),
        ("Create a Python program that creates a simple chatbot with memory with README and tests", "chatbot_with_memory", "run_scripted_chat"),
        ("Create a Python program that automatically runs manual and unit tests for generated code with README and tests", "test_runner", "run_unit_tests"),
    ]

    for task, expected_kind, expected_function in all_cases:
        kind = _detect_project_kind(task)
        checks.check(kind == expected_kind, f"{expected_kind}: request is routed to the correct template")
        template = build_app_template(kind, FakeSpec(task))
        if kind in {"quadratic_solver", "todo_cli"}:
            # These two templates live in the historical BMAD orchestrator file.
            checks.check(template is None, f"{expected_kind}: legacy built-in template remains in orchestrator")
        else:
            checks.check(template is not None and expected_function in template, f"{expected_kind}: specialized app template exists")
        checks.check(bool(_kind_behavior_checks(kind)), f"{expected_kind}: runtime behavior checks are defined")

    workflow_cases = [
        ("Create a Python program that stores and retrieves user memories with README and tests", "store_memory", "memory_"),
        ("Create a Python program to read a CSV file and calculate statistics with README and tests", "calculate_statistics", "csv_"),
        ("Create a Python program that validates email addresses and phone numbers with README and tests", "is_valid_email", "email"),
        ("Create a Python program that organizes files in a folder by extension with README and tests", "organize_by_extension", "file_organizer"),
        ("Create a Python program that encrypts and decrypts messages using Caesar cipher with README and tests", "caesar_encrypt", "caesar_"),
    ]

    for task, expected_function, expected_behavior_prefix in workflow_cases:
        clear_staging()
        clear_workspace()
        result = run_bmad_task(task, clean_workspace=True, use_llm=False, max_revision_cycles=1)
        label = expected_function
        checks.check(result.approved and result.published, f"{label}: workflow approved and published")
        checks.check(set(result.workspace_files).issubset({"app.py", "README.md", "test_app.py", "requirements.txt"}), f"{label}: workspace contains only useful files")
        app_code = _workspace_text("app.py")
        checks.check(expected_function in app_code, f"{label}: app implements expected function")
        checks.check("def describe_project" not in app_code, f"{label}: app is not a generic placeholder")
        checks.check(any(expected_behavior_prefix in item for item in result.qa_report.checks), f"{label}: runtime behavior check executed")
        checks.check("Manual test commands:" in result.final_message, f"{label}: final message includes manual test commands")

    clear_staging()
    clear_workspace()
    unknown = run_bmad_task(
        "Create a Python program that downloads weather data and plots it with README and tests",
        clean_workspace=True,
        use_llm=False,
        max_revision_cycles=0,
    )
    checks.check(not unknown.approved, "unknown specific request is not approved as generic placeholder")
    checks.check(any("generic placeholder" in failure for failure in unknown.qa_report.failures), "unknown rejection explains generic placeholder")

    clear_staging()
    clear_workspace()
    return checks.finish()


if __name__ == "__main__":
    raise SystemExit(main())
