#!/usr/bin/env python3
"""Minimal checks for the deterministic BMAD coding team.

This test intentionally does not use Agno, Ollama or Titan. It only validates the
new orchestration layer and the staging/workspace tools.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.bmad_coding_team import BMADCodingTeam, run_bmad_task
from tools.file_tools import STAGING_DIR, WORKSPACE_DIR, clear_staging, clear_workspace, list_staging_files


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
        print(f"\nBMAD coding team checks: {self.passed}/{self.total} passed")
        return 0 if self.passed == self.total else 1


def _workspace_text(path: str) -> str:
    return (WORKSPACE_DIR / path).read_text(encoding="utf-8")


def main() -> int:
    checks = CheckRunner()
    clear_staging()
    clear_workspace()

    result = run_bmad_task(
        "Create a simple Python CLI app with a README and a basic test",
        run_docker=False,
        clean_workspace=True,
    )

    checks.check(result.approved, "workflow is approved")
    checks.check(result.published, "workflow publishes validated output")
    checks.check("app.py" in result.workspace_files, "app.py is published to workspace")
    checks.check("README.md" in result.workspace_files, "README.md is published to workspace")
    checks.check("test_app.py" in result.workspace_files, "test_app.py is published when task asks for tests")
    checks.check("Dockerfile" not in result.workspace_files, "Dockerfile is not published as final user artifact")
    checks.check("docker-compose.yml" not in result.workspace_files, "docker-compose is not published as final user artifact")
    checks.check(result.staging_files_after_run == [], "staging is empty after publish")
    checks.check("The staging folder is empty." in list_staging_files(), "staging list confirms empty folder")

    expected_order = [
        "Project Manager",
        "Business Agent",
        "Dev",
        "Designer",
        "DevOps",
        "QA",
        "Evaluator",
        "Project Manager",
    ]
    actual_order = [step.agent for step in result.steps]
    checks.check(actual_order == expected_order, "BMAD agents run in the expected order")
    checks.check(result.qa_report.passed, "QA report passes")
    checks.check(any("syntax validation passed" in check for check in result.qa_report.checks), "QA performs syntax validation")
    checks.check(any("Docker execution skipped" in check for check in result.qa_report.checks), "Docker is skipped by default")
    checks.check("BMAD coding workflow result: APPROVED" in result.final_message, "final message summarizes approval")

    app_text = _workspace_text("app.py")
    readme_text = _workspace_text("README.md")
    checks.check("def describe_project" in app_text, "generated app contains describe_project function")
    checks.check("## Run" in readme_text, "README contains run instructions")
    checks.check("Create a simple Python CLI app" in readme_text, "README preserves original request")

    team = BMADCodingTeam()
    try:
        team.run("   ")
        empty_task_failed = False
    except ValueError:
        empty_task_failed = True
    checks.check(empty_task_failed, "empty tasks are rejected")

    module_source = (ROOT / "agents" / "bmad_coding_team.py").read_text(encoding="utf-8").lower()
    forbidden_imports = ["import agno", "from agno", "import ollama", "from models.titan_model", "titanagentmemory"]
    checks.check(not any(item in module_source for item in forbidden_imports), "BMAD step 2 does not import Agno, Ollama or Titan")

    return checks.finish()


if __name__ == "__main__":
    raise SystemExit(main())
