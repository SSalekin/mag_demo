#!/usr/bin/env python3
"""Checks for Step 11 BMAD Docker/sandbox execution.

These tests do not require a real Docker daemon. They monkeypatch the Docker
runner used by BMAD QA so the workflow can validate Docker integration, skip
fallback behavior and command-failure handling on any machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agents.bmad_coding_team as bmad
from tools.docker_tools import DockerBatchResult, DockerCommandResult
from tools.file_tools import STAGING_DIR, clear_staging, clear_workspace


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
        print(f"\nBMAD Docker execution checks: {self.passed}/{self.total} passed")
        return 0 if self.passed == self.total else 1


def _run_with_fake_docker(fake_runner, task: str):
    original_runner = bmad.run_commands_in_docker
    bmad.run_commands_in_docker = fake_runner
    try:
        return bmad.run_bmad_task(task, run_docker=True, clean_workspace=True, use_llm=False, max_revision_cycles=0)
    finally:
        bmad.run_commands_in_docker = original_runner
        clear_staging()
        clear_workspace()


def main() -> int:
    checks = CheckRunner()
    clear_staging()
    clear_workspace()

    # DevOps should prepare a real test-oriented Dockerfile, not a plain app.py run
    team = bmad.BMADCodingTeam()
    step = team.devops.prepare_environment()
    dockerfile = (STAGING_DIR / "Dockerfile").read_text(encoding="utf-8")
    compose = (STAGING_DIR / "docker-compose.yml").read_text(encoding="utf-8")
    checks.check("pip install --no-cache-dir -r requirements.txt" in dockerfile, "Dockerfile installs requirements.txt when present")
    checks.check('["python", "-m", "unittest", "test_app.py"]' in dockerfile, "Dockerfile default command runs unit tests")
    checks.check("command:" in compose and "unittest" in compose, "docker-compose service runs tests")
    checks.check("prepare_sandbox_test_command" in step.actions, "DevOps step records sandbox test preparation")
    clear_staging()

    def fake_skipped(commands, **kwargs):
        return DockerBatchResult(
            success=True,
            skipped=True,
            reason="Docker daemon is not running in test environment.",
            image_name="fake-image",
        )

    skipped = _run_with_fake_docker(
        fake_skipped,
        "Create a Python program to solve quadratic equations with README and tests",
    )
    checks.check(skipped.approved, "workflow remains approved when Docker is unavailable but local QA passed")
    checks.check(any("Docker execution skipped" in check for check in skipped.qa_report.checks), "QA reports Docker skip fallback clearly")
    checks.check(skipped.qa_report.docker_output is not None and "skipped" in skipped.qa_report.docker_output.lower(), "Docker skip report is stored")

    captured_commands: list[tuple[str, list[str]]] = []

    def fake_success(commands, **kwargs):
        captured_commands.extend((name, list(command)) for name, command in commands)
        results = []
        for name, command in commands:
            output = ""
            if name == "docker_unit_tests":
                output = "OK"
            elif name == "quadratic_two_roots_cli":
                output = "Two real roots: 1 and 2"
            elif name == "quadratic_repeated_root_cli":
                output = "One repeated real root: -1"
            elif name == "quadratic_no_real_roots_cli":
                output = "No real roots"
            elif name == "quadratic_invalid_a_cli":
                output = "Error: invalid coefficient"
            results.append(DockerCommandResult(name=name, command=list(command), returncode=0 if name != "quadratic_invalid_a_cli" else 2, stdout=output))
        return DockerBatchResult(
            success=True,
            skipped=False,
            reason="fake docker success",
            image_name="fake-image",
            build_returncode=0,
            commands=results,
        )

    success = _run_with_fake_docker(
        fake_success,
        "Create a Python program to solve quadratic equations with README and tests",
    )
    checks.check(success.approved, "workflow approves when Docker unit and behavior checks pass")
    checks.check(any(name == "docker_unit_tests" for name, _ in captured_commands), "Docker executes generated unit tests")
    checks.check(any(name == "quadratic_two_roots_cli" for name, _ in captured_commands), "Docker executes CLI behavior checks")
    checks.check(any("Docker image build passed" in check for check in success.qa_report.checks), "QA reports Docker build success")
    checks.check(any("Docker check 'quadratic_two_roots_cli' passed" in check for check in success.qa_report.checks), "QA reports passing Docker behavior check")

    def fake_command_failure(commands, **kwargs):
        results = []
        for name, command in commands:
            if name == "docker_unit_tests":
                results.append(DockerCommandResult(name=name, command=list(command), returncode=1, stderr="unit test failed"))
            else:
                results.append(DockerCommandResult(name=name, command=list(command), returncode=0, stdout="placeholder"))
        return DockerBatchResult(
            success=False,
            skipped=False,
            reason="fake docker command failure",
            image_name="fake-image",
            build_returncode=0,
            commands=results,
        )

    failed = _run_with_fake_docker(
        fake_command_failure,
        "Create a Python program to solve quadratic equations with README and tests",
    )
    checks.check(not failed.approved, "workflow rejects Docker command failures when Docker actually ran")
    checks.check(any("docker_unit_tests" in failure for failure in failed.qa_report.failures), "QA failure names the failing Docker command")

    clear_staging()
    clear_workspace()
    return checks.finish()


if __name__ == "__main__":
    raise SystemExit(main())
