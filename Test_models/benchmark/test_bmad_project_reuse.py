#!/usr/bin/env python3
"""Checks for persistent generated project discovery and reuse context."""

from __future__ import annotations

import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.unified_chat_interface import UnifiedChatAssistant, UnifiedChatConfig, route_message
from tools import project_registry as registry


def _make_temp_registry(tmpdir: Path):
    old = (registry.WORKSPACE_DIR, registry.PROJECTS_DIR, registry.PROJECT_INDEX_PATH)
    registry.WORKSPACE_DIR = tmpdir / "workspace"
    registry.PROJECTS_DIR = registry.WORKSPACE_DIR / "projects"
    registry.PROJECT_INDEX_PATH = registry.PROJECTS_DIR / "project_index.json"
    registry.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    return old


def _restore_registry(old) -> None:
    registry.WORKSPACE_DIR, registry.PROJECTS_DIR, registry.PROJECT_INDEX_PATH = old


def run_checks() -> tuple[int, int]:
    checks = 0
    passed = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks, passed
        checks += 1
        if not condition:
            raise AssertionError(message)
        passed += 1

    with tempfile.TemporaryDirectory() as tmp:
        old = _make_temp_registry(Path(tmp))
        try:
            (registry.WORKSPACE_DIR / "app.py").write_text(
                "def solve_quadratic(a, b, c):\n    return (1.0, 2.0)\n",
                encoding="utf-8",
            )
            (registry.WORKSPACE_DIR / "README.md").write_text("# Quadratic solver\n", encoding="utf-8")
            (registry.WORKSPACE_DIR / "test_app.py").write_text("import unittest\n", encoding="utf-8")

            manifest = registry.archive_generated_project(
                task="Create a Python program to solve quadratic equations with README and tests",
                project_kind="quadratic_solver",
                workspace_files=["app.py", "README.md", "test_app.py"],
                qa_passed=True,
                manual_test_commands=[r"python workspace\app.py 1 -3 2"],
            )

            check(bool(manifest.project_id), "archive should create a project id")
            check((registry.PROJECTS_DIR / manifest.project_id / "project_manifest.json").exists(), "manifest should be written")

            listing = registry.list_generated_projects()
            check(manifest.project_id in listing, "list_generated_projects should show the archive")

            matches = registry.find_generated_projects("reuse the quadratic equation solver", limit=3)
            check(matches and matches[0]["project_id"] == manifest.project_id, "project search should find the quadratic project")

            search_text = registry.format_project_search_results("quadratic equation")
            check(manifest.project_id in search_text, "formatted search should include project id")

            detail = registry.format_project_detail(manifest.project_id)
            check("solve_quadratic" in detail, "project detail should include main functions")

            reuse_context = registry.build_project_reuse_context("modify the quadratic solver")
            check(manifest.project_id in reuse_context, "reuse context should include matched project")
            check("app.py preview" in reuse_context, "reuse context should include app.py preview")

            check(route_message("projects").kind == "projects", "projects should route to project list")
            check(route_message("find project quadratic").kind == "project_search", "find project should route to search")
            check(route_message(f"project {manifest.project_id}").kind == "project_detail", "project <id> should route to detail")

            assistant = UnifiedChatAssistant(
                UnifiedChatConfig(use_titan_memory=False, use_llm_for_chat=False, color_enabled=False)
            )
            project_list = assistant.process("projects")
            check(manifest.project_id in project_list, "chat projects command should display archives")
            project_search = assistant.process("find project quadratic")
            check(manifest.project_id in project_search, "chat project search should display matches")
            project_detail = assistant.process(f"project {manifest.project_id}")
            check("solve_quadratic" in project_detail, "chat project detail should display functions")
        finally:
            _restore_registry(old)

    return passed, checks


if __name__ == "__main__":
    passed, checks = run_checks()
    print(f"BMAD project reuse checks: {passed}/{checks} passed")
