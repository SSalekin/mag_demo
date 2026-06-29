#!/usr/bin/env python3
"""Checks for explicit archived-project reuse and file previews."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.bmad_coding_team import BusinessAgent
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
                "def calculate_statistics(values):\n    return {'count': len(values)}\n\nif __name__ == '__main__':\n    print('stats')\n",
                encoding="utf-8",
            )
            (registry.WORKSPACE_DIR / "README.md").write_text("# CSV statistics\n", encoding="utf-8")
            (registry.WORKSPACE_DIR / "test_app.py").write_text(
                "from app import calculate_statistics\n",
                encoding="utf-8",
            )

            manifest = registry.archive_generated_project(
                task="Create a Python program to read a CSV file and calculate statistics with README and tests",
                project_kind="csv_statistics",
                workspace_files=["app.py", "README.md", "test_app.py"],
                qa_passed=True,
                manual_test_commands=[r"python workspace\app.py workspace\sample.csv --column score"],
            )

            check(registry.extract_project_id_from_text(f"modify project {manifest.project_id} to add median") == manifest.project_id,
                  "explicit project id should be extracted from natural text")

            exact_matches = registry.find_generated_projects(f"modify project {manifest.project_id} to add median", limit=3)
            check(exact_matches and exact_matches[0]["project_id"] == manifest.project_id, "exact id search should prioritize the referenced project")
            check(exact_matches[0].get("match_reason") == "explicit_project_id", "exact search should expose match reason")

            preview = registry.format_project_file_preview(manifest.project_id, "app.py")
            check("calculate_statistics" in preview, "project file preview should read archived app.py")

            check(route_message(f"reuse project {manifest.project_id} to add median with README and tests").kind == "task",
                  "reuse project <id> should route to BMAD task")
            check(route_message(f"project {manifest.project_id} add median with README and tests").kind == "task",
                  "project <id> plus modification words should route to BMAD task")
            check(route_message(f"project {manifest.project_id}").kind == "project_detail",
                  "plain project <id> should still show details")
            check(route_message(f"project file {manifest.project_id} app.py").kind == "project_file",
                  "project file command should route to file preview")

            agent = BusinessAgent()
            context = registry.build_project_reuse_context(f"modify project {manifest.project_id} to add median")
            spec, _step = agent.define_spec(f"modify project {manifest.project_id} to add median with README and tests", project_reuse_context=context)
            check(spec.base_project_id == manifest.project_id, "BusinessAgent should select exact base project id")
            check(spec.project_kind == "csv_statistics", "BusinessAgent should inherit archived project kind when request is generic")
            check(spec.reuse_mode == "modify_existing", "BusinessAgent should enable modify_existing reuse mode")

            assistant = UnifiedChatAssistant(
                UnifiedChatConfig(use_titan_memory=False, use_llm_for_chat=False, color_enabled=False)
            )
            file_text = assistant.process(f"project file {manifest.project_id} app.py")
            check("calculate_statistics" in file_text, "chat project file command should display app.py preview")
        finally:
            _restore_registry(old)

    return passed, checks


if __name__ == "__main__":
    passed, checks = run_checks()
    print(f"BMAD project reuse v2 checks: {passed}/{checks} passed")
