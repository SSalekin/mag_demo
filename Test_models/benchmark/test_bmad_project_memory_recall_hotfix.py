#!/usr/bin/env python3
"""Checks for robust project memory recall and safe archive reuse routing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.bmad_coding_team import BMADCodingTeam
from agents.titan_agent_memory import AgentMemoryRecord
from agents.unified_chat_interface import UnifiedChatAssistant, UnifiedChatConfig
from tools.file_tools import clear_staging, clear_workspace
from tools.project_memory import build_project_memory_records
from tools.project_registry import archive_generated_project


class EmptyMemory:
    def store(self, text, metadata=None):
        return [AgentMemoryRecord(id=1, text=text, metadata=metadata or {})]

    def recall(self, query, top_k=None, min_score=None):
        return []

    def forget(self, query, safe_top_k=5):
        return []

    def consolidate(self, keep_existing_ltm=False, max_items=None, steps=3, include_inactive=False):
        return {"status": "ok"}

    def build_context(self, query, top_k=None):
        return "No relevant long-term memory found."

    def save(self, path=None):
        pass

    def load(self, path=None):
        return True

    def stats(self):
        return {"active_items": 0}


def check(condition: bool, label: str, failures: list[str]) -> None:
    if not condition:
        failures.append(label)


def main() -> int:
    failures: list[str] = []
    clear_staging()
    clear_workspace(include_project_archives=True)

    # Build a realistic project archive without needing the full BMAD workflow.
    workspace = ROOT / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "app.py").write_text("def is_valid_email(value):\n    return '@' in value\n", encoding="utf-8")
    (workspace / "README.md").write_text("# Email validator\n", encoding="utf-8")
    (workspace / "test_app.py").write_text("from app import is_valid_email\n", encoding="utf-8")
    manifest = archive_generated_project(
        task="Create a Python program that validates email addresses and phone numbers with README and tests",
        project_kind="email_phone_validator",
        workspace_files=["README.md", "app.py", "test_app.py"],
        qa_passed=True,
        manual_test_commands=["python workspace\\app.py email test@example.com"],
    )

    records = build_project_memory_records(manifest.to_dict())
    joined_records = "\n".join(record.text for record in records).lower()
    check("email validator" in joined_records, "project memory records should include natural email validator aliases", failures)
    check("phone validator" in joined_records, "project memory records should include phone validator aliases", failures)

    assistant = UnifiedChatAssistant(
        UnifiedChatConfig(use_titan_memory=True, use_llm_for_chat=False, color_enabled=False),
        memory=EmptyMemory(),
        chat_backend=lambda message, memory_context: "chat",
        bmad_runner=lambda *args, **kwargs: None,
    )
    response = assistant.process("What do you remember about email validator projects?")
    check("Generated project archive matches" in response, "recall should fall back to project archive search", failures)
    check(str(manifest.project_id) in response, "project archive fallback should include project id", failures)
    check("email_phone_validator" in response, "project archive fallback should include project kind", failures)

    team = BMADCodingTeam(use_titan_memory=False)
    fresh_context = team._retrieve_project_reuse_context("Create a Python program that validates email addresses and phone numbers with README and tests")
    reuse_context = team._retrieve_project_reuse_context("Modify the existing email validator project to support French phone numbers with README and tests")
    check(fresh_context == "", "fresh create requests should not inject unrelated project reuse context", failures)
    check("email_phone_validator" in reuse_context, "explicit modify requests should inject relevant archive context", failures)

    clear_staging()
    clear_workspace(include_project_archives=True)

    if failures:
        print("BMAD project memory recall hotfix checks failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("BMAD project memory recall hotfix checks: 7/7 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
