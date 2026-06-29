#!/usr/bin/env python3
"""End-to-end checks for the current BMAD coding workflow.

This benchmark is intentionally higher level than the step-specific tests. It
verifies the complete path the project now cares about:

create code -> run QA -> publish -> archive -> write compact Titan records ->
recall/search archive -> reuse an existing project and create a new version.

It uses deterministic templates and a fake Titan adapter, so it stays fast and
stable without requiring Ollama or a live Docker daemon.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.bmad_coding_team import BMADCodingTeam
from tools.file_tools import clear_staging, clear_workspace
from tools.project_registry import find_generated_projects, project_integrity_status, read_project_manifest


@dataclass
class FakeRecord:
    id: int
    text: str
    score: float = 1.0
    subject: str = "Project"
    property: str = "archive"
    active: bool = True
    metadata: dict[str, Any] | None = None


class FakeProjectMemory:
    """Minimal TitanAgentMemory-like fake used for deterministic benchmarks."""

    def __init__(self) -> None:
        self.loaded_from: list[str | Path | None] = []
        self.recall_queries: list[str] = []
        self.stored: list[tuple[str, dict[str, Any] | None]] = []
        self.saved = False

    def load(self, path: str | Path | None = None) -> bool:
        self.loaded_from.append(path)
        return True

    def recall(self, query: str, top_k: int | None = None, min_score: float | None = None) -> list[FakeRecord]:
        self.recall_queries.append(query)
        return []

    def store(self, text: str, metadata: dict[str, Any] | None = None) -> list[FakeRecord]:
        self.stored.append((text, metadata))
        return [FakeRecord(id=len(self.stored), text=text, metadata=metadata)]

    def save(self, path: str | Path | None = None) -> None:
        self.saved = True


def check(condition: bool, label: str, failures: list[str]) -> None:
    if not condition:
        failures.append(label)


def run_workspace_app(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "workspace/app.py", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )


def main() -> int:
    failures: list[str] = []
    clear_staging()
    clear_workspace(include_project_archives=True)

    memory = FakeProjectMemory()
    team = BMADCodingTeam(
        memory=memory,
        use_titan_memory=True,
        store_approved_memory=True,
        max_revision_cycles=1,
    )

    # 1) Create a validated project and make sure it behaves like real code.
    first = team.run(
        "Create a Python program that encrypts and decrypts messages using Caesar cipher with README and tests",
        clean_workspace=True,
        run_docker=False,
    )
    first_archive = first.project_archive or {}
    first_id = str(first_archive.get("project_id") or "")

    encrypted = run_workspace_app("encrypt", "hello", "3")
    decrypted = run_workspace_app("decrypt", "khoor", "3")
    integrity_after_first = project_integrity_status()
    caesar_matches = find_generated_projects("caesar cipher", limit=5)

    check(first.approved and first.published, "initial Caesar workflow should be approved and published", failures)
    check(first_archive.get("project_kind") == "caesar_cipher", "initial archive should be classified as caesar_cipher", failures)
    check(bool(first_id), "initial workflow should create a project id", failures)
    check(first.memory_validation.stored, "initial workflow should store approved project memory records", failures)
    check(first.memory_validation.records_stored >= 3, "initial workflow should store multiple compact project records", failures)
    check(memory.saved, "fake Titan memory should be saved after storing project records", failures)
    check(any((metadata or {}).get("project_id") == first_id for _text, metadata in memory.stored), "stored metadata should include the initial project id", failures)
    check(encrypted.returncode == 0 and "khoor" in encrypted.stdout.lower(), "Caesar encrypt manual command should work", failures)
    check(decrypted.returncode == 0 and "hello" in decrypted.stdout.lower(), "Caesar decrypt manual command should work", failures)
    check(integrity_after_first.get("ok") is True, "project archive integrity should be ok after initial project", failures)
    check(any(item.get("project_id") == first_id for item in caesar_matches), "project search should find the initial Caesar archive", failures)

    # 2) Create a project, then reuse it to build a new version with extra behavior.
    clear_staging()
    clear_workspace(include_project_archives=True)
    memory = FakeProjectMemory()
    team = BMADCodingTeam(memory=memory, use_titan_memory=True, store_approved_memory=True, max_revision_cycles=1)

    base = team.run(
        "Create a Python program to solve quadratic equations with README and tests",
        clean_workspace=True,
        run_docker=False,
    )
    base_archive = base.project_archive or {}
    base_id = str(base_archive.get("project_id") or "")
    versioned = team.run(
        "Modify the existing quadratic solver to support complex roots with README and tests",
        clean_workspace=True,
        run_docker=False,
    )
    version_archive = versioned.project_archive or {}
    version_id = str(version_archive.get("project_id") or "")
    version_manifest = read_project_manifest(version_id) or {}
    complex_run = run_workspace_app("1", "2", "5", "--complex")
    no_complex_run = run_workspace_app("1", "2", "5")
    integrity_after_version = project_integrity_status()

    check(base.approved and base.published, "base quadratic workflow should be approved and published", failures)
    check(bool(base_id), "base quadratic workflow should create a project id", failures)
    check(versioned.approved and versioned.published, "versioned quadratic workflow should be approved and published", failures)
    check(bool(version_id) and version_id != base_id, "versioned workflow should create a new project id", failures)
    check(base_id in (version_archive.get("metadata", {}) or {}).get("reused_project_ids", []), "versioned archive should reference the base project id", failures)
    check((version_archive.get("metadata", {}) or {}).get("base_project_id") == base_id, "versioned metadata should contain base_project_id", failures)
    check(version_manifest.get("metadata", {}).get("reuse_mode") == "modify_existing", "versioned manifest should record modify_existing reuse mode", failures)
    check(complex_run.returncode == 0 and "Complex roots" in complex_run.stdout and "i" in complex_run.stdout, "versioned CLI should support --complex roots", failures)
    check(no_complex_run.returncode == 0 and "No real roots" in no_complex_run.stdout, "versioned CLI should preserve real-only default behavior", failures)
    check(versioned.memory_validation.stored, "versioned workflow should store approved project memory records", failures)
    check(any((metadata or {}).get("project_id") == version_id for _text, metadata in memory.stored), "stored metadata should include versioned project id", failures)
    check(integrity_after_version.get("ok") is True, "project archive integrity should be ok after versioning", failures)
    check(integrity_after_version.get("index_entries", 0) >= 2, "project index should include base and versioned projects", failures)

    clear_staging()
    clear_workspace(include_project_archives=True)

    if failures:
        print("BMAD full coding workflow checks failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("BMAD full coding workflow checks: 25/25 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
