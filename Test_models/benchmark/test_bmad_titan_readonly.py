#!/usr/bin/env python3
"""Checks for BMAD + Titan memory read-only integration.

This test does not use Agno or Ollama. It injects a fake Titan-like memory object
so the BMAD workflow can prove that it retrieves memory context without calling
write-side operations such as store, save, forget or consolidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.bmad_coding_team import BMADCodingTeam
from tools.file_tools import WORKSPACE_DIR, clear_staging, clear_workspace


@dataclass
class FakeMemoryRecord:
    id: int
    text: str
    score: float
    subject: str
    property: str
    active: bool = True


class FakeReadOnlyTitanMemory:
    """Tiny fake implementing the read-only TitanAgentMemory surface."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def load(self, path: str | Path | None = None) -> bool:
        self.calls.append(("load", path))
        return True

    def recall(self, query: str, top_k: int | None = None, min_score: float | None = None) -> list[FakeMemoryRecord]:
        self.calls.append(("recall", query, top_k, min_score))
        return [
            FakeMemoryRecord(
                id=1,
                text="Project rule: keep generated code self-contained and include pytest-style tests when possible.",
                score=0.91,
                subject="Project rule",
                property="coding_convention",
            )
        ]

    # Write-side methods deliberately fail if the BMAD workflow calls them.
    def store(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("BMAD step 3 must not call store()")

    def forget(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("BMAD step 3 must not call forget()")

    def save(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("BMAD step 3 must not call save()")

    def consolidate(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("BMAD step 3 must not call consolidate()")


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
        print(f"\nBMAD Titan read-only checks: {self.passed}/{self.total} passed")
        return 0 if self.passed == self.total else 1


def _workspace_text(path: str) -> str:
    return (WORKSPACE_DIR / path).read_text(encoding="utf-8")


def main() -> int:
    checks = CheckRunner()
    clear_staging()
    clear_workspace()

    fake_memory = FakeReadOnlyTitanMemory()
    team = BMADCodingTeam(memory=fake_memory, use_titan_memory=True, memory_top_k=3, memory_min_score=0.2)
    result = team.run("Create a Python CLI app for the project", run_docker=False, clean_workspace=True)

    checks.check(result.approved, "workflow remains approved with read-only memory enabled")
    checks.check(result.published, "workflow still publishes validated output")
    checks.check(result.memory_context.enabled, "memory context is enabled")
    checks.check(result.memory_context.loaded, "memory load is attempted before recall")
    checks.check(result.memory_context.source == "injected_memory", "injected memory source is reported")
    checks.check(len(result.memory_context.records) == 1, "one memory record is retrieved")
    checks.check("Project rule" in result.memory_context.context, "memory context contains retrieved project rule")
    checks.check("retrieve_titan_memory_read_only" in result.steps[0].actions, "Project Manager retrieves Titan memory read-only")
    checks.check("consume_read_only_memory_context" in result.steps[1].actions, "Business Agent receives memory context")
    checks.check("test_app.py" in result.workspace_files, "memory context can influence deterministic requirements")
    checks.check("README.md" in result.workspace_files, "README is published")
    checks.check("Titan long-term memory was reviewed in read-only mode" in _workspace_text("README.md"), "README notes read-only memory review")
    checks.check(result.memory_candidate is not None, "result exposes a candidate memory")
    checks.check("do not store automatically" in result.memory_candidate, "candidate memory is explicitly not auto-stored")
    checks.check(result.memory_validation.should_store is False, "read-only run keeps memory write gate disabled")
    checks.check(result.memory_validation.stored is False, "read-only run does not store memory")
    checks.check(result.memory_validation.store_result == "skipped_write_disabled", "read-only run reports disabled write gate")
    checks.check("Memory candidate (not stored automatically)" in result.final_message, "final message includes candidate memory")
    checks.check("Memory validation:" in result.final_message, "final message includes memory validation gate")

    called_names = [call[0] for call in fake_memory.calls]
    checks.check(called_names == ["load", "recall"], "only read-side memory methods are called")
    checks.check("store" not in called_names and "save" not in called_names, "no write-side memory calls occurred")

    return checks.finish()


if __name__ == "__main__":
    raise SystemExit(main())
