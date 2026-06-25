#!/usr/bin/env python3
"""Checks for BMAD controlled memory validation and explicit Titan writes.

Step 4 keeps Titan memory clean by default. A generated memory candidate is
stored only when the Evaluator/Project Manager gate approves it and the caller
explicitly enables store_approved_memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.bmad_coding_team import BMADCodingTeam, BMADStep, QAReport
from tools.file_tools import clear_staging, clear_workspace


@dataclass
class FakeMemoryRecord:
    id: int
    text: str
    score: float = 1.0
    subject: str = "Project"
    property: str = "rule"
    active: bool = True


class FakeControlledTitanMemory:
    """Tiny fake implementing both read and controlled write memory calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.stored_texts: list[str] = []

    def load(self, path: str | Path | None = None) -> bool:
        self.calls.append(("load", path))
        return True

    def recall(self, query: str, top_k: int | None = None, min_score: float | None = None) -> list[FakeMemoryRecord]:
        self.calls.append(("recall", query, top_k, min_score))
        return [
            FakeMemoryRecord(
                id=1,
                text="Project rule: approved BMAD outputs may be stored only through the explicit memory gate.",
                score=0.93,
                subject="Project rule",
                property="memory_policy",
            )
        ]

    def store(self, text: str, metadata: dict[str, Any] | None = None) -> list[FakeMemoryRecord]:
        self.calls.append(("store", text, metadata))
        self.stored_texts.append(text)
        return [FakeMemoryRecord(id=2, text=text, subject="BMAD", property="validated_result")]

    def save(self, path: str | Path | None = None) -> None:
        self.calls.append(("save", path))

    def forget(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Controlled BMAD memory validation must not call forget()")

    def consolidate(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Controlled BMAD memory validation must not call consolidate()")


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
        print(f"\nBMAD controlled memory validation checks: {self.passed}/{self.total} passed")
        return 0 if self.passed == self.total else 1


def _call_names(memory: FakeControlledTitanMemory) -> list[str]:
    return [call[0] for call in memory.calls]


def main() -> int:
    checks = CheckRunner()
    clear_staging()
    clear_workspace()

    # Default: candidate is validated but write gate remains closed.
    default_memory = FakeControlledTitanMemory()
    default_team = BMADCodingTeam(memory=default_memory, use_titan_memory=True)
    default_result = default_team.run("Create a Python CLI app with a basic test", clean_workspace=True)

    checks.check(default_result.approved, "default workflow is approved")
    checks.check(default_result.memory_candidate is not None, "default workflow proposes a memory candidate")
    checks.check(default_result.memory_validation.should_store is False, "default memory gate does not request storage")
    checks.check(default_result.memory_validation.stored is False, "default memory gate does not store")
    checks.check(default_result.memory_validation.store_result == "skipped_write_disabled", "default memory gate reports disabled write")
    checks.check(_call_names(default_memory) == ["load", "recall"], "default run uses only read-side memory calls")

    # Explicit write gate: approved + published + QA-passing output is stored and saved.
    write_memory = FakeControlledTitanMemory()
    write_team = BMADCodingTeam(
        memory=write_memory,
        use_titan_memory=True,
        store_approved_memory=True,
    )
    write_result = write_team.run("Create a Python CLI app with a README and a basic test", clean_workspace=True)

    checks.check(write_result.approved and write_result.published, "explicit-write workflow is approved and published")
    checks.check(write_result.memory_validation.should_store is True, "explicit write gate approves storage")
    checks.check(write_result.memory_validation.stored is True, "explicit write gate stores memory")
    checks.check(write_result.memory_validation.records_stored == 1, "one memory record is reported as stored")
    checks.check(write_result.memory_validation.store_result == "stored_and_saved", "memory result reports stored and saved")
    checks.check(_call_names(write_memory) == ["load", "recall", "store", "save"], "explicit write run calls load, recall, store and save")
    checks.check(len(write_memory.stored_texts) == 1, "fake memory received exactly one stored candidate")
    checks.check("BMAD coding workflow approved" in write_memory.stored_texts[0], "stored candidate records approved workflow")
    checks.check("Memory validation:" in write_result.final_message, "final message includes memory validation section")
    checks.check("stored: True" in write_result.final_message, "final message reports stored memory")
    checks.check("store_validated_memory" in write_result.steps[-1].actions, "Project Manager final step records controlled storage")

    # Rejected/failed QA: even with the write gate enabled, memory must not be stored.
    failing_memory = FakeControlledTitanMemory()
    failing_team = BMADCodingTeam(
        memory=failing_memory,
        use_titan_memory=True,
        store_approved_memory=True,
    )

    def forced_failed_qa(run_docker: bool = False) -> tuple[QAReport, BMADStep]:
        report = QAReport(passed=False, checks=["forced QA failure"], failures=["forced failure"])
        step = BMADStep(
            agent="QA",
            role="validation",
            summary="QA failed by test override.",
            actions=["forced_failure"],
            status="failed",
        )
        return report, step

    failing_team.qa.validate = forced_failed_qa  # type: ignore[method-assign]
    failing_result = failing_team.run("Create a Python CLI app that should fail QA", clean_workspace=True)

    checks.check(failing_result.approved is False, "failing workflow is rejected")
    checks.check(failing_result.published is False, "failing workflow is not published")
    checks.check(failing_result.memory_validation.should_store is False, "failing workflow is not eligible for memory storage")
    checks.check(failing_result.memory_validation.stored is False, "failing workflow does not store memory")
    checks.check(_call_names(failing_memory) == ["load", "recall"], "failing workflow never calls store or save")
    checks.check("memory_write_skipped" in failing_result.steps[-1].actions, "Project Manager skips memory write after failure")

    return checks.finish()


if __name__ == "__main__":
    raise SystemExit(main())
