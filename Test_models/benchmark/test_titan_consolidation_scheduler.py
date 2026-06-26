#!/usr/bin/env python3
"""Checks for policy-based Titan consolidation scheduling."""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.consolidation_scheduler import ConsolidationPolicy, ConsolidationScheduler
from agents.titan_agent_memory import TitanAgentMemory


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
        print(f"\nTitan consolidation scheduler checks: {self.passed}/{self.total} passed")
        return 0 if self.passed == self.total else 1


class FakeMemory:
    def __init__(self) -> None:
        self.consolidate_calls: list[dict[str, Any]] = []
        self.saved = 0
        self.active_items = 3

    def consolidate(self, keep_existing_ltm: bool = False, max_items: int | None = None, steps: int = 3, include_inactive: bool = False) -> dict[str, Any]:
        call = {
            "keep_existing_ltm": keep_existing_ltm,
            "max_items": max_items,
            "steps": steps,
            "include_inactive": include_inactive,
        }
        self.consolidate_calls.append(call)
        return {"consolidated": self.active_items, "message": "fake consolidation ok", **call}

    def save(self, path: str | Path | None = None) -> None:
        self.saved += 1

    def stats(self) -> dict[str, Any]:
        return {"active_items": self.active_items, "saved": self.saved}


def main() -> int:
    checks = CheckRunner()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        policy = ConsolidationPolicy(
            schedule_hour=2,
            schedule_minute=0,
            min_hours_between_runs=20,
            max_items=2,
            steps=5,
            keep_existing_ltm=True,
            include_inactive=False,
            log_dir=tmp_path,
            state_path=tmp_path / "state.json",
            log_path=tmp_path / "log.jsonl",
        )
        fake = FakeMemory()
        scheduler = ConsolidationScheduler(memory=fake, policy=policy)
        due_time = datetime(2026, 1, 1, 2, 15, tzinfo=timezone.utc)
        not_due_time = datetime(2026, 1, 1, 1, 59, tzinfo=timezone.utc)

        checks.check(not scheduler.should_run_now(not_due_time), "scheduler is not due before the configured hour")
        checks.check(scheduler.should_run_now(due_time), "scheduler is due in the configured nightly window")

        result = scheduler.run_if_due(due_time)
        checks.check(result.ran and result.status == "ok", "run_if_due performs consolidation when due")
        checks.check(fake.consolidate_calls[-1]["max_items"] == 2, "policy max_items is passed to memory.consolidate")
        checks.check(fake.consolidate_calls[-1]["steps"] == 5, "policy steps is passed to memory.consolidate")
        checks.check(fake.consolidate_calls[-1]["keep_existing_ltm"] is True, "policy keep_existing_ltm is passed to memory.consolidate")
        checks.check(fake.saved == 1, "scheduler saves memory after successful consolidation")
        checks.check(policy.state_path.exists(), "scheduler writes a state file")
        checks.check(policy.log_path.exists(), "scheduler writes a JSONL log")

        state = json.loads(policy.state_path.read_text(encoding="utf-8"))
        checks.check("last_run_at" in state, "state file records last_run_at")
        skipped = scheduler.run_if_due(datetime(2026, 1, 1, 2, 45, tzinfo=timezone.utc))
        checks.check(not skipped.ran and skipped.status == "skipped", "scheduler skips a second run in the same nightly window")

        forced = scheduler.run_once(reason="manual_test", current_time=datetime(2026, 1, 1, 3, 0, tzinfo=timezone.utc), force=True)
        checks.check(forced.ran and forced.reason == "manual_test", "manual run_once can force consolidation outside the schedule")

    with tempfile.TemporaryDirectory() as tmp:
        memory_path = Path(tmp) / "agent_titan_memory.pt"
        log_dir = Path(tmp) / "logs"
        memory = TitanAgentMemory(memory_path=memory_path, max_items=50, device="cpu")
        memory.store("BMAD project rule is to write generated files to staging before workspace.")
        memory.store("Validated coding results should be proposed as memory candidates only.")
        real_policy = ConsolidationPolicy(
            schedule_hour=2,
            log_dir=log_dir,
            state_path=log_dir / "state.json",
            log_path=log_dir / "log.jsonl",
            steps=1,
            max_items=5,
        )
        real_scheduler = ConsolidationScheduler(memory=memory, policy=real_policy)
        real_result = real_scheduler.run_once(reason="manual_real_titan_test", current_time=datetime(2026, 1, 2, 2, 0, tzinfo=timezone.utc))
        checks.check(real_result.ran and real_result.status == "ok", "scheduler can consolidate real TitanAgentMemory")
        checks.check(real_result.consolidation.get("consolidated", 0) >= 1, "real Titan consolidation reports consolidated items")
        checks.check(memory_path.exists(), "real Titan memory is saved after scheduler run")
        checks.check(real_policy.log_path.exists(), "real Titan scheduler run writes a log")

    return checks.finish()


if __name__ == "__main__":
    raise SystemExit(main())
