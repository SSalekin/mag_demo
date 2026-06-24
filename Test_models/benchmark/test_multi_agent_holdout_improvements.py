#!/usr/bin/env python3
"""Focused regression tests for holdout/adversarial multi-agent weaknesses.

Targets:
- unseen latest-value properties (code formatter, API auth method);
- forget+retention tasks;
- consolidation command filtering;
- policy summary routing to evaluator.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.multi_agent_titan_team import MultiAgentTitanTeam


def assert_contains(text: str, expected: str) -> None:
    if expected.lower() not in text.lower():
        raise AssertionError(f"missing {expected!r} in answer:\n{text}")


def assert_not_contains(text: str, forbidden: str) -> None:
    if forbidden.lower() in text.lower():
        raise AssertionError(f"contains forbidden {forbidden!r} in answer:\n{text}")


def main() -> int:
    passed = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # 1) Unseen latest-value property: code formatter.
        team = MultiAgentTitanTeam(memory_path=tmp / "formatter.pt", top_k=8)
        for value in ["black", "yapf", "ruff format"]:
            team.store(f"The current code formatter is {value}.")
        answer = team.run_task("What is the current code formatter right now?", store_decision=False).final_answer
        assert_contains(answer, "ruff format")
        assert_not_contains(answer, "black")
        assert_not_contains(answer, "yapf")
        passed += 1

        # 2) Unseen latest-value property: API auth method.
        team = MultiAgentTitanTeam(memory_path=tmp / "auth.pt", top_k=8)
        for value in ["basic auth", "API key", "JWT"]:
            team.store(f"The current API auth method is {value}.")
        answer = team.run_task("What is the current API auth method right now?", store_decision=False).final_answer
        assert_contains(answer, "JWT")
        assert_not_contains(answer, "basic auth")
        assert_not_contains(answer, "API key")
        passed += 1

        # 3) Forget+retention should forget the temporary secret first, then keep the rule.
        team = MultiAgentTitanTeam(memory_path=tmp / "forget_retention.pt", top_k=8)
        team.store("Lucas Martin's temporary secret code for holdout 1 is TEMP-HOLD-1.")
        team.store("Retention rule holdout 1: benchmark summaries must mention limitations and next steps.")
        team.store("Distractor: Lucas Martin's archived color was blue-1.")
        answer = team.run_task(
            "Forget Lucas Martin's temporary secret code, then explain the remaining retention rule.",
            store_decision=False,
        ).final_answer
        assert_contains(answer, "limitations")
        assert_contains(answer, "next steps")
        assert_not_contains(answer, "TEMP-HOLD-1")
        assert_not_contains(answer, "blue-1")
        passed += 1

        # 4) Consolidation should retrieve the rule, not the operational command.
        team = MultiAgentTitanTeam(memory_path=tmp / "consolidation.pt", top_k=8)
        team.store("Consolidation holdout rule B: reports must include CSV output and markdown summaries.")
        team.store("Please consolidate the active project rules after storing them.")
        team.consolidate()
        answer = team.run_task("After consolidation, what project rule should be preserved?", store_decision=False).final_answer
        assert_contains(answer, "CSV")
        assert_contains(answer, "markdown")
        assert_not_contains(answer, "Please consolidate")
        passed += 1

        # 5) Policy summary should route to evaluator and preserve limitations.
        team = MultiAgentTitanTeam(memory_path=tmp / "policy.pt", top_k=8)
        team.store("Tester policy: tests must run before benchmarks.")
        team.store("Evaluator policy: reports must mention limitations.")
        team.store("Manager policy: only the manager should write validated decisions into Titan memory.")
        result = team.run_task("Summarize the team policies for memory, tests and evaluation", store_decision=False)
        answer = result.final_answer
        assert_contains(answer, "limitations")
        if "evaluator" not in result.selected_agents:
            raise AssertionError(f"evaluator not selected: {result.selected_agents}")
        passed += 1

    print(f"Multi-agent holdout improvement checks: {passed}/5 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
