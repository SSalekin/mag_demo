#!/usr/bin/env python3
"""Focused regression tests for the improved multi-agent Titan team.

These checks target the weaknesses found in the large benchmark:
- noisy memory/task filtering;
- update recency;
- identity collision filtering;
- role routing from memory context.
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

        # 1) Noisy query should not repeat benchmark noise in the final answer.
        team = MultiAgentTitanTeam(memory_path=tmp / "noise.pt", top_k=6)
        team.store("Noise distractor: banana router sticker coffee hotel orange.")
        team.store("Critical memory: the team uses shared Titan memory for long-term project recall.")
        answer = team.run_task("Ignore banana !!! router ### coffee ??? What memory does the team use for long-term recall?", store_decision=False).final_answer
        assert_contains(answer, "Titan")
        for word in ["banana", "coffee", "hotel", "router"]:
            assert_not_contains(answer, word)
        passed += 1

        # 2) Update-heavy project property should keep only the latest value.
        team = MultiAgentTitanTeam(memory_path=tmp / "updates.pt", top_k=6)
        for value in ["Gemini", "Claude", "Ollama"]:
            team.store(f"The preferred LLM provider for current tests is {value}.")
        answer = team.run_task("What is the current preferred LLM provider for current tests?", store_decision=False).final_answer
        assert_contains(answer, "Ollama")
        assert_not_contains(answer, "Gemini")
        assert_not_contains(answer, "Claude")
        passed += 1

        # 3) Identity collision should not include the nearby wrong owner.
        team = MultiAgentTitanTeam(memory_path=tmp / "identity.pt", top_k=6)
        team.store("Emma Laurent owns the benchmark report task.")
        team.store("Ethan Laurent owns the database migration task.")
        answer = team.run_task("Who owns the benchmark report task?", store_decision=False).final_answer
        assert_contains(answer, "Emma Laurent")
        assert_not_contains(answer, "Ethan Laurent")
        passed += 1

        # 4) Role routing should use memory content, not only the task wording.
        team = MultiAgentTitanTeam(memory_path=tmp / "routing.pt", top_k=6)
        team.store("Critical project convention for billing: use FastAPI, pytest, and type hints.")
        result = team.run_task("Create a plan for the billing work and preserve the critical convention.", store_decision=False)
        answer = result.final_answer
        assert_contains(answer, "FastAPI")
        assert_contains(answer, "pytest")
        for role in ["devweb", "devsoft", "tester", "evaluator"]:
            if role not in result.selected_agents:
                raise AssertionError(f"missing routed role {role!r}; selected={result.selected_agents}")
        passed += 1

    print(f"Multi-agent routing improvement checks: {passed}/4 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
