#!/usr/bin/env python3
"""Focused tests for the Titan-backed multi-agent team prototype."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.multi_agent_titan_team import MultiAgentTitanTeam, handle_cli_message


def check(condition: bool, label: str, failures: list[str]) -> None:
    if not condition:
        failures.append(label)


def main() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        memory_path = Path(tmp) / "multi_agent_test_memory.pt"
        team = MultiAgentTitanTeam(memory_path=memory_path, ollama_model="llama3.2:1b")

        out = handle_cli_message(team, "Remember that this project uses pytest and type hints.")
        check("pytest" in out.lower(), "store project conventions", failures)

        out = handle_cli_message(team, "What testing framework does this project use?")
        check("pytest" in out.lower(), "recall shared Titan memory", failures)

        result = team.run_task("Create a plan to add a FastAPI endpoint for Titan memory search.")
        agents = set(result.selected_agents)
        check("manager" in agents, "manager selected", failures)
        check("devweb" in agents, "devweb selected for API task", failures)
        check("devsoft" in agents, "devsoft selected for Titan/code task", failures)
        check("tester" in agents, "tester included", failures)
        check("evaluator" in agents, "evaluator included", failures)
        check("pytest" in result.final_answer.lower(), "memory convention used in team output", failures)

        out = handle_cli_message(team, "/agents")
        check("devops" in out and "evaluator" in out, "role summary", failures)

        out = handle_cli_message(team, "/consolidate")
        check("consolidated" in out.lower() or "ltm" in out.lower(), "consolidation command", failures)

        team.save()
        team2 = MultiAgentTitanTeam(memory_path=memory_path, ollama_model="llama3.2:1b")
        loaded = team2.memory.load()
        out = team2.ask_memory("What testing framework does this project use?")
        check(loaded, "memory file loaded", failures)
        check("pytest" in out.lower(), "multi-session shared memory recall", failures)

    total = 10
    passed = total - len(failures)
    print(f"Multi-agent Titan team checks: {passed}/{total} passed")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
