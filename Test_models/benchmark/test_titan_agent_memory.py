#!/usr/bin/env python3
"""Focused checks for using Titan as an AI-agent memory layer."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.titan_agent_memory import TitanAgentMemory
from agents.simple_titan_agent import SimpleTitanAgent


def check(condition: bool, label: str, failures: list[str]) -> None:
    if not condition:
        failures.append(label)


def main() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        memory_path = Path(tmp) / "agent_titan_memory.pt"
        memory = TitanAgentMemory(memory_path=memory_path, max_items=100, device="cpu")
        agent = SimpleTitanAgent(memory=memory, ollama_model="llama3.2:1b")

        r1 = agent.ask("/store Lucas Martin's secret code is 8392.", use_ollama=False)
        check("Stored" in r1 or "Saved" in r1, "store command should store a fact", failures)

        r2 = agent.ask("/ask What is Lucas Martin's secret code?", use_ollama=False)
        check("8392" in r2, "agent should answer from Titan memory", failures)

        agent.ask("/store Lucas Martin's secret code is 1245.", use_ollama=False)
        r3 = agent.ask("/ask What is Lucas Martin's secret code?", use_ollama=False)
        check("1245" in r3 and "8392" not in r3, "agent should use updated memory", failures)

        agent.ask("/store Sarah Martin's favorite color is green.", use_ollama=False)
        agent.ask("/store Lucas Martin's favorite color is violet.", use_ollama=False)
        r4 = agent.ask("/ask What is Sarah Martin's favorite color?", use_ollama=False)
        r5 = agent.ask("/ask What is Lucas Martin's favorite color?", use_ollama=False)
        check("green" in r4.lower(), "agent should separate Sarah Martin memory", failures)
        check("violet" in r5.lower(), "agent should separate Lucas Martin memory", failures)

        r6 = agent.ask("/forget Sarah Martin's favorite color", use_ollama=False)
        r7 = agent.ask("/ask What is Sarah Martin's favorite color?", use_ollama=False)
        r8 = agent.ask("/ask What is Lucas Martin's favorite color?", use_ollama=False)
        check("Forgotten" in r6, "forget command should deactivate a targeted memory", failures)
        check("green" not in r7.lower(), "forgotten Sarah color should not be recalled", failures)
        check("violet" in r8.lower(), "forgetting Sarah should not remove Lucas memory", failures)

        context = memory.build_context("What is Lucas Martin's favorite color?")
        check("violet" in context.lower(), "adapter should build a memory context block", failures)

        consolidation = memory.consolidate()
        check(isinstance(consolidation, dict), "consolidate should return a status dictionary", failures)

        memory.save()
        reloaded = TitanAgentMemory(memory_path=memory_path, max_items=100, device="cpu")
        check(len(reloaded.recall("Lucas Martin favorite color", top_k=3)) >= 1, "saved agent memory should reload", failures)

    if failures:
        print("Titan agent memory checks failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Titan agent memory checks: 10/10 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
