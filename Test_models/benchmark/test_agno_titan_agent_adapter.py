#!/usr/bin/env python3
"""Tests for the Agno/Titan adapter layer.

This test intentionally does not require Agno to be installed. It validates the
Titan tool wrapper that Agno will use. The real Agno runtime test is manual
because it depends on the external agno package and Ollama being installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.agno_titan_agent import TitanMemoryTools
from agents.titan_agent_memory import TitanAgentMemory


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    memory_path = Path("Test_models/benchmark/results/tmp_agno_titan_memory.pt")
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    if memory_path.exists():
        memory_path.unlink()

    memory = TitanAgentMemory(memory_path=memory_path, d_model=64, hidden_dim=128, top_k=5)
    tools = TitanMemoryTools(memory)
    failures: list[str] = []

    r1 = tools.remember("Lucas Martin's secret code is 8392.")
    check("8392" in r1, "remember should store Lucas secret code", failures)

    recall = tools.recall_memory("What is Lucas Martin's secret code?")
    check("8392" in recall, "recall_memory should retrieve Lucas secret code", failures)

    tools.remember("Lucas Martin's secret code is 1245.")
    recall2 = tools.recall_memory("What is Lucas Martin's secret code?")
    check("1245" in recall2, "recall_memory should prefer updated Lucas code", failures)

    tools.remember("Sarah Martin's favorite color is green.")
    tools.remember("Lucas Martin's favorite color is violet.")
    sarah = tools.recall_memory("What is Sarah Martin's favorite color?")
    lucas = tools.recall_memory("What is Lucas Martin's favorite color?")
    check("green" in sarah.lower(), "Sarah color should be green", failures)
    check("violet" in lucas.lower(), "Lucas color should be violet", failures)

    consolidation = tools.consolidate_memory()
    check("consolid" in consolidation.lower(), "consolidate_memory should return consolidation stats", failures)

    stats = tools.memory_stats()
    check("TitanAgentMemory" in stats or "adapter" in stats, "memory_stats should return adapter stats", failures)

    if failures:
        print("Agno Titan adapter checks FAILED:")
        for failure in failures:
            print("-", failure)
        return 1

    print("Agno Titan adapter checks: 7/7 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
