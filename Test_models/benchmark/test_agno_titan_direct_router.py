#!/usr/bin/env python3
"""Focused checks for deterministic Titan routing in the Agno CLI."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.agno_titan_agent import TitanMemoryTools, handle_direct_memory_intent
from agents.titan_agent_memory import TitanAgentMemory


def main() -> int:
    passed = 0
    total = 7
    with tempfile.TemporaryDirectory() as tmp:
        memory = TitanAgentMemory(memory_path=Path(tmp) / "router_memory.pt", top_k=5)
        tools = TitanMemoryTools(memory)

        out = handle_direct_memory_intent("Remember that Lucas Martin's secret code is 8392.", tools)
        assert out and "Stored in Titan memory" in out
        passed += 1

        out = handle_direct_memory_intent("What is Lucas Martin's secret code?", tools)
        assert out and "8392" in out
        passed += 1

        out = handle_direct_memory_intent("/store Lucas Martin's secret code is 1245.", tools)
        assert out and "1245" in out
        passed += 1

        out = handle_direct_memory_intent("/ask What is Lucas Martin's secret code?", tools)
        assert out and "1245" in out and "8392" not in out
        passed += 1

        out = handle_direct_memory_intent("Consolidate memory.", tools)
        assert out and "consolidat" in out.lower()
        passed += 1

        out = handle_direct_memory_intent("/stats", tools)
        assert out and "adapter" in out.lower()
        passed += 1

        out = handle_direct_memory_intent("/help", tools)
        assert out and "Commands" in out
        passed += 1

    print(f"Agno Titan direct router checks: {passed}/{total} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
