#!/usr/bin/env python3
"""Tests for the unified main chat interface."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.titan_agent_memory import AgentMemoryRecord  # type: ignore
from agents.unified_chat_interface import UnifiedChatAssistant, UnifiedChatConfig, route_message  # type: ignore


class FakeMemory:
    def __init__(self) -> None:
        self.stored: list[str] = []
        self.forgotten: list[str] = []
        self.saved = False

    def store(self, text, metadata=None):
        self.stored.append(text)
        return [AgentMemoryRecord(id=len(self.stored), text=text, score=1.0, subject="Project", property="rule")]

    def recall(self, query, top_k=None, min_score=None):
        return [AgentMemoryRecord(id=1, text="Project rule: generated code must include README and tests.", score=0.9, subject="Project", property="rule")]

    def forget(self, query, safe_top_k=5):
        self.forgotten.append(query)
        return [AgentMemoryRecord(id=1, text=query, score=1.0)]

    def consolidate(self, keep_existing_ltm=False, max_items=None, steps=3, include_inactive=False):
        return {"status": "ok", "message": "fake consolidation completed", "steps": steps}

    def build_context(self, query, top_k=None):
        return "Relevant long-term Titan memories:\n1. Project rule: generated code must include README and tests."

    def save(self, path=None):
        self.saved = True

    def load(self, path=None):
        return True

    def stats(self):
        return {"active_items": len(self.stored), "adapter": "FakeMemory"}


class FakeBMADResult:
    final_message = "BMAD coding workflow result: APPROVED\nPublished files: app.py, README.md, test_app.py"


def fake_bmad_runner(task, **kwargs):
    return FakeBMADResult()


def fake_chat_backend(message: str, memory_context: str) -> str:
    return f"chat response to: {message}; memory_used={bool(memory_context)}"


def check(condition: bool, message: str, results: list[str]) -> None:
    if condition:
        results.append(f"PASS: {message}")
    else:
        results.append(f"FAIL: {message}")


def main() -> int:
    results: list[str] = []

    check(route_message("Create a Python program to solve equations with README and tests").kind == "task", "natural coding requests route to BMAD", results)
    check(route_message("Remember that generated apps need tests").kind == "store", "natural remember requests route to Titan store", results)
    check(route_message("What do you remember about project rules?").kind == "recall", "natural memory questions route to Titan recall", results)
    check(route_message("Forget the old API key").kind == "forget", "natural forget requests route to Titan forget", results)
    check(route_message("How are you today?").kind == "chat", "normal messages route to chat", results)
    check(route_message("consolidate memory").kind == "consolidate", "natural consolidation command routes correctly", results)

    memory = FakeMemory()
    assistant = UnifiedChatAssistant(
        UnifiedChatConfig(use_llm_for_chat=True, use_llm_for_coding=True, use_titan_memory=True, color_enabled=False),
        memory=memory,
        chat_backend=fake_chat_backend,
        bmad_runner=fake_bmad_runner,
    )

    store_response = assistant.process("Remember that the project uses BMAD agents")
    check("Saved in Titan memory" in store_response and memory.stored, "assistant stores natural memory facts", results)

    recall_response = assistant.process("What do you remember about project rules?")
    check("Relevant Titan memories" in recall_response, "assistant recalls Titan memories naturally", results)

    task_response = assistant.process("Create a Python program to solve quadratic equations with README and tests")
    check("BMAD coding workflow result" in task_response, "assistant runs BMAD for natural coding task", results)

    chat_response = assistant.process("Explain what this project does")
    check("chat response" in chat_response and "memory_used=True" in chat_response, "assistant chats with memory context", results)

    forget_response = assistant.process("Forget the old API key")
    check("Forgotten" in forget_response and memory.forgotten, "assistant forgets natural memory targets", results)

    consolidation_response = assistant.process("consolidate memory")
    check("consolidated" in consolidation_response.lower(), "assistant can consolidate from natural command", results)

    exit_response = assistant.process("exit")
    check("Bye" in exit_response and memory.saved, "assistant saves memory on exit", results)


    colored_assistant = UnifiedChatAssistant(
        UnifiedChatConfig(use_llm_for_chat=True, use_llm_for_coding=True, use_titan_memory=True, color_enabled=True),
        memory=FakeMemory(),
        chat_backend=fake_chat_backend,
        bmad_runner=fake_bmad_runner,
    )
    check("\033[" in colored_assistant.welcome(), "welcome screen uses ANSI colors when enabled", results)
    check("\033[" in colored_assistant.process("help"), "help text uses ANSI colors when enabled", results)
    plain_assistant = UnifiedChatAssistant(
        UnifiedChatConfig(use_llm_for_chat=True, use_llm_for_coding=True, use_titan_memory=True, color_enabled=False),
        memory=FakeMemory(),
        chat_backend=fake_chat_backend,
        bmad_runner=fake_bmad_runner,
    )
    check("\033[" not in plain_assistant.welcome(), "ANSI colors can be disabled", results)

    for line in results:
        print(line)
    failed = [line for line in results if line.startswith("FAIL")]
    if failed:
        print(f"Unified main chat interface checks: {len(results)-len(failed)}/{len(results)} passed")
        return 1
    print(f"Unified main chat interface checks: {len(results)}/{len(results)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
