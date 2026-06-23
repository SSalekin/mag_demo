#!/usr/bin/env python3
"""Minimal AI agent using Titan as long-term memory.

This is deliberately framework-free. It validates the architecture before
integrating Agno. Later, this same TitanAgentMemory adapter can be used as Agno
agent tools: remember(), recall(), forget(), consolidate_memory().
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.titan_agent_memory import TitanAgentMemory


class SimpleTitanAgent:
    """Small single-agent prototype with Titan long-term memory."""

    def __init__(self, memory: Optional[TitanAgentMemory] = None, ollama_model: Optional[str] = None) -> None:
        self.memory = memory or TitanAgentMemory()
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        self.short_term_messages: List[str] = []

    def remember(self, text: str) -> str:
        records = self.memory.store(text)
        if not records:
            return "Nothing was stored."
        return "Stored in Titan memory:\n" + "\n".join(f"- #{r.id}: {r.text}" for r in records)

    def recall(self, query: str, top_k: int = 5) -> str:
        return self.memory.build_context(query, top_k=top_k)

    def forget(self, query: str) -> str:
        records = self.memory.forget(query)
        if not records:
            return "No safe matching memory was forgotten."
        return "Forgotten Titan memory:\n" + "\n".join(f"- #{r.id}: {r.text}" for r in records)

    def consolidate_memory(self, keep_existing_ltm: bool = False) -> str:
        result = self.memory.consolidate(keep_existing_ltm=keep_existing_ltm)
        return "Titan memory consolidation result: " + str(result)

    def build_prompt(self, user_message: str) -> List[dict]:
        memory_context = self.memory.build_context(user_message, top_k=5)
        recent_context = "\n".join(self.short_term_messages[-6:]) or "No recent short-term context."
        system = (
            "You are a helpful AI agent with two memory systems.\n"
            "Short-term memory is the recent conversation context.\n"
            "Long-term memory is provided by Titan.\n"
            "Use the Titan memories when they are relevant. If memory is missing, say so."
        )
        user = (
            f"SHORT-TERM CONTEXT:\n{recent_context}\n\n"
            f"LONG-TERM TITAN MEMORY:\n{memory_context}\n\n"
            f"USER REQUEST:\n{user_message}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def ask(self, user_message: str, use_ollama: bool = True) -> str:
        stripped = user_message.strip()
        lower = stripped.lower()
        self.short_term_messages.append(f"User: {stripped}")

        if lower.startswith("/store "):
            answer = self.remember(stripped[len("/store "):].strip())
        elif lower.startswith("/ask "):
            answer = self.answer_with_memory(stripped[len("/ask "):].strip(), use_ollama=use_ollama)
        elif lower.startswith("/search "):
            answer = self.recall(stripped[len("/search "):].strip())
        elif lower.startswith("/forget "):
            answer = self.forget(stripped[len("/forget "):].strip())
        elif lower.startswith("/consolidate"):
            answer = self.consolidate_memory(keep_existing_ltm="--keep" in lower)
        else:
            answer = self.answer_with_memory(stripped, use_ollama=use_ollama)

        self.short_term_messages.append(f"Assistant: {answer}")
        return answer

    def answer_with_memory(self, user_message: str, use_ollama: bool = True) -> str:
        records = self.memory.recall(user_message, top_k=5)
        # Deterministic answer when the retrieved memory is clear enough. This
        # makes the local prototype testable without depending on an LLM.
        if records and records[0].score >= 0.40:
            return f"Based on Titan memory: {records[0].text}"
        if not use_ollama:
            return "I do not know based on Titan memory."

        try:
            import ollama  # type: ignore
            response = ollama.chat(model=self.ollama_model, messages=self.build_prompt(user_message), stream=False)
            return response.get("message", {}).get("content", "").strip() or "No response generated."
        except Exception as exc:
            return f"[Ollama unavailable: {exc}]\n" + self.memory.build_context(user_message)


def main() -> int:
    agent = SimpleTitanAgent(memory=TitanAgentMemory(memory_path="agent_titan_memory.pt"))
    print("Simple Titan Agent. Commands: /store, /ask, /search, /forget, /consolidate, /exit")
    while True:
        try:
            msg = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg.lower() in {"/exit", "exit", "quit"}:
            break
        print(agent.ask(msg))
    agent.memory.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
