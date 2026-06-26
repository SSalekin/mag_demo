#!/usr/bin/env python3
"""Unified natural chat interface for MAG Demo.

This module is the new default entry point behind ``main.py``.  It replaces the
old architecture-selection menu with a single chat-style assistant that can:

- chat normally through Ollama;
- use Titan memory naturally, without forcing /store or /ask;
- route coding requests to the BMAD coding workflow;
- keep slash commands available for explicit control;
- stay testable without starting Ollama or an interactive terminal.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.bmad_coding_team import run_bmad_task  # type: ignore
from agents.titan_agent_memory import AgentMemoryRecord, TitanAgentMemory  # type: ignore
from tools.file_tools import clear_staging, clear_workspace, list_staging_files, list_workspace_files  # type: ignore


DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
DEFAULT_MEMORY_PATH = os.getenv("TITAN_MEMORY_PATH", str(ROOT.parent / "agent_titan_memory.pt"))


class ChatMemory(Protocol):
    def store(self, text: str, metadata: dict[str, Any] | None = None) -> list[AgentMemoryRecord]: ...
    def recall(self, query: str, top_k: int | None = None, min_score: float | None = None) -> list[AgentMemoryRecord]: ...
    def forget(self, query: str, safe_top_k: int = 5) -> list[AgentMemoryRecord]: ...
    def consolidate(self, keep_existing_ltm: bool = False, max_items: int | None = None, steps: int = 3, include_inactive: bool = False) -> dict[str, Any]: ...
    def build_context(self, query: str, top_k: int | None = None) -> str: ...
    def save(self, path: str | Path | None = None) -> None: ...
    def load(self, path: str | Path | None = None) -> bool: ...
    def stats(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class UnifiedChatConfig:
    """Runtime configuration for the unified assistant."""

    ollama_model: str = DEFAULT_MODEL
    memory_path: str | Path = DEFAULT_MEMORY_PATH
    memory_top_k: int = 5
    memory_min_score: float = 0.12
    use_titan_memory: bool = True
    use_llm_for_chat: bool = True
    use_llm_for_coding: bool = True
    run_docker: bool = False
    clean_workspace_for_tasks: bool = True
    store_approved_memory: bool = False
    max_revisions: int = 2


@dataclass(frozen=True)
class RoutedAction:
    """Deterministic routing decision for one user message."""

    kind: str
    payload: str
    reason: str


class OllamaChatBackend:
    """Small Ollama wrapper used for normal chat responses."""

    def __init__(self, model: str) -> None:
        self.model = model

    def __call__(self, message: str, memory_context: str = "") -> str:
        if not message.strip():
            return ""
        try:
            import ollama  # type: ignore
        except Exception as exc:
            return (
                "Ollama is not available in this environment. "
                "Install/start Ollama or use BMAD coding commands that can fall back to templates. "
                f"Details: {exc}"
            )

        system = (
            "You are the MAG Demo AI assistant. Be helpful, concise and practical. "
            "You can discuss the project, explain code, and use Titan memory context when relevant. "
            "Do not invent stored memories. If memory context is empty or irrelevant, answer normally."
        )
        user = message
        if memory_context and "No relevant long-term memory found" not in memory_context:
            user = f"Titan memory context:\n{memory_context}\n\nUser message:\n{message}"
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            msg = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", {})
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            return (content or "").strip() or "(empty Ollama response)"
        except Exception as exc:
            return (
                f"Ollama could not answer with model '{self.model}'. "
                f"Try: ollama pull {self.model}. Details: {exc}"
            )


STORE_PATTERNS = (
    r"^remember\b",
    r"^please remember\b",
    r"^store\b",
    r"^save this\b",
    r"^note that\b",
    r"^project rule\s*:",
    r"^from now on\b",
)

RECALL_PATTERNS = (
    r"^what do you remember\b",
    r"^what did i tell you\b",
    r"^do you remember\b",
    r"^what are the project rules\b",
    r"^what is my\b",
    r"^what are my\b",
    r"^tell me what you know about\b",
)

FORGET_PATTERNS = (
    r"^forget\b",
    r"^delete .*memory",
    r"^remove .*memory",
    r"^do not remember\b",
    r"^don't remember\b",
)

TASK_STARTERS = (
    "create", "build", "write", "generate", "make", "develop", "code", "implement",
    "can you create", "can you build", "can you write", "please create", "please build", "please write",
)

TASK_MARKERS = (
    "python", "program", "script", "app", "application", "cli", "readme", "test", "tests",
    "docker", "file", "project", "function", "api", "website", "tool",
)


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _strip_store_prefix(message: str) -> str:
    text = message.strip()
    cleaned = re.sub(r"(?i)^please remember(?: that| this)?[:\s]*", "", text).strip()
    cleaned = re.sub(r"(?i)^remember(?: that| this)?[:\s]*", "", cleaned).strip()
    cleaned = re.sub(r"(?i)^store(?: this)?[:\s]*", "", cleaned).strip()
    cleaned = re.sub(r"(?i)^save this[:\s]*", "", cleaned).strip()
    cleaned = re.sub(r"(?i)^note that[:\s]*", "", cleaned).strip()
    return cleaned or text


def _strip_forget_prefix(message: str) -> str:
    text = message.strip()
    cleaned = re.sub(r"(?i)\b(forget|delete|remove|erase|do not remember|don't remember)\b", "", text).strip()
    cleaned = re.sub(r"(?i)\b(memory|memories|this|that|about)\b", "", cleaned).strip()
    return cleaned or text


def route_message(message: str) -> RoutedAction:
    """Route a natural user message to chat, memory, coding, or commands."""

    text = message.strip()
    lower = text.lower()
    if not text:
        return RoutedAction("empty", "", "empty input")
    if lower in {"exit", "quit", "bye", "/exit", "/quit"}:
        return RoutedAction("exit", "", "exit keyword")
    if lower in {"help", "/help", "?"}:
        return RoutedAction("help", "", "help keyword")
    if lower in {"stats", "memory stats", "/stats"}:
        return RoutedAction("stats", "", "stats keyword")
    if lower in {"workspace", "list workspace", "/workspace"}:
        return RoutedAction("workspace", "", "workspace keyword")
    if lower in {"staging", "list staging", "/staging"}:
        return RoutedAction("staging", "", "staging keyword")
    if lower in {"clear workspace", "/clear-workspace"}:
        return RoutedAction("clear_workspace", "", "clear workspace keyword")
    if lower in {"clear staging", "/clear-staging"}:
        return RoutedAction("clear_staging", "", "clear staging keyword")
    if lower.startswith("/task ") or lower.startswith("/code "):
        return RoutedAction("task", text.split(" ", 1)[1].strip(), "explicit coding command")
    if lower.startswith("/store "):
        return RoutedAction("store", text.split(" ", 1)[1].strip(), "explicit store command")
    if lower.startswith("/ask ") or lower.startswith("/memory "):
        return RoutedAction("recall", text.split(" ", 1)[1].strip(), "explicit memory query")
    if lower.startswith("/forget "):
        return RoutedAction("forget", text.split(" ", 1)[1].strip(), "explicit forget command")
    if lower.startswith("/consolidate") or lower in {"consolidate memory", "consolidate titan memory"}:
        return RoutedAction("consolidate", text, "consolidation command")

    if lower.startswith(TASK_STARTERS) and any(marker in lower for marker in TASK_MARKERS):
        return RoutedAction("task", text, "natural coding request")
    if _matches_any(lower, STORE_PATTERNS) and not lower.startswith("do you remember"):
        return RoutedAction("store", _strip_store_prefix(text), "natural store request")
    if _matches_any(lower, FORGET_PATTERNS):
        return RoutedAction("forget", _strip_forget_prefix(text), "natural forget request")
    if "?" in text and _matches_any(lower, RECALL_PATTERNS):
        return RoutedAction("recall", text, "natural memory question")
    return RoutedAction("chat", text, "normal chat")


class UnifiedChatAssistant:
    """Natural chat assistant combining Titan memory, BMAD and Ollama."""

    def __init__(
        self,
        config: UnifiedChatConfig | None = None,
        *,
        memory: ChatMemory | None = None,
        chat_backend: Callable[[str, str], str] | None = None,
        bmad_runner: Callable[..., Any] = run_bmad_task,
    ) -> None:
        self.config = config or UnifiedChatConfig()
        self.memory = memory or TitanAgentMemory(
            memory_path=self.config.memory_path,
            top_k=self.config.memory_top_k,
            min_score=self.config.memory_min_score,
        )
        self.chat_backend = chat_backend or OllamaChatBackend(self.config.ollama_model)
        self.bmad_runner = bmad_runner
        if self.config.use_titan_memory:
            try:
                self.memory.load(self.config.memory_path)
            except Exception:
                # A missing or incompatible memory file should not prevent chat startup.
                pass

    def welcome(self) -> str:
        return (
            "MAG Demo AI Coding Assistant\n"
            f"LLM model: {self.config.ollama_model}\n"
            f"Titan memory: {'enabled' if self.config.use_titan_memory else 'disabled'}\n"
            "Write naturally. Examples:\n"
            "  Remember that generated apps must include README and tests.\n"
            "  Create a Python program to solve quadratic equations with README and tests\n"
            "  What do you remember about project rules?\n"
            "Type help for commands, exit to quit."
        )

    def help_text(self) -> str:
        return (
            "Available natural actions:\n"
            "- Normal chat: ask anything normally.\n"
            "- Coding: 'Create a Python program ... with README and tests'.\n"
            "- Memory store: 'Remember that ...' or 'Project rule: ...'.\n"
            "- Memory recall: 'What do you remember about ...?'.\n"
            "- Forget: 'Forget ...'.\n"
            "- Consolidate: 'consolidate memory'.\n\n"
            "Optional explicit commands still work:\n"
            "/task, /code, /store, /ask, /forget, /consolidate, /stats, /workspace, /staging, /clear-workspace, /clear-staging, /exit."
        )

    def process(self, message: str) -> str:
        action = route_message(message)
        kind = action.kind
        if kind == "empty":
            return ""
        if kind == "exit":
            self._safe_save()
            return "Bye. Titan memory saved."
        if kind == "help":
            return self.help_text()
        if kind == "stats":
            return self._format_stats()
        if kind == "workspace":
            return "Workspace files:\n" + self._format_file_list(list_workspace_files())
        if kind == "staging":
            return "Staging files:\n" + self._format_file_list(list_staging_files())
        if kind == "clear_workspace":
            return clear_workspace()
        if kind == "clear_staging":
            return clear_staging()
        if kind == "consolidate":
            return self._consolidate()
        if kind == "store":
            return self._store(action.payload)
        if kind == "recall":
            return self._recall(action.payload)
        if kind == "forget":
            return self._forget(action.payload)
        if kind == "task":
            return self._run_task(action.payload)
        return self._chat(action.payload)

    def _safe_save(self) -> None:
        try:
            self.memory.save(self.config.memory_path)
        except Exception:
            pass

    def _store(self, text: str) -> str:
        if not self.config.use_titan_memory:
            return "Titan memory is disabled."
        records = self.memory.store(text, metadata={"source": "unified_chat"})
        self._safe_save()
        if not records:
            return "Nothing was stored."
        lines = ["Saved in Titan memory:"]
        lines.extend(f"- #{record.id}: {record.text}" for record in records)
        return "\n".join(lines)

    def _recall(self, query: str) -> str:
        if not self.config.use_titan_memory:
            return "Titan memory is disabled."
        records = self.memory.recall(query, top_k=self.config.memory_top_k, min_score=self.config.memory_min_score)
        if not records:
            return "No relevant Titan memory found."
        lines = ["Relevant Titan memories:"]
        for record in records:
            subject = f"subject={record.subject}" if record.subject else "subject=?"
            prop = f"property={record.property}" if record.property else "property=?"
            lines.append(f"- #{record.id} score={record.score:.3f} {subject} {prop}: {record.text}")
        return "\n".join(lines)

    def _forget(self, query: str) -> str:
        if not self.config.use_titan_memory:
            return "Titan memory is disabled."
        records = self.memory.forget(query)
        self._safe_save()
        if not records:
            return "No safe matching memory found. Nothing was forgotten."
        return "Forgotten/deactivated:\n" + "\n".join(f"- #{record.id}: {record.text}" for record in records)

    def _consolidate(self) -> str:
        if not self.config.use_titan_memory:
            return "Titan memory is disabled."
        result = self.memory.consolidate(keep_existing_ltm=False, steps=3, include_inactive=False)
        self._safe_save()
        lines = ["Titan memory consolidated."]
        for key in ("message", "status", "replayed", "selected_items", "loss_before", "loss_after"):
            if key in result:
                lines.append(f"- {key}: {result[key]}")
        return "\n".join(lines)

    def _run_task(self, task: str) -> str:
        result = self.bmad_runner(
            task,
            run_docker=self.config.run_docker,
            clean_workspace=self.config.clean_workspace_for_tasks,
            use_titan_memory=self.config.use_titan_memory,
            memory_path=self.config.memory_path,
            memory_top_k=self.config.memory_top_k,
            memory_min_score=self.config.memory_min_score,
            store_approved_memory=self.config.store_approved_memory,
            use_llm=self.config.use_llm_for_coding,
            ollama_model=self.config.ollama_model,
            max_revision_cycles=self.config.max_revisions,
        )
        return getattr(result, "final_message", str(result))

    def _chat(self, message: str) -> str:
        if not self.config.use_llm_for_chat:
            return "LLM chat is disabled. Use a coding request, memory request, or enable chat LLM."
        context = ""
        if self.config.use_titan_memory:
            try:
                context = self.memory.build_context(message, top_k=self.config.memory_top_k)
            except Exception:
                context = ""
        return self.chat_backend(message, context)

    def _format_stats(self) -> str:
        if not self.config.use_titan_memory:
            return "Titan memory is disabled."
        stats = self.memory.stats()
        return "Titan memory stats:\n" + "\n".join(f"- {key}: {value}" for key, value in sorted(stats.items()))

    @staticmethod
    def _format_file_list(result: Any) -> str:
        if isinstance(result, str):
            return result
        if not result:
            return "- none"
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
            return "\n".join(f"- {item}" for item in result)
        return str(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MAG Demo unified AI coding assistant.")
    parser.add_argument("--ollama-model", default=DEFAULT_MODEL, help="Ollama model for normal chat and BMAD LLM generation.")
    parser.add_argument("--memory-path", default=DEFAULT_MEMORY_PATH, help="Titan memory file used by the unified chat.")
    parser.add_argument("--memory-top-k", type=int, default=5, help="Number of memories to retrieve.")
    parser.add_argument("--memory-min-score", type=float, default=0.12, help="Minimum Titan retrieval score.")
    parser.add_argument("--disable-titan-memory", action="store_true", help="Disable Titan memory for this session.")
    parser.add_argument("--disable-chat-llm", action="store_true", help="Disable normal Ollama chat responses.")
    parser.add_argument("--disable-coding-llm", action="store_true", help="Use BMAD deterministic templates only for coding tasks.")
    parser.add_argument("--run-docker", action="store_true", help="Run Docker validation during BMAD coding tasks.")
    parser.add_argument("--keep-workspace", action="store_true", help="Do not clear workspace before each BMAD coding task.")
    parser.add_argument("--store-approved-memory", action="store_true", help="Allow approved BMAD task summaries to be stored in Titan memory.")
    parser.add_argument("--max-revisions", type=int, default=2, help="Maximum BMAD auto-repair cycles.")
    return parser


def config_from_args(args: argparse.Namespace) -> UnifiedChatConfig:
    return UnifiedChatConfig(
        ollama_model=args.ollama_model,
        memory_path=args.memory_path,
        memory_top_k=args.memory_top_k,
        memory_min_score=args.memory_min_score,
        use_titan_memory=not args.disable_titan_memory,
        use_llm_for_chat=not args.disable_chat_llm,
        use_llm_for_coding=not args.disable_coding_llm,
        run_docker=bool(args.run_docker),
        clean_workspace_for_tasks=not args.keep_workspace,
        store_approved_memory=bool(args.store_approved_memory),
        max_revisions=int(args.max_revisions),
    )


def run_interactive_chat(config: UnifiedChatConfig) -> int:
    assistant = UnifiedChatAssistant(config)
    print(assistant.welcome())
    while True:
        try:
            message = input("\nYou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n" + assistant.process("exit"))
            return 0
        if not message:
            continue
        action = route_message(message)
        response = assistant.process(message)
        if response:
            print("\nAI> " + response)
        if action.kind == "exit":
            return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_interactive_chat(config_from_args(args))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
