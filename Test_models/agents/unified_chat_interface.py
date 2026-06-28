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
from memory.consolidation_scheduler import ConsolidationPolicy, ConsolidationRunResult, ConsolidationScheduler  # type: ignore
from tools.file_tools import clear_staging, clear_workspace, list_staging_files, list_workspace_files  # type: ignore
from tools.project_registry import (  # type: ignore
    format_project_detail,
    format_project_search_results,
    list_generated_projects,
)


DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
DEFAULT_MEMORY_PATH = os.getenv("TITAN_MEMORY_PATH", str(ROOT.parent / "agent_titan_memory.pt"))
DEFAULT_COLOR_ENABLED = os.getenv("NO_COLOR", "").strip() == ""


class Ansi:
    """Small ANSI helper used to keep the terminal interface readable without extra dependencies."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


def color_text(text: str, color: str = "", *, bold: bool = False, enabled: bool = True) -> str:
    """Return colored text when enabled, otherwise plain text."""

    if not enabled:
        return text
    prefix = f"{Ansi.BOLD if bold else ''}{color}"
    return f"{prefix}{text}{Ansi.RESET}" if prefix else text


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
    color_enabled: bool = DEFAULT_COLOR_ENABLED
    auto_consolidation_enabled: bool = True
    consolidation_hour: int = int(os.getenv("TITAN_CONSOLIDATION_HOUR", "2"))
    consolidation_minute: int = int(os.getenv("TITAN_CONSOLIDATION_MINUTE", "0"))
    consolidation_steps: int = int(os.getenv("TITAN_CONSOLIDATION_STEPS", "3"))
    consolidation_max_items: int | None = None
    consolidation_min_hours_between_runs: float = 20.0
    consolidation_log_dir: str | Path = ROOT / "logs"
    force_startup_consolidation: bool = False


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
    if lower in {"projects", "list projects", "generated projects", "project archives", "/projects"}:
        return RoutedAction("projects", "", "generated projects keyword")
    if lower.startswith("find project ") or lower.startswith("search project "):
        return RoutedAction("project_search", text.split(" ", 2)[2].strip(), "project archive search")
    if lower.startswith("show project ") or lower.startswith("project "):
        return RoutedAction("project_detail", text.split(" ", 2)[2].strip() if lower.startswith("show project ") else text.split(" ", 1)[1].strip(), "project archive detail")
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
    if lower in {"consolidation status", "nightly consolidation status", "/consolidation-status"}:
        return RoutedAction("consolidation_status", text, "consolidation status command")
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
        self.startup_consolidation_result: ConsolidationRunResult | None = None
        if self.config.use_titan_memory:
            try:
                self.memory.load(self.config.memory_path)
            except Exception:
                # A missing or incompatible memory file should not prevent chat startup.
                pass
            self.startup_consolidation_result = self._run_startup_consolidation()

    def _c(self, text: str, color: str = "", *, bold: bool = False) -> str:
        return color_text(text, color, bold=bold, enabled=self.config.color_enabled)

    def welcome(self) -> str:
        return (
            self._c("MAG Demo AI Coding Assistant", Ansi.CYAN, bold=True) + "\n"
            + self._c("LLM model:", Ansi.BLUE, bold=True) + f" {self.config.ollama_model}\n"
            + self._c("Titan memory:", Ansi.MAGENTA, bold=True) + f" {'enabled' if self.config.use_titan_memory else 'disabled'}\n"
            + self._c("Nightly consolidation:", Ansi.BLUE, bold=True) + f" {self._format_startup_consolidation_inline()}\n"
            + self._c("Docker validation:", Ansi.YELLOW, bold=True) + f" {'enabled' if self.config.run_docker else 'disabled'}\n\n"
            + self._c("Write naturally. Examples:", Ansi.GREEN, bold=True) + "\n"
            + "  Remember that generated apps must include README and tests.\n"
            + "  Create a Python program to solve quadratic equations with README and tests\n"
            + "  What do you remember about project rules?\n"
            + self._c("Type help for commands, exit to quit.", Ansi.GRAY)
        )

    def help_text(self) -> str:
        return (
            self._c("Available natural actions:", Ansi.CYAN, bold=True) + "\n"
            + self._c("- Normal chat:", Ansi.GREEN, bold=True) + " ask anything normally.\n"
            + self._c("- Coding:", Ansi.GREEN, bold=True) + " 'Create a Python program ... with README and tests'.\n"
            + self._c("- Memory store:", Ansi.MAGENTA, bold=True) + " 'Remember that ...' or 'Project rule: ...'.\n"
            + self._c("- Memory recall:", Ansi.MAGENTA, bold=True) + " 'What do you remember about ...?'.\n"
            + self._c("- Forget:", Ansi.YELLOW, bold=True) + " 'Forget ...'.\n"
            + self._c("- Consolidate:", Ansi.BLUE, bold=True) + " 'consolidate memory'.\n"
            + self._c("- Consolidation status:", Ansi.BLUE, bold=True) + " 'consolidation status'.\n\n"
            + self._c("Optional explicit commands still work:", Ansi.GRAY, bold=True) + "\n"
            + "/task, /code, /store, /ask, /forget, /consolidate, /consolidation-status, /stats, /workspace, /projects, find project ..., project <id>, /staging, /clear-workspace, /clear-staging, /exit."
        )

    def process(self, message: str) -> str:
        action = route_message(message)
        kind = action.kind
        if kind == "empty":
            return ""
        if kind == "exit":
            self._safe_save()
            return self._c("Bye. Titan memory saved.", Ansi.GREEN, bold=True)
        if kind == "help":
            return self.help_text()
        if kind == "stats":
            return self._format_stats()
        if kind == "workspace":
            return self._c("Workspace files:", Ansi.CYAN, bold=True) + "\n" + self._format_file_list(list_workspace_files())
        if kind == "projects":
            return self._c("Generated project archives:", Ansi.CYAN, bold=True) + "\n" + list_generated_projects()
        if kind == "project_search":
            return self._c("Generated project search:", Ansi.CYAN, bold=True) + "\n" + format_project_search_results(action.payload)
        if kind == "project_detail":
            return self._c("Generated project detail:", Ansi.CYAN, bold=True) + "\n" + format_project_detail(action.payload)
        if kind == "staging":
            return self._c("Staging files:", Ansi.CYAN, bold=True) + "\n" + self._format_file_list(list_staging_files())
        if kind == "clear_workspace":
            return clear_workspace()
        if kind == "clear_staging":
            return clear_staging()
        if kind == "consolidate":
            return self._consolidate()
        if kind == "consolidation_status":
            return self._consolidation_status()
        if kind == "store":
            return self._store(action.payload)
        if kind == "recall":
            return self._recall(action.payload)
        if kind == "forget":
            return self._forget(action.payload)
        if kind == "task":
            return self._run_task(action.payload)
        return self._chat(action.payload)

    def _build_consolidation_policy(self) -> ConsolidationPolicy:
        """Build the startup consolidation policy from chat configuration."""

        log_dir = Path(self.config.consolidation_log_dir)
        return ConsolidationPolicy(
            enabled=bool(self.config.auto_consolidation_enabled),
            schedule_hour=int(self.config.consolidation_hour),
            schedule_minute=int(self.config.consolidation_minute),
            min_hours_between_runs=float(self.config.consolidation_min_hours_between_runs),
            steps=int(self.config.consolidation_steps),
            max_items=self.config.consolidation_max_items,
            log_dir=log_dir,
            state_path=log_dir / "consolidation_state.json",
            log_path=log_dir / "consolidation_log.jsonl",
            keep_existing_ltm=True,
            include_inactive=False,
            save_after=True,
        )

    def _run_startup_consolidation(self) -> ConsolidationRunResult | None:
        """Run the scheduler once at startup without forcing repeated consolidation."""

        if not self.config.use_titan_memory:
            return None
        try:
            policy = self._build_consolidation_policy()
            scheduler = ConsolidationScheduler(memory=self.memory, policy=policy)
            if self.config.force_startup_consolidation:
                return scheduler.run_once(reason="startup_forced", force=True)
            return scheduler.run_if_due()
        except Exception as exc:  # pragma: no cover - defensive startup path
            return ConsolidationRunResult(
                ran=False,
                status="error",
                reason="startup",
                message="Automatic consolidation check failed during startup.",
                run_at="startup",
                policy={},
                error=str(exc),
            )

    def _format_startup_consolidation_inline(self) -> str:
        result = self.startup_consolidation_result
        if not self.config.use_titan_memory:
            return "disabled because Titan memory is disabled"
        if not self.config.auto_consolidation_enabled:
            return "disabled"
        if result is None:
            return "not checked"
        if result.status == "ok" and result.ran:
            return "completed"
        if result.status == "skipped":
            return f"skipped ({result.reason})"
        if result.status == "error":
            return "error"
        return f"{result.status}; ran={result.ran}"

    def _consolidation_status(self) -> str:
        result = self.startup_consolidation_result
        if not self.config.use_titan_memory:
            return self._c("Titan memory is disabled.", Ansi.YELLOW, bold=True)
        if not self.config.auto_consolidation_enabled:
            return self._c("Automatic consolidation is disabled.", Ansi.YELLOW, bold=True)
        if result is None:
            return self._c("Automatic consolidation has not been checked yet.", Ansi.YELLOW, bold=True)
        lines = [self._c("Nightly consolidation status:", Ansi.BLUE, bold=True)]
        lines.append(f"- status: {result.status}")
        lines.append(f"- ran: {result.ran}")
        lines.append(f"- reason: {result.reason}")
        lines.append(f"- message: {result.message}")
        if result.log_path:
            lines.append(f"- log_path: {result.log_path}")
        if result.state_path:
            lines.append(f"- state_path: {result.state_path}")
        if result.error:
            lines.append(f"- error: {result.error}")
        return "\n".join(lines)

    def _safe_save(self) -> None:
        try:
            self.memory.save(self.config.memory_path)
        except Exception:
            pass

    def _store(self, text: str) -> str:
        if not self.config.use_titan_memory:
            return self._c("Titan memory is disabled.", Ansi.YELLOW, bold=True)
        records = self.memory.store(text, metadata={"source": "unified_chat"})
        self._safe_save()
        if not records:
            return self._c("Nothing was stored.", Ansi.YELLOW, bold=True)
        lines = [self._c("Saved in Titan memory:", Ansi.GREEN, bold=True)]
        lines.extend(f"- #{record.id}: {record.text}" for record in records)
        return "\n".join(lines)

    def _recall(self, query: str) -> str:
        if not self.config.use_titan_memory:
            return self._c("Titan memory is disabled.", Ansi.YELLOW, bold=True)
        records = self.memory.recall(query, top_k=self.config.memory_top_k, min_score=self.config.memory_min_score)
        if not records:
            return self._c("No relevant Titan memory found.", Ansi.YELLOW, bold=True)
        lines = [self._c("Relevant Titan memories:", Ansi.MAGENTA, bold=True)]
        for record in records:
            subject = f"subject={record.subject}" if record.subject else "subject=?"
            prop = f"property={record.property}" if record.property else "property=?"
            lines.append(f"- #{record.id} score={record.score:.3f} {subject} {prop}: {record.text}")
        return "\n".join(lines)

    def _forget(self, query: str) -> str:
        if not self.config.use_titan_memory:
            return self._c("Titan memory is disabled.", Ansi.YELLOW, bold=True)
        records = self.memory.forget(query)
        self._safe_save()
        if not records:
            return self._c("No safe matching memory found. Nothing was forgotten.", Ansi.YELLOW, bold=True)
        return self._c("Forgotten/deactivated:", Ansi.YELLOW, bold=True) + "\n" + "\n".join(f"- #{record.id}: {record.text}" for record in records)

    def _consolidate(self) -> str:
        if not self.config.use_titan_memory:
            return self._c("Titan memory is disabled.", Ansi.YELLOW, bold=True)
        result = self.memory.consolidate(keep_existing_ltm=False, steps=3, include_inactive=False)
        self._safe_save()
        lines = [self._c("Titan memory consolidated.", Ansi.GREEN, bold=True)]
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
        return self._colorize_bmad_output(getattr(result, "final_message", str(result)))

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


    def _colorize_bmad_output(self, text: str) -> str:
        """Add readable colors to BMAD summaries without changing their content."""

        if not self.config.color_enabled:
            return text
        colored: list[str] = []
        for line in text.splitlines():
            lower = line.lower()
            if "bmad coding workflow result: approved" in lower:
                colored.append(self._c(line, Ansi.GREEN, bold=True))
            elif "bmad coding workflow result: rejected" in lower or "failed" in lower:
                colored.append(self._c(line, Ansi.RED, bold=True))
            elif line.startswith("Task:") or line.startswith("Published:") or line.startswith("Workspace files:"):
                colored.append(self._c(line, Ansi.CYAN, bold=True))
            elif line.startswith("QA checks:") or line.startswith("Manual test commands:"):
                colored.append(self._c(line, Ansi.YELLOW, bold=True))
            elif line.startswith("Memory candidate") or line.startswith("Memory validation"):
                colored.append(self._c(line, Ansi.MAGENTA, bold=True))
            elif line.startswith("- ") and "passed" in lower:
                colored.append(self._c(line, Ansi.GREEN))
            elif line.startswith("- ") and ("failed" in lower or "error" in lower):
                colored.append(self._c(line, Ansi.RED))
            else:
                colored.append(line)
        return "\n".join(colored)

    def user_prompt(self) -> str:
        return self._c("You>", Ansi.CYAN, bold=True) + " "

    def ai_prefix(self) -> str:
        return self._c("AI>", Ansi.GREEN, bold=True) + " "

    def _format_stats(self) -> str:
        if not self.config.use_titan_memory:
            return self._c("Titan memory is disabled.", Ansi.YELLOW, bold=True)
        stats = self.memory.stats()
        return self._c("Titan memory stats:", Ansi.CYAN, bold=True) + "\n" + "\n".join(f"- {key}: {value}" for key, value in sorted(stats.items()))

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
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors in the terminal interface.")
    parser.add_argument("--disable-auto-consolidation", action="store_true", help="Disable the startup nightly consolidation check.")
    parser.add_argument("--consolidation-hour", type=int, default=2, help="Nightly consolidation hour, 0-23. Default: 2.")
    parser.add_argument("--consolidation-minute", type=int, default=0, help="Nightly consolidation minute. Default: 0.")
    parser.add_argument("--consolidation-steps", type=int, default=3, help="Number of replay steps for scheduled consolidation.")
    parser.add_argument("--consolidation-max-items", type=int, default=None, help="Maximum memories replayed by scheduled consolidation.")
    parser.add_argument("--consolidation-min-hours", type=float, default=20.0, help="Minimum hours between scheduled consolidation runs.")
    parser.add_argument("--consolidation-log-dir", default=str(ROOT / "logs"), help="Directory for consolidation state and JSONL logs.")
    parser.add_argument("--force-startup-consolidation", action="store_true", help="Force one consolidation pass when main.py starts. Useful for testing only.")
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
        color_enabled=not bool(args.no_color),
        auto_consolidation_enabled=not bool(args.disable_auto_consolidation),
        consolidation_hour=int(args.consolidation_hour),
        consolidation_minute=int(args.consolidation_minute),
        consolidation_steps=int(args.consolidation_steps),
        consolidation_max_items=args.consolidation_max_items,
        consolidation_min_hours_between_runs=float(args.consolidation_min_hours),
        consolidation_log_dir=args.consolidation_log_dir,
        force_startup_consolidation=bool(args.force_startup_consolidation),
    )


def run_interactive_chat(config: UnifiedChatConfig) -> int:
    assistant = UnifiedChatAssistant(config)
    print(assistant.welcome())
    while True:
        try:
            message = input("\n" + assistant.user_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n" + assistant.process("exit"))
            return 0
        if not message:
            continue
        action = route_message(message)
        response = assistant.process(message)
        if response:
            print("\n" + assistant.ai_prefix() + response)
        if action.kind == "exit":
            return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_interactive_chat(config_from_args(args))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
