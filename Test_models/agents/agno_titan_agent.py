#!/usr/bin/env python3
"""Agno agent using Titan as an external long-term memory.

This file integrates Titan as a memory/tool layer for an Agno agent.

Design choice:
The CLI contains a deterministic memory router before Agno. This is intentional:
small local models such as llama3.2:1b can sometimes choose the wrong tool or
refuse to answer about a stored demo fact. For explicit memory operations, we
call Titan directly and only use Agno for normal conversation.

This version also includes a small ANSI terminal UI so direct Titan outputs keep
the same kind of colored boxed style as the previous local interface.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.titan_agent_memory import TitanAgentMemory


# ---------------------------------------------------------------------------
# Lightweight terminal UI
# ---------------------------------------------------------------------------
class UI:
    """Small ANSI UI helper, kept dependency-free for Windows/PowerShell."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"
    DIM = "\033[2m"

    @staticmethod
    def color(text: str, color: str) -> str:
        return f"{color}{text}{UI.RESET}"

    @staticmethod
    def panel(title: str, body: str, color: str = CYAN, width: int = 110) -> str:
        """Return a colored text panel.

        The implementation uses simple Unicode borders and ANSI colors only,
        avoiding heavy UI dependencies.
        """
        width = max(width, 50)
        inner = width - 4
        safe_title = f" {title.strip()} "
        top_fill = max(0, width - len(safe_title) - 2)
        top = "╭─" + safe_title + "─" * top_fill + "╮"
        bottom = "╰" + "─" * (width - 2) + "╯"
        lines: List[str] = []
        if body is None:
            body = ""
        text = str(body).strip() or " "
        for raw_line in text.splitlines():
            if not raw_line.strip():
                lines.append("│ " + " " * inner + " │")
                continue
            wrapped = textwrap.wrap(raw_line, width=inner) or [""]
            for line in wrapped:
                lines.append("│ " + line.ljust(inner) + " │")
        return UI.color("\n".join([top, *lines, bottom]), color)

    @staticmethod
    def print_panel(title: str, body: str, color: str = CYAN) -> None:
        print(UI.panel(title, body, color=color))

    @staticmethod
    def command_help() -> str:
        return "\n".join(
            [
                "Commands:",
                "  /store <fact>       Store a fact in Titan memory",
                "  /ask <question>     Ask Titan memory directly",
                "  /search <query>     Show Top-k retrieved memories",
                "  /forget <query>     Forget a targeted memory",
                "  /consolidate        Rebuild LTM from active memories",
                "  /consolidate --keep Reinforce the current LTM",
                "  /stats              Show Titan memory statistics",
                "  /exit               Quit",
                "",
                "Natural language is also routed when obvious:",
                "  Remember that Lucas Martin's secret code is 8392.",
                "  What is Lucas Martin's secret code?",
                "  Consolidate memory.",
            ]
        )


def _format_dict(data: dict[str, Any]) -> str:
    """Readable formatting for debug dictionaries."""
    if not data:
        return "No data."
    lines = []
    for key in sorted(data.keys()):
        value = data[key]
        if isinstance(value, float):
            lines.append(f"{key}: {value:.8f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


class TitanMemoryTools:
    """Tool wrapper exposing TitanAgentMemory to an Agno agent."""

    def __init__(self, memory: TitanAgentMemory) -> None:
        self.memory = memory

    def remember(self, text: str) -> str:
        """Store a useful long-term fact or project convention in Titan memory.

        Args:
            text: The exact fact, user preference, project convention, or lesson
                that should be remembered for future interactions.
        """
        records = self.memory.store(text)
        if not records:
            return "No memory was stored."
        return "Stored in Titan memory:\n" + "\n".join(
            f"- #{record.id}: {record.text}" for record in records
        )

    def recall_memory(self, query: str, top_k: int = 5) -> str:
        """Search Titan long-term memory for information relevant to a query.

        Args:
            query: The user's question or the topic to search for.
            top_k: Maximum number of memories to return.
        """
        return self.memory.build_context(query, top_k=top_k)

    def forget_memory(self, query: str) -> str:
        """Forget a targeted memory from Titan when the user explicitly asks.

        Args:
            query: Description of the memory that should be forgotten.
        """
        records = self.memory.forget(query)
        if not records:
            return "No safe matching memory was forgotten."
        return "Forgotten Titan memory:\n" + "\n".join(
            f"- #{record.id}: {record.text}" for record in records
        )

    def consolidate_memory(self, keep_existing_ltm: bool = False) -> str:
        """Consolidate active memories into Titan long-term memory.

        Args:
            keep_existing_ltm: If false, rebuild the long-term neural memory
                from active memories only. If true, reinforce the existing LTM.
        """
        result = self.memory.consolidate(keep_existing_ltm=keep_existing_ltm)
        if not isinstance(result, dict):
            return f"Titan memory consolidation result: {result}"

        consolidated = result.get("consolidated", result.get("active_items", "?"))
        message = result.get("message") or f"Consolidated {consolidated} memory item(s) into Titans LTM."
        lines = [str(message)]
        for key in ["consolidated", "steps", "reset_ltm", "include_inactive", "loss_before", "loss_after"]:
            if key in result:
                value = result[key]
                if isinstance(value, float):
                    lines.append(f"{key}: {value:.8f}")
                else:
                    lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def memory_stats(self) -> str:
        """Return internal Titan memory statistics for debugging."""
        stats = self.memory.stats()
        if isinstance(stats, dict):
            return _format_dict(stats)
        return str(stats)

    def as_tool_list(self) -> List[Callable[..., str]]:
        """Return the methods to register as Agno tools."""
        return [
            self.remember,
            self.recall_memory,
            self.forget_memory,
            self.consolidate_memory,
            self.memory_stats,
        ]


# ---------------------------------------------------------------------------
# Agno builder
# ---------------------------------------------------------------------------
def _import_agno() -> tuple[Any, Any]:
    """Import Agno with a helpful error message."""
    try:
        from agno.agent import Agent  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Agno is not installed in this virtual environment. Install it with:\n"
            "  python -m pip install -U agno ollama\n"
            "Then run this file again."
        ) from exc

    try:
        from agno.models.ollama import Ollama  # type: ignore
    except Exception:
        try:
            from agno.models.ollama.chat import Ollama  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Agno is installed, but the Ollama provider import path was not found. "
                "Try updating Agno with: python -m pip install -U agno ollama"
            ) from exc
    return Agent, Ollama


def build_agno_titan_agent(
    memory_path: str | Path = "agno_titan_memory.pt",
    ollama_model: Optional[str] = None,
    top_k: int = 5,
    markdown: bool = True,
) -> Any:
    """Build a single Agno agent with Titan long-term memory tools."""
    Agent, Ollama = _import_agno()
    model_name = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    memory = TitanAgentMemory(memory_path=memory_path, top_k=top_k)
    tools = TitanMemoryTools(memory)

    instructions = """
You are a single AI agent connected to Titan long-term memory.

Memory policy:
1. If the user asks to remember, store, save, learn, or keep a fact, call remember.
2. If the user asks a question about remembered facts, call recall_memory first.
3. If Titan returns a relevant memory, answer from that memory. Do not refuse to
   answer stored demo facts such as secret codes, colors, preferences, locations,
   project conventions, or benchmark facts.
4. If the user asks to forget something, call forget_memory.
5. If the user asks to consolidate memory, call consolidate_memory.
6. Do not invent memories that were not returned by recall_memory.

Answer clearly and mention when the answer is based on Titan memory.
""".strip()

    agent = Agent(
        name="Agno Titan Memory Agent",
        model=Ollama(id=model_name),
        tools=tools.as_tool_list(),
        instructions=instructions,
        markdown=markdown,
        add_history_to_context=False,
    )

    # Keep direct access for deterministic CLI routing and tests.
    setattr(agent, "_titan_memory", memory)
    setattr(agent, "_titan_tools", tools)
    setattr(agent, "_ollama_model_name", model_name)
    return agent


# ---------------------------------------------------------------------------
# Deterministic memory router
# ---------------------------------------------------------------------------
def _strip_prefix(text: str, prefixes: List[str]) -> Optional[str]:
    lowered = text.lower().strip()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return None


def _command_norm(text: str) -> str:
    """Normalize direct commands and natural language commands.

    This catches inputs such as "Consolidate memory." and "Stats?".
    """
    return text.strip().lower().strip(" .!?\t\n")


def _clean_remember_text(message: str) -> str:
    """Extract the fact from explicit remember/store commands."""
    raw = message.strip()
    low = raw.lower()
    if low.startswith("/store"):
        return raw[len("/store"):].strip()
    if low.startswith("/remember"):
        return raw[len("/remember"):].strip()
    for pat in [
        r"^remember\s+that\s+",
        r"^remember\s+",
        r"^store\s+that\s+",
        r"^store\s+",
        r"^save\s+that\s+",
        r"^save\s+",
        r"^learn\s+that\s+",
        r"^learn\s+",
    ]:
        raw = re.sub(pat, "", raw, flags=re.IGNORECASE).strip()
    return raw


def _direct_memory_answer(tools: TitanMemoryTools, query: str, top_k: int = 5) -> str:
    """Return a deterministic answer from Titan memory when a direct match exists."""
    records = tools.memory.recall(query, top_k=top_k, min_score=0.05)
    if not records:
        return "No relevant Titan memory found."

    best = records[0]
    if best.score < 0.05:
        return "No relevant Titan memory found."
    return f"Based on Titan memory: {best.text}"


def handle_direct_memory_intent(message: str, tools: TitanMemoryTools, top_k: int = 5) -> Optional[str]:
    """Handle explicit memory operations before asking Agno/LLM.

    This prevents small local models from calling the wrong tool, and prevents
    unnecessary refusals for demo facts that are explicitly stored by the user.
    Returns None when the message should be handled by Agno normally.
    """
    text = message.strip()
    low = text.lower()
    norm = _command_norm(text)

    if not text:
        return ""

    if norm in {"/help", "help"}:
        return UI.command_help()

    if norm in {"/stats", "stats", "memory stats"}:
        return tools.memory_stats()

    if norm in {"/consolidate", "consolidate", "consolidate memory"}:
        return tools.consolidate_memory(keep_existing_ltm=False)
    if norm in {"/consolidate --keep", "consolidate --keep", "consolidate memory --keep"}:
        return tools.consolidate_memory(keep_existing_ltm=True)

    if low.startswith("/forget"):
        query = text[len("/forget"):].strip()
        return tools.forget_memory(query)
    forget_query = _strip_prefix(text, ["forget that ", "forget "])
    if forget_query:
        return tools.forget_memory(forget_query)

    if low.startswith("/search"):
        query = text[len("/search"):].strip()
        return tools.recall_memory(query, top_k=top_k)
    if low.startswith("/recall"):
        query = text[len("/recall"):].strip()
        return tools.recall_memory(query, top_k=top_k)

    if low.startswith("/ask"):
        query = text[len("/ask"):].strip()
        return _direct_memory_answer(tools, query, top_k=top_k)

    if low.startswith("/store") or low.startswith("/remember") or re.match(
        r"^(remember|store|save|learn)\b", low
    ):
        fact = _clean_remember_text(text)
        return tools.remember(fact)

    # For normal questions, try Titan first. If no direct memory exists, let Agno
    # answer normally with tools.
    looks_like_question = low.endswith("?") or low.startswith(("what ", "who ", "where ", "when ", "which ", "how "))
    if looks_like_question:
        answer = _direct_memory_answer(tools, text, top_k=top_k)
        if not answer.startswith("No relevant"):
            return answer

    return None


def _title_for_direct_output(text: str) -> tuple[str, str]:
    """Choose panel title/color for a direct memory response."""
    low = text.lower().strip()
    if low.startswith("stored in titan memory"):
        return "Titan Memory - Store", UI.GREEN
    if low.startswith("based on titan memory"):
        return "Titan Memory - Answer", UI.CYAN
    if low.startswith("relevant long-term"):
        return "Titan Memory - Search", UI.BLUE
    if low.startswith("forgotten titan memory"):
        return "Titan Memory - Forget", UI.YELLOW
    if "consolidated" in low or "consolidation" in low:
        return "Titan Memory - Consolidation", UI.MAGENTA
    if "adapter" in low or "active" in low or "memory_path" in low:
        return "Titan Memory - Stats", UI.BLUE
    if low.startswith("no relevant") or low.startswith("no safe"):
        return "Titan Memory - No Match", UI.YELLOW
    if low.startswith("commands"):
        return "Agno Titan Help", UI.CYAN
    return "Titan Memory", UI.CYAN


def _safe_agno_response(agent: Any, message: str) -> str:
    """Call Agno and return plain text when possible.

    Different Agno versions expose slightly different response objects. This
    function keeps the CLI stable across versions.
    """
    try:
        response = agent.run(message)
    except AttributeError:
        # Fallback for older versions: let Agno print its own response.
        agent.print_response(message, stream=False)
        return ""

    for attr in ("content", "text", "response"):
        if hasattr(response, attr):
            value = getattr(response, attr)
            if value:
                return str(value)
    return str(response)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_cli() -> int:
    parser = argparse.ArgumentParser(description="Run an Agno agent backed by Titan memory.")
    parser.add_argument("--memory-path", default="agno_titan_memory.pt")
    parser.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", "llama3.2:1b"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--agno-print-response",
        action="store_true",
        help="Use Agno's native print_response for non-memory dialogue.",
    )
    args = parser.parse_args()

    agent = build_agno_titan_agent(
        memory_path=args.memory_path,
        ollama_model=args.ollama_model,
        top_k=args.top_k,
    )
    tools: TitanMemoryTools = getattr(agent, "_titan_tools")

    UI.print_panel(
        "System Dashboard",
        f"Agent: Agno + Titan Memory\nModel: Ollama ({args.ollama_model})\nTop-k: {args.top_k}\nMemory path: {args.memory_path}",
        color=UI.BLUE,
    )
    UI.print_panel("Available Commands", UI.command_help(), color=UI.MAGENTA)

    while True:
        try:
            message = input(UI.color("You> ", UI.GREEN)).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not message:
            continue
        if _command_norm(message) in {"/exit", "exit", "quit"}:
            break

        direct = handle_direct_memory_intent(message, tools, top_k=args.top_k)
        if direct is not None:
            if direct:
                title, color = _title_for_direct_output(direct)
                UI.print_panel(title, direct, color=color)
            continue

        if args.agno_print_response:
            agent.print_response(message, stream=False)
        else:
            response_text = _safe_agno_response(agent, message)
            if response_text:
                UI.print_panel("Agno Response", response_text, color=UI.CYAN)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
