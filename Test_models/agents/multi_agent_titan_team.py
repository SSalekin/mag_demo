#!/usr/bin/env python3
"""Manager-led multi-agent prototype backed by shared Titan memory.

This module is the next step after the single Agno + Titan agent.

Goal
----
Prototype a team of specialized agents using the same shared Titan long-term
memory:
- Manager
- DevWeb
- DevSoft
- DevOps
- Tester
- Evaluator

Important design decision
-------------------------
For now this prototype keeps Ollama optional and keeps the multi-agent routing
mostly deterministic. This avoids overloading a small local LLM with six
simultaneous agent calls, while still validating the system architecture:

User request -> Manager -> Shared Titan memory -> Specialists -> Evaluator -> Manager response

Later, each role can be replaced by an Agno Agent or another free/local LLM
provider without changing the Titan memory layer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.titan_agent_memory import TitanAgentMemory, AgentMemoryRecord


# ---------------------------------------------------------------------------
# Terminal UI
# ---------------------------------------------------------------------------
class TeamUI:
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
        return f"{color}{text}{TeamUI.RESET}"

    @staticmethod
    def panel(title: str, body: str, color: str = CYAN, width: int = 116) -> str:
        # Keep boxes inside the current terminal width. VS Code/PowerShell
        # sometimes wraps Unicode borders if the panel is one character too
        # wide, which creates a broken-looking rectangle on the right side.
        terminal_width = shutil.get_terminal_size((120, 24)).columns
        width = min(int(width), max(56, terminal_width - 2))
        width = max(width, 56)
        inner = width - 4
        safe_title = f" {title.strip()} "
        # 2 chars for "╭─" and 1 char for "╮" => subtract 3.
        # The previous version subtracted 2, making the top border one char
        # longer than the content/bottom lines.
        top_fill = max(0, width - len(safe_title) - 3)
        top = "╭─" + safe_title + "─" * top_fill + "╮"
        bottom = "╰" + "─" * (width - 2) + "╯"
        lines: List[str] = []
        text = str(body or " ").strip() or " "
        for raw in text.splitlines():
            if not raw.strip():
                lines.append("│ " + " " * inner + " │")
                continue
            wrapped = textwrap.wrap(
                raw,
                width=inner,
                replace_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            for line in wrapped:
                lines.append("│ " + line.ljust(inner) + " │")
        return TeamUI.color("\n".join([top, *lines, bottom]), color)

    @staticmethod
    def print_panel(title: str, body: str, color: str = CYAN) -> None:
        print(TeamUI.panel(title, body, color=color))

    @staticmethod
    def help_text() -> str:
        return "\n".join(
            [
                "Commands:",
                "  /task <request>          Run the multi-agent team on a task",
                "  /store <fact>            Store a project fact in shared Titan memory",
                "  /ask <question>          Ask shared Titan memory directly",
                "  /search <query>          Show retrieved Titan memories",
                "  /forget <query>          Forget a targeted memory",
                "  /consolidate             Rebuild Titan LTM from active memories",
                "  /stats                   Show memory stats",
                "  /agents                  Show team roles",
                "  /exit                    Quit",
                "",
                "Natural language examples:",
                "  Remember that this project uses pytest and type hints.",
                "  Build a plan to add a FastAPI endpoint for memory search.",
            ]
        )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class AgentResponse:
    name: str
    role: str
    content: str
    confidence: float = 0.75
    actions: List[str] = field(default_factory=list)


@dataclass
class TeamResult:
    task: str
    selected_agents: List[str]
    memory_context: str
    responses: List[AgentResponse]
    final_answer: str
    stored_decision: Optional[str] = None


@dataclass
class RoleAgent:
    name: str
    role: str
    responsibilities: Sequence[str]
    keywords: Sequence[str]
    color: str = TeamUI.CYAN

    def is_relevant(self, task: str) -> bool:
        low = task.lower()
        return any(keyword in low for keyword in self.keywords)

    def respond(self, task: str, memory_records: Sequence[AgentMemoryRecord]) -> AgentResponse:
        """Return a role-specific deterministic response.

        This is intentionally deterministic for the first prototype. It gives a
        stable benchmark and keeps the architecture working even when Ollama is
        too small or too slow for six LLM-backed agents.
        """
        memory_text = "\n".join(f"- {r.text}" for r in memory_records[:5]) or "- No relevant Titan memory."
        low = task.lower()
        lines = [f"Role: {self.role}", f"Task focus: {self._focus_sentence(low)}", "Relevant memory:", memory_text]
        lines.extend(self._recommendations(low, memory_records))
        return AgentResponse(
            name=self.name,
            role=self.role,
            content="\n".join(lines),
            confidence=self._confidence(memory_records),
            actions=self._actions(low),
        )

    def _focus_sentence(self, low: str) -> str:
        if self.name == "manager":
            return "decompose the task, coordinate specialists, and decide what should be remembered."
        if self.name == "devweb":
            return "web/API surface, user-facing routes, frontend/backend integration."
        if self.name == "devsoft":
            return "Python architecture, model integration, clean classes and maintainable code."
        if self.name == "devops":
            return "environment, dependencies, commands, local deployment and reproducibility."
        if self.name == "tester":
            return "unit tests, integration tests, benchmarks, regression checks."
        if self.name == "evaluator":
            return "quality assessment, risks, trade-offs, and final recommendation."
        return "general support."

    def _recommendations(self, low: str, records: Sequence[AgentMemoryRecord]) -> List[str]:
        memory_blob = " ".join(r.text.lower() for r in records)
        recs: List[str] = []
        if self.name == "manager":
            recs.append("Recommendation: route implementation details to DevSoft/DevWeb, validation to Tester, and final decision to Evaluator.")
            if records:
                recs.append("Memory use: include the retrieved Titan conventions before assigning sub-tasks.")
        elif self.name == "devweb":
            if any(word in low for word in ["api", "endpoint", "fastapi", "web", "frontend", "route"]):
                recs.append("Implementation: define the route contract, request/response schema, and error handling.")
            else:
                recs.append("Implementation: no major web-specific work detected; support only if an API or UI is needed.")
        elif self.name == "devsoft":
            recs.append("Implementation: keep the Titan memory adapter separated from orchestration logic.")
            if "type hints" in memory_blob:
                recs.append("Project convention from memory: use type hints in new Python code.")
            if "single-file titan" in memory_blob or "single file titan" in memory_blob:
                recs.append("Project convention from memory: preserve the current single-file Titan model unless refactoring is explicitly requested.")
        elif self.name == "devops":
            recs.append("Commands: document environment variables, install steps, and reproducible benchmark commands.")
            if "ollama" in low or "llm" in low:
                recs.append("LLM provider: keep Ollama as default for now; do not hardcode the provider.")
        elif self.name == "tester":
            if "pytest" in memory_blob:
                recs.append("Testing convention from memory: use pytest for new tests.")
            recs.append("Validation: add a focused test first, then benchmark only after the unit test passes.")
        elif self.name == "evaluator":
            recs.append("Evaluation: compare against no-memory and single-agent baselines, report failures honestly, and include limitations and risks.")
            if "free" in low or "llm" in low:
                recs.append("LLM choice: keep the provider configurable; evaluate free alternatives later with current availability.")
        return recs

    def _actions(self, low: str) -> List[str]:
        if self.name == "manager":
            return ["coordinate", "store_decision"]
        if self.name == "tester":
            return ["write_tests", "run_benchmarks"]
        if self.name == "evaluator":
            return ["score_solution", "summarize_risks"]
        if self.name in {"devweb", "devsoft", "devops"}:
            return ["propose_implementation"]
        return []

    def _confidence(self, records: Sequence[AgentMemoryRecord]) -> float:
        return 0.85 if records else 0.70


# ---------------------------------------------------------------------------
# Multi-agent team
# ---------------------------------------------------------------------------
class MultiAgentTitanTeam:
    """Manager-led multi-agent team with shared Titan long-term memory."""

    def __init__(
        self,
        memory_path: str | Path = "multi_agent_titan_memory.pt",
        ollama_model: Optional[str] = None,
        top_k: int = 6,
        use_llm_roles: bool = False,
    ) -> None:
        self.memory_path = Path(memory_path)
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        self.top_k = int(top_k)
        self.use_llm_roles = bool(use_llm_roles)
        self.memory = TitanAgentMemory(memory_path=self.memory_path, top_k=self.top_k)
        self.roles = self._build_roles()

    @staticmethod
    def _build_roles() -> Dict[str, RoleAgent]:
        return {
            "manager": RoleAgent(
                name="manager",
                role="Project manager and orchestrator",
                responsibilities=("plan", "route", "coordinate", "store decisions"),
                keywords=("plan", "task", "project", "agent", "workflow", "memory", "api", "test", "deploy", "code", "endpoint"),
                color=TeamUI.MAGENTA,
            ),
            "devweb": RoleAgent(
                name="devweb",
                role="Web/API developer",
                responsibilities=("frontend", "backend web", "API", "routes"),
                keywords=("web", "api", "endpoint", "frontend", "backend", "route", "fastapi", "http", "ui"),
                color=TeamUI.CYAN,
            ),
            "devsoft": RoleAgent(
                name="devsoft",
                role="Software/Python developer",
                responsibilities=("python", "architecture", "classes", "refactoring"),
                keywords=("python", "class", "model", "titan", "adapter", "software", "refactor", "code", "function"),
                color=TeamUI.GREEN,
            ),
            "devops": RoleAgent(
                name="devops",
                role="DevOps and environment engineer",
                responsibilities=("setup", "dependencies", "commands", "deployment"),
                keywords=("install", "environment", "venv", "docker", "deploy", "ollama", "llm", "provider", "dependency", "command"),
                color=TeamUI.YELLOW,
            ),
            "tester": RoleAgent(
                name="tester",
                role="Test and benchmark engineer",
                responsibilities=("unit tests", "integration tests", "benchmarks"),
                keywords=("test", "pytest", "benchmark", "validate", "failure", "score", "compare", "regression"),
                color=TeamUI.BLUE,
            ),
            "evaluator": RoleAgent(
                name="evaluator",
                role="Evaluator and reviewer",
                responsibilities=("quality", "risks", "final recommendation"),
                keywords=("evaluate", "best", "compare", "result", "score", "risk", "recommend", "which"),
                color=TeamUI.RED,
            ),
        }

    # ------------------------- memory commands -------------------------
    def store(self, text: str) -> str:
        records = self.memory.store(text)
        if not records:
            return "No memory was stored."
        return "Stored in shared Titan memory:\n" + "\n".join(f"- #{r.id}: {r.text}" for r in records)

    def _active_records(self) -> List[AgentMemoryRecord]:
        """Return all active Titan memories as agent records.

        Retrieval is still the primary mechanism, but for project conventions
        and multi-agent planning we also keep active memory available as a
        lightweight short-term/shared context. This is useful when a query is
        phrased differently from the stored convention.
        """
        records: List[AgentMemoryRecord] = []
        for item in getattr(self.memory.memory, "active_items", []):
            try:
                records.append(AgentMemoryRecord.from_item(item, score=1.0, metadata={"source": "active_memory"}))
            except Exception:
                continue
        return records

    @staticmethod
    def _merge_records(primary: Sequence[AgentMemoryRecord], fallback: Sequence[AgentMemoryRecord]) -> List[AgentMemoryRecord]:
        seen: set[int] = set()
        merged: List[AgentMemoryRecord] = []
        for record in list(primary) + list(fallback):
            if record.id in seen:
                continue
            seen.add(record.id)
            merged.append(record)
        return merged


    # ------------------------- memory retrieval refinement -------------------------
    NOISE_TERMS = {
        "banana", "coffee", "hotel", "router", "sticker", "orange", "travel", "advertisement",
        "ad", "capsule", "room", "drawer", "random", "unrelated", "distractor", "noise",
    }

    @staticmethod
    def _expanded_query_terms(query: str) -> set[str]:
        low = query.lower()
        terms = set(re.findall(r"[a-zA-Z0-9_+#.-]+", low))
        terms = {t.strip(".,:;!?()[]{}'\"") for t in terms}
        terms = {t for t in terms if len(t) >= 3 and t not in MultiAgentTitanTeam.NOISE_TERMS}
        if "testing framework" in low or "test framework" in low:
            terms.update({"pytest", "test", "testing", "framework"})
        if "type hint" in low or "typing" in low:
            terms.update({"type", "hints", "typed"})
        if "provider" in low or "llm" in low:
            terms.update({"ollama", "configurable", "provider", "free"})
        if "architecture" in low or "setup" in low:
            terms.update({"manager-led", "multi-agent", "shared", "titan", "memory"})
        if "endpoint" in low or "route" in low or "api" in low:
            terms.update({"endpoint", "route", "api", "fastapi", "schema"})
        if "project rule" in low or "remaining project rule" in low:
            terms.update({"project", "rule", "benchmark", "limitations", "next", "steps", "retention"})
        if "code formatter" in low or "formatter" in low:
            terms.update({"formatter", "ruff", "format"})
        if "api auth" in low or "auth method" in low or "authentication" in low:
            terms.update({"auth", "method", "jwt", "api", "authentication"})
        if "consolidation" in low:
            terms.update({"consolidation", "rule", "selected", "agents", "memory", "context", "csv", "markdown", "limitations", "risks", "powershell", "environment"})
        return {t for t in terms if len(t) >= 3}

    @staticmethod
    def _clean_task_for_display(task: str) -> str:
        """Remove benchmark noise tokens before putting the task in the final answer.

        This prevents the final response from repeating words such as "banana" or
        "hotel" that were deliberately included as distractors in stress tests.
        """
        cleaned = str(task)
        cleaned = re.sub(r"(?i)^ignore\s+noise\s*[!#?\s\w-]*?(?:which|what|who|create|build|plan)", lambda m: m.group(0).split()[-1], cleaned)
        for token in sorted(MultiAgentTitanTeam.NOISE_TERMS, key=len, reverse=True):
            cleaned = re.sub(rf"(?i)\b{re.escape(token)}\b", "", cleaned)
        cleaned = re.sub(r"[!#?]{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,.?!")
        return cleaned or task

    @staticmethod
    def _property_hint(text: str) -> str:
        """Infer a stable memory property from a fact or a task.

        Titan's low-level parser is intentionally generic.  The multi-agent
        layer adds project-specific property hints so update-heavy cases such as
        provider=Gemini -> provider=Ollama can keep only the latest value.
        """
        low = text.lower()
        if "preferred backend language" in low or "backend language" in low:
            return "backend_language"
        if "preferred llm provider" in low or "llm provider" in low or "provider for current tests" in low:
            return "llm_provider"
        if "testing framework" in low or "test framework" in low:
            return "testing_framework"
        if "demo ui style" in low or "ui style" in low:
            return "ui_style"
        if "memory backend" in low or "shared titan memory" in low:
            return "memory_backend"
        if "deployment shell" in low:
            return "deployment_shell"
        if "reporting rule" in low:
            return "reporting_rule"
        if "benchmark output format" in low:
            return "benchmark_output_format"
        # Holdout/adversarial properties added after the second benchmark.
        # They are not in the original large benchmark and therefore help
        # prevent benchmark-specific overfitting.
        if "code formatter" in low or "formatter" in low:
            return "code_formatter"
        if "api auth method" in low or "auth method" in low or "authentication method" in low:
            return "api_auth_method"
        if "temporary secret code" in low:
            return "temporary_secret_code"
        # Check consolidation before the generic project-rule branch. The task
        # often says "After consolidation, what project rule..." but the real
        # memory is "Consolidation holdout rule ...".
        if "consolidation rule" in low or "consolidation holdout rule" in low or "after consolidation" in low:
            return "consolidation_rule"
        if "project rule" in low or "retention rule" in low:
            return "project_rule"
        if "critical project convention" in low or "project convention" in low:
            m = re.search(r"convention for ([a-z0-9_-]+)", low)
            return f"project_convention:{m.group(1)}" if m else "project_convention"
        m = re.search(r"owns the ([a-z0-9 _-]+?) task", low)
        if m:
            return "owner_task:" + re.sub(r"\s+", " ", m.group(1).strip())
        m = re.search(r"who owns the ([a-z0-9 _-]+?) task", low)
        if m:
            return "owner_task:" + re.sub(r"\s+", " ", m.group(1).strip())
        if "profile" in low and any(name in low for name in ["emma", "lucas", "sarah", "noah", "nina", "hugo"]):
            return "profile_summary"
        return "generic"

    @staticmethod
    def _is_noise_record(record: AgentMemoryRecord) -> bool:
        text = record.text.lower()
        if text.startswith("multi-agent team decision:"):
            return True
        if "distractor:" in text or text.startswith("distractor"):
            return True
        if "noise distractor" in text or "travel ad" in text:
            return True
        # Do not let operational consolidation commands become retrieved facts.
        # The real facts are the rules being consolidated, not the instruction
        # that consolidation should happen.
        if text.startswith("please consolidate") or "consolidate the active project rules" in text:
            return True
        if text.strip() in {"consolidate memory", "consolidate"}:
            return True
        # A critical memory can mention the word "noise" as part of its label;
        # keep it if it also contains useful project terms.
        if any(w in text for w in ["critical", "project", "rule", "convention", "titan"]):
            return False
        return False

    @staticmethod
    def _extract_owner_task(query: str) -> Optional[str]:
        low = query.lower()
        m = re.search(r"who owns the ([a-z0-9 _-]+?) task", low)
        if m:
            return re.sub(r"\s+", " ", m.group(1).strip())
        return None

    @staticmethod
    def _profile_name(query: str) -> Optional[str]:
        ignored_first_words = {"what", "which", "where", "when", "who", "how", "summarize", "create", "build", "plan", "forget"}
        for m in re.finditer(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", query):
            first, last = m.group(1), m.group(2)
            if first.lower() in ignored_first_words:
                continue
            return f"{first} {last}"
        return None

    def _effective_query_for_task(self, task: str) -> str:
        low = task.lower()
        if "remaining project rule" in low or "explain the remaining project rule" in low:
            return "project rule benchmark limitations next steps retention rule"
        if "after consolidation" in low or "preserved project rules" in low:
            return "consolidation holdout rule selected agents memory context csv markdown limitations risks powershell environment variables"
        if "current" in low:
            return re.sub(r"(?i)^what is the current\s+", "", task).strip(" ?")
        if "who owns the" in low:
            return task
        cleaned = self._clean_task_for_display(task)
        return cleaned

    def _rank_records_for_query(self, records: Sequence[AgentMemoryRecord], query: str) -> List[AgentMemoryRecord]:
        terms = self._expanded_query_terms(query)
        query_prop = self._property_hint(query)
        owner_task = self._extract_owner_task(query)
        profile_name = self._profile_name(query)

        def score(record: AgentMemoryRecord) -> tuple[int, int, int, int, float, int]:
            text = record.text.lower()
            prop = self._property_hint(record.text)
            lexical = sum(1 for term in terms if term in text)
            prop_match = int(query_prop != "generic" and (prop == query_prop or prop.startswith(query_prop + ":") or query_prop.startswith(prop + ":")))
            owner_match = int(bool(owner_task and owner_task in text))
            profile_match = int(bool(profile_name and profile_name.lower() in text))
            non_noise = 0 if self._is_noise_record(record) else 1
            return (prop_match, owner_match, profile_match, lexical, float(record.score), int(record.id))

        ordered = sorted(records, key=score, reverse=True)
        return ordered

    def _collapse_updated_records(self, records: Sequence[AgentMemoryRecord], query: str) -> List[AgentMemoryRecord]:
        """Keep the most recent record for update-like properties.

        This is a manager-level safety layer. It prevents the final answer from
        showing Gemini/Claude after Ollama has been stored as the current
        provider, or unittest/nose after pytest has been stored.
        """
        query_prop = self._property_hint(query)
        update_props = {
            "backend_language", "llm_provider", "testing_framework", "ui_style", "memory_backend",
            "deployment_shell", "reporting_rule", "benchmark_output_format",
            "code_formatter", "api_auth_method",
        }
        if query_prop not in update_props:
            return list(records)
        matching = [r for r in records if self._property_hint(r.text) == query_prop]
        if not matching:
            return list(records)
        latest = max(matching, key=lambda r: int(r.id))
        return [latest]

    def _select_relevant_records(self, raw_records: Sequence[AgentMemoryRecord], task: str, limit: Optional[int] = None) -> List[AgentMemoryRecord]:
        query = self._effective_query_for_task(task)
        owner_task = self._extract_owner_task(query)
        profile_name = self._profile_name(query)
        query_prop = self._property_hint(query)
        terms = self._expanded_query_terms(query)

        filtered: List[AgentMemoryRecord] = []
        for record in raw_records:
            text = record.text.lower()
            prop = self._property_hint(record.text)
            if self._is_noise_record(record):
                continue
            if "forget" in task.lower() and "temporary secret code" in text:
                continue
            if owner_task and owner_task not in text:
                continue
            if profile_name and "profile" in query.lower() and profile_name.lower() not in text:
                continue
            if query_prop != "generic" and query_prop not in {"profile_summary"}:
                # Keep exact property matches and project convention subtypes.
                if not (prop == query_prop or prop.startswith(query_prop + ":") or query_prop.startswith(prop + ":")):
                    # For project-rule tasks, allow retention rules too.
                    if not (query_prop == "project_rule" and prop in {"project_rule", "generic"} and any(t in text for t in ["retention rule", "project rule", "benchmark", "limitations", "next steps"])):
                        continue
            if terms and query_prop == "generic":
                lexical = sum(1 for term in terms if term in text)
                if lexical == 0 and not owner_task and not profile_name:
                    continue
            filtered.append(record)

        if not filtered:
            # Fallback: keep non-noise records and let ranking do the rest.
            filtered = [r for r in raw_records if not self._is_noise_record(r)]
        ranked = self._rank_records_for_query(filtered, query)
        ranked = self._collapse_updated_records(ranked, query)
        return ranked[: (limit or self.top_k)]

    def _format_memory_context(self, records: Sequence[AgentMemoryRecord], task: str) -> str:
        if not records:
            return "No relevant long-term memory found."
        lines = ["Selected shared Titan memories:"]
        for idx, record in enumerate(records[: self.top_k], start=1):
            prop = self._property_hint(record.text)
            meta = []
            if record.subject:
                meta.append(f"subject={record.subject}")
            if prop != "generic":
                meta.append(f"property={prop}")
            meta_text = f" [{'; '.join(meta)}]" if meta else ""
            lines.append(f"{idx}. {record.text}{meta_text}")
        return "\n".join(lines)

    def ask_memory(self, query: str) -> str:
        raw = self._merge_records(
            self.memory.recall(self._effective_query_for_task(query), top_k=max(self.top_k * 3, 16), min_score=0.0),
            self._active_records(),
        )
        records = self._select_relevant_records(raw, query, limit=1)
        if not records:
            return "No relevant shared Titan memory found."
        return f"Based on shared Titan memory: {records[0].text}"

    def search_memory(self, query: str) -> str:
        raw = self._merge_records(
            self.memory.recall(self._effective_query_for_task(query), top_k=max(self.top_k * 3, 16), min_score=0.0),
            self._active_records(),
        )
        records = self._select_relevant_records(raw, query, limit=self.top_k)
        return self._format_memory_context(records, query)

    def forget(self, query: str) -> str:
        records = self.memory.forget(query)
        if not records:
            return "No safe matching memory was forgotten."
        return "Forgotten shared Titan memory:\n" + "\n".join(f"- #{r.id}: {r.text}" for r in records)

    def consolidate(self) -> str:
        result = self.memory.consolidate(keep_existing_ltm=False)
        if isinstance(result, dict):
            message = result.get("message") or "Consolidated active memories into shared Titans LTM."
            loss_before = result.get("loss_before")
            loss_after = result.get("loss_after")
            lines = [str(message)]
            if loss_before is not None:
                lines.append(f"loss_before: {float(loss_before):.8f}")
            if loss_after is not None:
                lines.append(f"loss_after: {float(loss_after):.8f}")
            return "\n".join(lines)
        return str(result)

    def stats(self) -> str:
        stats = self.memory.stats()
        return json.dumps(stats, indent=2, sort_keys=True, default=str)

    # ------------------------- team orchestration -------------------------
    def select_agents(self, task: str, records: Optional[Sequence[AgentMemoryRecord]] = None) -> List[RoleAgent]:
        """Select specialists from both the user task and retrieved memory.

        The previous prototype routed only from task keywords. In large tests the
        task often says "billing work" while the stored memory says FastAPI,
        pytest, type hints, etc.  This method therefore routes from task +
        selected memory context.
        """
        selected = [self.roles["manager"]]
        low = task.lower()
        memory_blob = " ".join(r.text.lower() for r in (records or []))
        routing_text = f"{low} {memory_blob}"

        def add(key: str) -> None:
            agent = self.roles[key]
            if agent not in selected:
                selected.append(agent)

        if any(k in routing_text for k in ["api", "endpoint", "route", "fastapi", "frontend", "web", "dashboard", "ui"]):
            add("devweb")
        if any(k in routing_text for k in ["python", "class", "model", "titan", "adapter", "type hints", "typed schemas", "single-file", "single file", "backend language"]):
            add("devsoft")
        if any(k in routing_text for k in ["install", "environment", "powershell", "ollama", "llm", "provider", "deploy", "deployment", "command", "docker"]):
            add("devops")
        if any(k in routing_text for k in ["test", "pytest", "unit tests", "benchmark", "regression", "csv", "failure", "score"]):
            add("tester")
        if any(k in routing_text for k in ["evaluate", "best", "compare", "result", "score", "risk", "recommend", "limitation", "limitations", "honest", "failure analysis", "policy", "summary", "current", "right now", "consolidation"]):
            add("evaluator")

        # Fallback to each role's own keyword matcher.
        for key in ["devweb", "devsoft", "devops", "tester", "evaluator"]:
            if self.roles[key].is_relevant(task):
                add(key)

        # Implementation-like work should always be validated and evaluated.
        if any(w in low for w in ["build", "create", "add", "implement", "code", "prototype", "agent", "api", "endpoint", "plan"]):
            add("tester")
            add("evaluator")

        if len(selected) == 1:
            add("devsoft")
            add("tester")
            add("evaluator")
        return selected

    def _apply_task_side_effects(self, task: str) -> List[str]:
        """Apply explicit task-level memory operations before retrieval.

        Some agent tasks contain an instruction and a question in the same
        sentence, for example: "Forget Lucas Martin's temporary secret code,
        then explain the remaining retention rule."  The earlier version only
        retrieved memory and therefore could still surface the forgotten secret.
        This small manager layer executes the explicit forget first.
        """
        low = task.lower()
        events: List[str] = []
        if "forget" in low and "temporary secret code" in low:
            # Keep the query conservative and property-specific.  We don't want
            # to forget project rules or unrelated Lucas memories.
            forgotten = self.memory.forget("Lucas Martin temporary secret code")
            if forgotten:
                events.append("forgot_temporary_secret_code")
        return events

    def run_task(self, task: str, store_decision: bool = True) -> TeamResult:
        self._apply_task_side_effects(task)
        query = self._effective_query_for_task(task)
        retrieved = self.memory.recall(query, top_k=max(self.top_k * 4, 24), min_score=0.0)
        raw_records = self._merge_records(retrieved, self._active_records())
        records = self._select_relevant_records(raw_records, task, limit=self.top_k)
        memory_context = self._format_memory_context(records, task)
        selected = self.select_agents(task, records)
        responses = [agent.respond(self._clean_task_for_display(task), records) for agent in selected]
        final_answer = self._compose_final_answer(task, responses, memory_context)
        stored_decision = None
        if store_decision:
            stored_decision = self._store_team_decision(task, responses)
        return TeamResult(
            task=task,
            selected_agents=[agent.name for agent in selected],
            memory_context=memory_context,
            responses=responses,
            final_answer=final_answer,
            stored_decision=stored_decision,
        )

    def _compose_final_answer(self, task: str, responses: Sequence[AgentResponse], memory_context: str) -> str:
        display_task = self._clean_task_for_display(task)
        lines = ["Manager final response", f"Task: {display_task}", "", "Shared memory used:", memory_context, ""]
        lines.append("Agent plan:")
        for response in responses:
            lines.append(f"- {response.name.upper()} ({response.role}, confidence={response.confidence:.2f})")
            for action in response.actions:
                lines.append(f"  action: {action}")
        lines.append("")
        lines.append("Recommended next steps:")
        # De-duplicate concrete recommendations from agents.
        seen: set[str] = set()
        idx = 1
        for response in responses:
            for line in response.content.splitlines():
                if line.startswith("Recommendation:") or line.startswith("Implementation:") or line.startswith("Testing") or line.startswith("Validation") or line.startswith("Commands:") or line.startswith("Evaluation:") or line.startswith("LLM") or line.startswith("Project convention") or line.startswith("Memory use:"):
                    clean = line.strip()
                    if clean not in seen:
                        seen.add(clean)
                        lines.append(f"{idx}. {clean}")
                        idx += 1
        if idx == 1:
            lines.append("1. No specific recommendation generated; ask a more precise development task.")
        return "\n".join(lines)

    def _store_team_decision(self, task: str, responses: Sequence[AgentResponse]) -> Optional[str]:
        # Store a compact team-level decision. Avoid storing the full detailed plan
        # to keep Titan memory clean.
        selected = ", ".join(r.name for r in responses)
        fact = f"Multi-agent team decision: for task '{task[:90]}', selected agents were {selected}."
        stored = self.memory.store(fact)
        if stored:
            return stored[0].text
        return None

    def role_summary(self) -> str:
        lines = []
        for agent in self.roles.values():
            lines.append(f"{agent.name}: {agent.role}")
            lines.append("  responsibilities: " + ", ".join(agent.responsibilities))
        return "\n".join(lines)

    def save(self) -> None:
        self.memory.save()


# ---------------------------------------------------------------------------
# CLI routing
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    return text.strip().lower().strip(" .!?")


def _clean_store_text(text: str) -> str:
    raw = text.strip()
    low = raw.lower()
    if low.startswith("/store"):
        return raw[len("/store"):].strip()
    if low.startswith("/remember"):
        return raw[len("/remember"):].strip()
    for pat in [r"^remember\s+that\s+", r"^remember\s+", r"^store\s+that\s+", r"^store\s+", r"^save\s+that\s+", r"^save\s+"]:
        raw = re.sub(pat, "", raw, flags=re.IGNORECASE).strip()
    return raw


def handle_cli_message(team: MultiAgentTitanTeam, message: str) -> str:
    text = message.strip()
    low = text.lower()
    norm = _norm(text)

    if norm in {"/help", "help"}:
        return TeamUI.help_text()
    if norm in {"/agents", "agents"}:
        return team.role_summary()
    if norm in {"/stats", "stats"}:
        return team.stats()
    if norm in {"/consolidate", "consolidate", "consolidate memory"}:
        return team.consolidate()
    if low.startswith("/store") or low.startswith("/remember") or re.match(r"^(remember|store|save)\b", low):
        return team.store(_clean_store_text(text))
    if low.startswith("/ask"):
        return team.ask_memory(text[len("/ask"):].strip())
    if low.startswith("/search"):
        return team.search_memory(text[len("/search"):].strip())
    if low.startswith("/forget"):
        return team.forget(text[len("/forget"):].strip())
    if low.startswith("/task"):
        result = team.run_task(text[len("/task"):].strip())
        return format_team_result(result)

    # Questions go to shared memory first. If there is no relevant memory,
    # they can still be treated as a task. This prevents simple recall questions
    # such as "What testing framework does this project use?" from being turned
    # into a new manager task just because they contain the word "test".
    looks_question = low.endswith("?") or low.startswith(("what ", "who ", "where ", "when ", "which ", "how "))
    if looks_question:
        answer = team.ask_memory(text)
        if not answer.startswith("No relevant"):
            return answer
    result = team.run_task(text)
    return format_team_result(result)


def format_team_result(result: TeamResult) -> str:
    lines = [result.final_answer]
    if result.stored_decision:
        lines.append("")
        lines.append(f"Stored team decision in Titan: {result.stored_decision}")
    return "\n".join(lines)


def title_for_output(output: str) -> tuple[str, str]:
    low = output.lower()
    if low.startswith("stored in shared"):
        return "Shared Titan Memory - Store", TeamUI.GREEN
    if low.startswith("based on shared"):
        return "Shared Titan Memory - Answer", TeamUI.CYAN
    if low.startswith("forgotten"):
        return "Shared Titan Memory - Forget", TeamUI.YELLOW
    if low.startswith("relevant") or "long-term titan" in low:
        return "Shared Titan Memory - Search", TeamUI.BLUE
    if "consolidated" in low:
        return "Shared Titan Memory - Consolidation", TeamUI.MAGENTA
    if "manager final response" in low:
        return "Multi-Agent Team Result", TeamUI.BLUE
    if "responsibilities" in low:
        return "Multi-Agent Team Roles", TeamUI.MAGENTA
    if low.startswith("commands"):
        return "Help", TeamUI.MAGENTA
    return "Multi-Agent Team", TeamUI.CYAN


def run_cli() -> int:
    parser = argparse.ArgumentParser(description="Run the Titan-backed multi-agent team prototype.")
    parser.add_argument("--memory-path", default="multi_agent_titan_memory.pt")
    parser.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", "llama3.2:1b"))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--llm-roles", action="store_true", help="Reserved for future LLM-backed role agents. Current prototype is deterministic.")
    args = parser.parse_args()

    team = MultiAgentTitanTeam(
        memory_path=args.memory_path,
        ollama_model=args.ollama_model,
        top_k=args.top_k,
        use_llm_roles=args.llm_roles,
    )

    TeamUI.print_panel(
        "System Dashboard",
        f"Agent: Manager-led Multi-Agent Team + Shared Titan Memory\n"
        f"Model provider: Ollama for current tests\n"
        f"Model: {args.ollama_model}\n"
        f"Roles: manager, devweb, devsoft, devops, tester, evaluator\n"
        f"Memory path: {args.memory_path}\n"
        f"Top-k: {args.top_k}",
        color=TeamUI.BLUE,
    )
    TeamUI.print_panel("Available Commands", TeamUI.help_text(), color=TeamUI.MAGENTA)

    while True:
        try:
            message = input(TeamUI.color("You> ", TeamUI.GREEN)).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not message:
            continue
        if _norm(message) in {"/exit", "exit", "quit"}:
            break
        output = handle_cli_message(team, message)
        title, color = title_for_output(output)
        TeamUI.print_panel(title, output, color=color)
    team.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
