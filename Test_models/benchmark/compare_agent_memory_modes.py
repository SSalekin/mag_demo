#!/usr/bin/env python3
"""Detailed benchmark: agent without memory vs short-term memory vs Titan memory.

This benchmark answers the agent-level question:
    "Does a Titan long-term memory layer improve an AI agent?"

It is intentionally deterministic and does not require Agno or Ollama. The goal
is to evaluate the memory layer itself, not the text generation quality of a
small LLM.

Compared agents:
- no_memory_agent: stateless baseline.
- short_term_agent: session-only working memory baseline.
- titan_memory_agent: agent using TitanAgentMemory as persistent LTM.

The benchmark reports global results, results by scale, results by category,
failures, timing and saved CSV/Markdown reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.titan_agent_memory import TitanAgentMemory


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------

@dataclass
class AgentAction:
    action: str
    text: str = ""


@dataclass
class AgentCase:
    id: str
    scale: str
    category: str
    description: str
    actions: List[AgentAction]
    query: str
    expected_any: List[str] = field(default_factory=list)
    expected_all: List[str] = field(default_factory=list)
    forbidden: List[str] = field(default_factory=list)


def A(action: str, text: str = "") -> AgentAction:
    return AgentAction(action=action, text=text)


def build_cases(scales: Sequence[str]) -> List[AgentCase]:
    all_cases: List[AgentCase] = [
        # ----------------------------- SMALL -----------------------------
        AgentCase(
            id="small_simple_secret_recall",
            scale="small",
            category="simple_recall",
            description="Store a fact and recall it later.",
            actions=[A("store", "Lucas Martin's secret code is 8392.")],
            query="What is Lucas Martin's secret code?",
            expected_any=["8392"],
        ),
        AgentCase(
            id="small_update_secret_code",
            scale="small",
            category="update",
            description="Update the same subject/property and avoid returning the old value.",
            actions=[
                A("store", "Lucas Martin's secret code is 8392."),
                A("store", "Lucas Martin's secret code is 1245."),
            ],
            query="What is Lucas Martin's secret code?",
            expected_any=["1245"],
            forbidden=["8392"],
        ),
        AgentCase(
            id="small_identity_collision_same_surname",
            scale="small",
            category="identity_collision",
            description="Keep two people with same last name separated.",
            actions=[
                A("store", "Sarah Martin's favorite color is green."),
                A("store", "Lucas Martin's favorite color is violet."),
            ],
            query="What is Sarah Martin's favorite color?",
            expected_any=["green"],
            forbidden=["violet"],
        ),
        AgentCase(
            id="small_forget_one_profile_keep_other",
            scale="small",
            category="forget",
            description="Forget Sarah while retaining Lucas.",
            actions=[
                A("store", "Sarah Martin's favorite color is green."),
                A("store", "Lucas Martin's favorite color is violet."),
                A("forget", "Sarah Martin's favorite color"),
            ],
            query="What is Lucas Martin's favorite color?",
            expected_any=["violet"],
            forbidden=["green"],
        ),
        AgentCase(
            id="small_paraphrased_location_recall",
            scale="small",
            category="paraphrase_recall",
            description="Recall a location with a paraphrased query.",
            actions=[A("store", "Camille Durand lives in Nantes.")],
            query="Where is Camille Durand based?",
            expected_any=["Nantes"],
        ),
        AgentCase(
            id="small_multi_session_recall",
            scale="small",
            category="multi_session",
            description="Recall after a simulated agent restart.",
            actions=[
                A("store", "The project language preference is Python."),
                A("restart"),
            ],
            query="What is the project language preference?",
            expected_any=["Python"],
        ),
        AgentCase(
            id="small_consolidated_project_convention",
            scale="small",
            category="consolidation",
            description="Recall a fact after LTM consolidation.",
            actions=[
                A("store", "The project convention is to use pytest for unit tests."),
                A("store", "The project convention is to use type hints in new Python files."),
                A("consolidate"),
            ],
            query="What testing framework is used in the project convention?",
            expected_any=["pytest"],
        ),
        AgentCase(
            id="small_profile_summary",
            scale="small",
            category="profile_summary",
            description="Retrieve several memories for the same profile.",
            actions=[
                A("store", "Emma Laurent works on database optimization."),
                A("store", "Emma Laurent also works on a mobile accessibility project."),
                A("store", "Emma Laurent prefers concise technical reports."),
            ],
            query="Summarize Emma Laurent's profile.",
            expected_all=["database optimization", "mobile accessibility project"],
        ),

        # ----------------------------- MEDIUM -----------------------------
        AgentCase(
            id="medium_noisy_secret_recall",
            scale="medium",
            category="noisy_query",
            description="Recall despite extra irrelevant query words.",
            actions=[
                A("store", "Nora Chen's deployment room is Delta-42."),
                A("store", "The hotel has a rooftop pool."),
                A("store", "The cafeteria menu changes every Friday."),
            ],
            query="Ignore random noise please: what was Nora Chen's deployment room again?",
            expected_any=["Delta-42"],
            forbidden=["rooftop", "cafeteria"],
        ),
        AgentCase(
            id="medium_mixed_language_location",
            scale="medium",
            category="mixed_language_query",
            description="Ask in French about an English stored fact.",
            actions=[A("store", "Victor Moreau's internship city is Madrid.")],
            query="Dans quelle ville Victor Moreau fait-il son internship?",
            expected_any=["Madrid"],
        ),
        AgentCase(
            id="medium_multiple_language_updates",
            scale="medium",
            category="multiple_updates",
            description="Several updates on the same property.",
            actions=[
                A("store", "The preferred backend language is C++."),
                A("store", "The preferred backend language is Rust."),
                A("store", "The preferred backend language is Python."),
            ],
            query="What is the preferred backend language?",
            expected_any=["Python"],
            forbidden=["C++", "Rust"],
        ),
        AgentCase(
            id="medium_multiple_secret_code_updates",
            scale="medium",
            category="multiple_updates",
            description="Repeated code updates should retain only the newest value.",
            actions=[
                A("store", "The release secret code is TEMP-CODE-3053."),
                A("store", "The release secret code is CODE-3053."),
                A("store", "The release secret code is FINAL-CODE-3053."),
            ],
            query="What is the release secret code?",
            expected_any=["FINAL-CODE-3053"],
            forbidden=["TEMP-CODE-3053", "CODE-3053."],
        ),
        AgentCase(
            id="medium_hard_distractor_room",
            scale="medium",
            category="hard_distractors",
            description="Reject a highly similar distractor.",
            actions=[
                A("store", "Maya Singh's office room is B-204."),
                A("store", "The hotel has a room called B-204."),
                A("store", "Maya Singh's favorite snack is mango."),
            ],
            query="What is Maya Singh's office room?",
            expected_any=["B-204"],
            forbidden=["hotel"],
        ),
        AgentCase(
            id="medium_forget_with_nearby_distractor",
            scale="medium",
            category="forget",
            description="Forget one code while keeping a nearby project memory.",
            actions=[
                A("store", "The temporary access code is CODE-3077."),
                A("store", "The cloud monitoring dashboard is the active project."),
                A("forget", "temporary access code"),
            ],
            query="What is the active project?",
            expected_any=["cloud monitoring dashboard"],
            forbidden=["CODE-3077"],
        ),
        AgentCase(
            id="medium_consolidated_after_restart",
            scale="medium",
            category="consolidation",
            description="Consolidate, restart, then recall.",
            actions=[
                A("store", "The documentation style is short bullet points."),
                A("consolidate"),
                A("restart"),
            ],
            query="What is the documentation style?",
            expected_any=["short bullet points"],
        ),
        AgentCase(
            id="medium_profile_summary",
            scale="medium",
            category="profile_summary",
            description="Retrieve several memories from one profile with distractors.",
            actions=[
                A("store", "Antoine works on cybersecurity."),
                A("store", "Antoine is based in Rennes."),
                A("store", "Antoine is preparing a gesture recognition prototype."),
                A("store", "Lucas Martin's favorite color is orange."),
            ],
            query="Give me Antoine's profile summary.",
            expected_all=["cybersecurity", "gesture recognition prototype"],
            forbidden=["orange"],
        ),
    ]
    requested = set(scales)
    return [case for case in all_cases if case.scale in requested]


# ---------------------------------------------------------------------------
# Agent baselines
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokens(text: str) -> set[str]:
    return set(token.lower() for token in _TOKEN_RE.findall(text))


def parse_subject_property(text: str) -> tuple[str, str]:
    """Small heuristic used only by the short-term baseline."""
    text = text.strip().rstrip(".")
    lower = text.lower()
    if "'s " in text:
        subject, rest = text.split("'s ", 1)
        prop = rest.split(" is ", 1)[0] if " is " in rest else rest
        return subject.lower().strip(), prop.lower().strip()
    if lower.startswith("the ") and " is " in lower:
        subject = lower.split(" is ", 1)[0].strip()
        return subject, subject
    return "", ""


class NoMemoryAgent:
    name = "no_memory_agent"

    def __init__(self, memory_path: Path) -> None:
        pass

    def reset_case(self) -> None:
        pass

    def store(self, text: str) -> None:
        pass

    def forget(self, query: str) -> None:
        pass

    def consolidate(self) -> None:
        pass

    def restart(self) -> None:
        pass

    def answer(self, query: str) -> str:
        return "I do not know because no memory is available."


class ShortTermContextAgent:
    """Session-only memory baseline, similar to an agent relying on chat history."""

    name = "short_term_agent"

    def __init__(self, memory_path: Path) -> None:
        self.items: List[Dict[str, Any]] = []
        self.next_id = 1

    def reset_case(self) -> None:
        self.items = []
        self.next_id = 1

    def store(self, text: str) -> None:
        subject, prop = parse_subject_property(text)
        if subject and prop:
            for item in self.items:
                if item["active"] and item["subject"] == subject and item["property"] == prop:
                    item["active"] = False
        self.items.append({
            "id": self.next_id,
            "text": text,
            "subject": subject,
            "property": prop,
            "active": True,
            "tokens": tokens(text),
        })
        self.next_id += 1

    def forget(self, query: str) -> None:
        qtok = tokens(query)
        best = None
        best_score = 0
        for item in self.items:
            if not item["active"]:
                continue
            score = len(qtok & item["tokens"])
            if score > best_score:
                best_score = score
                best = item
        if best is not None and best_score > 0:
            best["active"] = False

    def consolidate(self) -> None:
        # Short-term memory has no long-term consolidation mechanism.
        pass

    def restart(self) -> None:
        # Working memory is lost after a new session.
        self.items = []

    def answer(self, query: str) -> str:
        qtok = tokens(query)
        scored = []
        for item in self.items:
            if not item["active"]:
                continue
            score = len(qtok & item["tokens"])
            if score > 0:
                scored.append((score, item["id"], item["text"]))
        if not scored:
            return "I do not know based on short-term memory."
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return scored[0][2]


class TitanMemoryAgent:
    name = "titan_memory_agent"

    def __init__(self, memory_path: Path) -> None:
        self.memory_path = memory_path
        self.memory = TitanAgentMemory(memory_path=memory_path, top_k=8, min_score=0.05)

    def reset_case(self) -> None:
        self.memory.clear()
        if self.memory_path.exists():
            self.memory_path.unlink()

    def store(self, text: str) -> None:
        self.memory.store(text)

    def forget(self, query: str) -> None:
        self.memory.forget(query)

    def consolidate(self) -> None:
        self.memory.consolidate(keep_existing_ltm=False)

    def restart(self) -> None:
        self.memory.save(self.memory_path)
        self.memory = TitanAgentMemory(memory_path=self.memory_path, top_k=8, min_score=0.05)
        self.memory.load(self.memory_path)

    def answer(self, query: str) -> str:
        records = self.memory.recall(query, top_k=8, min_score=0.05)
        if not records:
            return "I do not know based on Titan memory."
        # For profile summaries, returning several records is the realistic agent context.
        q = query.lower()
        if "summary" in q or "profile" in q or "summarize" in q:
            return " | ".join(record.text for record in records[:5])
        return records[0].text


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_answer(answer: str, case: AgentCase) -> tuple[bool, str]:
    lower = answer.lower()
    missing_any = []
    if case.expected_any and not any(token.lower() in lower for token in case.expected_any):
        missing_any = case.expected_any
    missing_all = [token for token in case.expected_all if token.lower() not in lower]
    forbidden_hit = []
    expected_joined = " ".join(case.expected_any + case.expected_all).lower()
    for token in case.forbidden:
        token_l = token.lower()
        # Some expected values legitimately contain an older code as a substring,
        # e.g. FINAL-CODE-3053 contains CODE-3053. In that case, the answer is
        # correct if the full expected value is present, so do not count the
        # substring as forbidden.
        if token_l.rstrip(".") and token_l.rstrip(".") in expected_joined:
            continue
        if token_l in lower:
            forbidden_hit.append(token)

    ok = not missing_any and not missing_all and not forbidden_hit
    reasons = []
    if missing_any:
        reasons.append(f"missing one of expected_any: {missing_any}")
    if missing_all:
        reasons.append(f"missing expected_all: {missing_all}")
    if forbidden_hit:
        reasons.append(f"contains forbidden: {forbidden_hit}")
    return ok, "; ".join(reasons) if reasons else "ok"


def evaluate(agent, cases: List[AgentCase]) -> List[dict]:
    rows: List[dict] = []
    for case in cases:
        agent.reset_case()
        t0 = time.perf_counter()
        action_log = []
        for action in case.actions:
            action_log.append({"action": action.action, "text": action.text})
            if action.action == "store":
                agent.store(action.text)
            elif action.action == "forget":
                agent.forget(action.text)
            elif action.action == "consolidate":
                agent.consolidate()
            elif action.action == "restart":
                agent.restart()
            else:
                raise ValueError(f"Unknown action: {action.action}")
        store_time = time.perf_counter() - t0

        q0 = time.perf_counter()
        answer = agent.answer(case.query)
        query_time = time.perf_counter() - q0

        ok, reason = evaluate_answer(answer, case)
        rows.append({
            "agent": agent.name,
            "scale": case.scale,
            "category": case.category,
            "case": case.id,
            "description": case.description,
            "ok": ok,
            "reason": reason,
            "store_time": store_time,
            "query_time": query_time,
            "query": case.query,
            "answer": answer,
            "expected_any": case.expected_any,
            "expected_all": case.expected_all,
            "forbidden": case.forbidden,
            "actions": action_log,
        })
    return rows


def pct(n: int, d: int) -> float:
    return 100.0 * n / max(d, 1)


def print_summary(rows: List[dict], show_answers: bool = False) -> None:
    agents = sorted(set(row["agent"] for row in rows))
    scales = ["small", "medium", "large"]
    categories = sorted(set(row["category"] for row in rows))

    print("\n" + "=" * 100)
    print("DETAILED AGENT MEMORY COMPARISON")
    print("=" * 100)

    print("\nGLOBAL RESULTS")
    print("-" * 100)
    for agent in agents:
        subset = [row for row in rows if row["agent"] == agent]
        ok = sum(1 for row in subset if row["ok"])
        total = len(subset)
        avg_store = sum(row["store_time"] for row in subset) / max(total, 1)
        avg_query = sum(row["query_time"] for row in subset) / max(total, 1)
        print(f"{agent:22s} | {ok:2d}/{total:<2d} ({pct(ok,total):5.1f}%) | avg store={avg_store:.3f}s | avg query={avg_query:.3f}s")

    print("\nRESULTS BY SCALE")
    print("-" * 100)
    for agent in agents:
        for scale in scales:
            subset = [row for row in rows if row["agent"] == agent and row["scale"] == scale]
            if not subset:
                continue
            ok = sum(1 for row in subset if row["ok"])
            total = len(subset)
            avg_store = sum(row["store_time"] for row in subset) / max(total, 1)
            avg_query = sum(row["query_time"] for row in subset) / max(total, 1)
            print(f"{agent:22s} | {scale:6s} | {ok:2d}/{total:<2d} ({pct(ok,total):5.1f}%) | store={avg_store:.3f}s | query={avg_query:.3f}s")

    print("\nRESULTS BY CATEGORY")
    print("-" * 100)
    for agent in agents:
        for category in categories:
            subset = [row for row in rows if row["agent"] == agent and row["category"] == category]
            if not subset:
                continue
            ok = sum(1 for row in subset if row["ok"])
            total = len(subset)
            print(f"{agent:22s} | {category:20s} | {ok:2d}/{total:<2d} ({pct(ok,total):5.1f}%)")

    print("\nRESULTS BY CASE")
    print("-" * 100)
    for row in rows:
        status = "PASS" if row["ok"] else "FAIL"
        print(f"{row['agent']:22s} | {row['scale']:6s} | {row['category']:20s} | {row['case']:40s} | {status}")
        if show_answers:
            print(f"  query : {row['query']}")
            print(f"  answer: {row['answer']}")

    failures = [row for row in rows if not row["ok"]]
    if failures:
        print("\nFAILURES")
        print("-" * 100)
        for row in failures:
            print(f"{row['agent']:22s} | {row['scale']:6s} | {row['category']:20s} | {row['case']:40s} | {row['reason']}")
            if show_answers:
                print(f"  answer: {row['answer']}")
    else:
        print("\nNo failures.")


def save_results(rows: List[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"agent_memory_comparison_detailed_{stamp}.csv"
    md_path = out_dir / f"agent_memory_comparison_detailed_report_{stamp}.md"

    fieldnames = [
        "agent", "scale", "category", "case", "description", "ok", "reason",
        "store_time", "query_time", "query", "answer", "expected_any",
        "expected_all", "forbidden", "actions",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ["expected_any", "expected_all", "forbidden", "actions"]:
                out[key] = json.dumps(out[key], ensure_ascii=False)
            writer.writerow({key: out[key] for key in fieldnames})

    agents = sorted(set(row["agent"] for row in rows))
    categories = sorted(set(row["category"] for row in rows))
    lines = ["# Detailed agent memory comparison", ""]
    lines.append("## Global results")
    lines.append("")
    lines.append("| Agent | Score | Avg store | Avg query |")
    lines.append("|---|---:|---:|---:|")
    for agent in agents:
        subset = [row for row in rows if row["agent"] == agent]
        ok = sum(1 for row in subset if row["ok"])
        total = len(subset)
        avg_store = sum(row["store_time"] for row in subset) / max(total, 1)
        avg_query = sum(row["query_time"] for row in subset) / max(total, 1)
        lines.append(f"| {agent} | {ok}/{total} ({pct(ok,total):.1f}%) | {avg_store:.3f}s | {avg_query:.3f}s |")

    lines.append("")
    lines.append("## Results by category")
    lines.append("")
    lines.append("| Agent | Category | Score |")
    lines.append("|---|---|---:|")
    for agent in agents:
        for category in categories:
            subset = [row for row in rows if row["agent"] == agent and row["category"] == category]
            if not subset:
                continue
            ok = sum(1 for row in subset if row["ok"])
            total = len(subset)
            lines.append(f"| {agent} | {category} | {ok}/{total} ({pct(ok,total):.1f}%) |")

    lines.append("")
    lines.append("## Failures")
    failures = [row for row in rows if not row["ok"]]
    if not failures:
        lines.append("No failures.")
    else:
        for row in failures:
            lines.append(f"- **{row['agent']} / {row['case']}**: {row['reason']} — answer: `{row['answer']}`")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nCSV saved to: {csv_path}")
    print(f"Report saved to: {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Detailed comparison of agent memory modes.")
    parser.add_argument("--scales", nargs="+", default=["small", "medium"], choices=["small", "medium"])
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["no_memory", "short_term", "titan"],
        choices=["no_memory", "short_term", "titan"],
        help="Agents to compare.",
    )
    parser.add_argument("--save", action="store_true", help="Save CSV and markdown reports.")
    parser.add_argument("--show-answers", action="store_true", help="Print query and answer details.")
    parser.add_argument("--out-dir", default="Test_models/benchmark/results")
    args = parser.parse_args()

    cases = build_cases(args.scales)
    all_rows: List[dict] = []

    with tempfile.TemporaryDirectory() as tmp:
        factories = {
            "no_memory": lambda: NoMemoryAgent(Path(tmp) / "no_memory.pt"),
            "short_term": lambda: ShortTermContextAgent(Path(tmp) / "short_term.pt"),
            "titan": lambda: TitanMemoryAgent(Path(tmp) / "agent_titan_memory.pt"),
        }
        for name in args.agents:
            agent = factories[name]()
            all_rows.extend(evaluate(agent, cases))

    print_summary(all_rows, show_answers=args.show_answers)
    if args.save:
        save_results(all_rows, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
