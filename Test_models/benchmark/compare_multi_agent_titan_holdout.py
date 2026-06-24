#!/usr/bin/env python3
"""Holdout/adversarial benchmark for the Titan-backed multi-agent team.

This benchmark is intentionally different from compare_multi_agent_titan_team.py.
It generates unseen cases with a deterministic seed in order to check that the
large-score improvement is not only due to memorizing one benchmark template.

Compared systems:
- no_memory_agent: baseline with no shared memory.
- single_titan_agent: one agent with Titan memory recall only.
- multi_agent_titan_team: manager-led team with shared Titan memory.

The benchmark uses synthetic project-management/coding memories, but the values,
people, modules, and distractors are intentionally different from the previous
large benchmark.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.multi_agent_titan_team import MultiAgentTitanTeam
from agents.titan_agent_memory import TitanAgentMemory


@dataclass
class Case:
    scale: str
    category: str
    name: str
    setup: List[str]
    task: str
    expected_any: List[str] = field(default_factory=list)
    expected_all: List[str] = field(default_factory=list)
    forbidden: List[str] = field(default_factory=list)
    expected_agents: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class Result:
    agent: str
    scale: str
    category: str
    name: str
    passed: bool
    answer: str
    selected_agents: List[str] = field(default_factory=list)
    failure: str = ""


def contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def clean_answer_for_eval(answer: str) -> str:
    # Do not remove content aggressively; we only normalize whitespace.
    return re.sub(r"\s+", " ", str(answer or "")).strip()


def evaluate_answer(case: Case, answer: str, selected_agents: Optional[Sequence[str]] = None, strict_agents: bool = False) -> Tuple[bool, str]:
    ans = clean_answer_for_eval(answer)
    selected = set(selected_agents or [])
    problems: List[str] = []

    if case.expected_all:
        missing = [x for x in case.expected_all if not contains(ans, x)]
        if missing:
            problems.append(f"missing_all={missing}")
    if case.expected_any:
        if not any(contains(ans, x) for x in case.expected_any):
            problems.append(f"missing_any={case.expected_any}")
    if case.forbidden:
        bad = [x for x in case.forbidden if contains(ans, x)]
        if bad:
            problems.append(f"forbidden={bad}")
    if strict_agents and case.expected_agents:
        missing_agents = [x for x in case.expected_agents if x not in selected]
        if missing_agents:
            problems.append(f"missing_agents={missing_agents}")

    if problems:
        return False, "; ".join(problems)
    return True, ""


class NoMemoryAgent:
    name = "no_memory_agent"

    def reset(self) -> None:
        pass

    def store_many(self, facts: Sequence[str]) -> None:
        pass

    def answer(self, task: str) -> Tuple[str, List[str]]:
        return "I do not know because no shared memory is available.", []


class SingleTitanAgent:
    name = "single_titan_agent"

    def __init__(self, top_k: int = 8) -> None:
        self.top_k = top_k
        self._tmp: Optional[tempfile.TemporaryDirectory[str]] = None
        self.memory: Optional[TitanAgentMemory] = None
        self.reset()

    def reset(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / "single_titan_holdout.pt"
        self.memory = TitanAgentMemory(memory_path=path, top_k=self.top_k)

    def store_many(self, facts: Sequence[str]) -> None:
        assert self.memory is not None
        for fact in facts:
            self.memory.store(fact)

    def answer(self, task: str) -> Tuple[str, List[str]]:
        assert self.memory is not None
        records = self.memory.recall(task, top_k=max(self.top_k * 3, 18), min_score=0.0)
        if not records:
            return "No relevant Titan memory found.", []
        # One-agent baseline has memory but no manager/evaluator filtering.
        lines = [record.text for record in records[: self.top_k]]
        return "\n".join(lines), []


class MultiAgentWrapper:
    name = "multi_agent_titan_team"

    def __init__(self, top_k: int = 8) -> None:
        self.top_k = top_k
        self._tmp: Optional[tempfile.TemporaryDirectory[str]] = None
        self.team: Optional[MultiAgentTitanTeam] = None
        self.reset()

    def reset(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / "multi_agent_holdout.pt"
        self.team = MultiAgentTitanTeam(memory_path=path, top_k=self.top_k)

    def store_many(self, facts: Sequence[str]) -> None:
        assert self.team is not None
        for fact in facts:
            self.team.store(fact)

    def answer(self, task: str) -> Tuple[str, List[str]]:
        assert self.team is not None
        result = self.team.run_task(task, store_decision=False)
        return result.final_answer, result.selected_agents


# ---------------------------------------------------------------------------
# Holdout case generation
# ---------------------------------------------------------------------------
PROJECT_MODULES = [
    "audit-log", "invoice", "inventory", "notification", "permission", "search", "billing", "analytics",
    "reporting", "scheduler", "profile", "gateway", "document", "ranking", "export", "subscription",
]

PROJECT_CONVENTIONS = [
    ("audit-log", "FastAPI", "pytest", "typed responses"),
    ("invoice", "REST", "pytest", "strict request schemas"),
    ("inventory", "FastAPI", "unit tests", "score field"),
    ("notification", "webhooks", "pytest", "retry policy"),
    ("permission", "RBAC", "regression tests", "typed policies"),
    ("search", "FastAPI", "integration tests", "rank score"),
    ("billing", "REST", "pytest", "error schema"),
    ("analytics", "API route", "unit tests", "typed metrics"),
]

UPDATE_PROPERTIES = [
    ("current preferred backend language", ["Ruby", "Go", "Python"], ["Ruby", "Go"]),
    ("current LLM provider for tests", ["Gemini", "Claude", "Ollama"], ["Gemini", "Claude"]),
    ("current testing framework", ["nose", "unittest", "pytest"], ["nose", "unittest"]),
    ("current demo UI style", ["plain text", "markdown only", "colored terminal panels"], ["plain text", "markdown only"]),
    ("current memory backend", ["vector store", "short-term memory", "shared Titan memory"], ["vector store", "short-term memory"]),
    ("current deployment shell", ["bash", "cmd", "PowerShell"], ["bash", "cmd"]),
    ("current reporting rule", ["hide failures", "only show wins", "report failures honestly"], ["hide failures", "only show wins"]),
    ("current benchmark output format", ["screenshots only", "text only", "CSV and markdown report"], ["screenshots only", "text only"]),
    # Unseen update-like properties: useful for detecting benchmark overfitting.
    ("current code formatter", ["black", "yapf", "ruff format"], ["black", "yapf"]),
    ("current API auth method", ["basic auth", "API key", "JWT"], ["basic auth", "API key"]),
]

PEOPLE = [
    ("Aline Durand", "release checklist"),
    ("Axel Durand", "schema migration"),
    ("Clara Moreau", "load testing"),
    ("Clement Moreau", "route review"),
    ("Mila Bernard", "UI accessibility"),
    ("Mathis Bernard", "provider evaluation"),
    ("Lina Robert", "benchmark report"),
    ("Leo Robert", "deployment script"),
    ("Nora Garnier", "risk summary"),
    ("Nathan Garnier", "memory adapter"),
]

NOISE_WORDS = ["banana", "coffee", "hotel", "router", "sticker", "capsule", "orange", "travel ad"]


def base_holdout_cases() -> List[Case]:
    return [
        Case(
            scale="holdout",
            category="provider_policy",
            name="holdout_free_llm_rule",
            setup=["For the current prototype, keep Ollama for tests; if the LLM changes later, it must be better and free."],
            task="Which LLM should we keep for tests now, and what constraint applies if we change later?",
            expected_all=["Ollama", "better", "free"],
            expected_agents=["manager", "devops", "evaluator"],
        ),
        Case(
            scale="holdout",
            category="manager_policy",
            name="holdout_memory_write_policy",
            setup=["Only the manager should write validated team decisions into shared Titan memory."],
            task="Who should write validated team decisions into memory?",
            expected_all=["manager", "Titan memory"],
            expected_agents=["manager", "evaluator"],
        ),
        Case(
            scale="holdout",
            category="testing_policy",
            name="holdout_test_before_benchmark",
            setup=["Before running a large benchmark, the focused unit test must pass first."],
            task="Prepare the validation order before a large benchmark.",
            expected_all=["test", "benchmark"],
            expected_agents=["manager", "tester", "evaluator"],
        ),
        Case(
            scale="holdout",
            category="ui_policy",
            name="holdout_visual_panels_rule",
            setup=["The terminal demo should keep colored rectangular panels for user-facing answers."],
            task="What visual style should the demo preserve?",
            expected_all=["colored", "panels"],
            expected_agents=["manager"],
        ),
        Case(
            scale="holdout",
            category="architecture_policy",
            name="holdout_manager_led_shared_memory",
            setup=["The strongest architecture tested so far is a manager-led team with shared Titan memory."],
            task="What architecture currently looks strongest for specialist agents with long-term recall?",
            expected_all=["manager", "shared Titan memory"],
            expected_agents=["manager", "evaluator"],
        ),
        Case(
            scale="holdout",
            category="devsoft_policy",
            name="holdout_single_file_titan_api",
            setup=["The Titan model should keep a clean single-file API with store, recall, forget, consolidate and save/load."],
            task="Summarize the software API rule for Titan integration.",
            expected_all=["single-file", "store", "recall", "forget", "consolidate"],
            expected_agents=["manager", "devsoft", "tester", "evaluator"],
        ),
    ]


def generate_project_convention_cases(rng: random.Random, count: int = 16) -> List[Case]:
    cases: List[Case] = []
    for i in range(count):
        module, api, tests, schema = PROJECT_CONVENTIONS[i % len(PROJECT_CONVENTIONS)]
        distractor_id = 700 + i
        setup = [
            f"Distractor: cafeteria token {distractor_id} is unrelated to development work.",
            f"Critical holdout convention for {module}: use {api}, {tests}, and {schema}.",
            f"Distractor: travel note HOLDOUT-{distractor_id} should not appear in implementation plans.",
        ]
        task = f"Create the implementation plan for the {module} feature and preserve the critical convention."
        expected_all = [api, tests]
        expected_any = [schema]
        cases.append(
            Case(
                scale="generated",
                category="project_convention",
                name=f"holdout_convention_{module.replace('-', '_')}",
                setup=setup,
                task=task,
                expected_all=expected_all,
                expected_any=expected_any,
                forbidden=["cafeteria", "travel note", f"HOLDOUT-{distractor_id}"],
                expected_agents=["manager", "devweb", "devsoft", "tester", "evaluator"],
            )
        )
    return cases


def generate_update_cases(count: int = 20) -> List[Case]:
    cases: List[Case] = []
    for i in range(count):
        prop, sequence, forbidden = UPDATE_PROPERTIES[i % len(UPDATE_PROPERTIES)]
        final = sequence[-1]
        setup = [f"The {prop} is {value}." for value in sequence]
        task = f"What is the {prop} right now?"
        cases.append(
            Case(
                scale="generated",
                category="update_latest_value",
                name=f"holdout_update_{i+1}_{re.sub(r'[^a-z0-9]+', '_', prop.lower()).strip('_')}",
                setup=setup,
                task=task,
                expected_any=[final],
                forbidden=forbidden,
                expected_agents=["manager", "evaluator"],
            )
        )
    return cases


def generate_identity_cases(count: int = 10) -> List[Case]:
    cases: List[Case] = []
    for i in range(count):
        person, task_owner = PEOPLE[i % len(PEOPLE)]
        other_person, other_task = PEOPLE[(i + 1) % len(PEOPLE)]
        setup = [
            f"{person} owns the {task_owner} task.",
            f"{other_person} owns the {other_task} task.",
            "Distractor: both people attended the same standup meeting.",
        ]
        cases.append(
            Case(
                scale="generated",
                category="identity_collision",
                name=f"holdout_owner_{i+1}_{person.replace(' ', '_').lower()}",
                setup=setup,
                task=f"Who owns the {task_owner} task?",
                expected_any=[person],
                forbidden=[other_person, other_task, "standup"],
                expected_agents=["manager"],
            )
        )
    return cases


def generate_noisy_cases(count: int = 12) -> List[Case]:
    cases: List[Case] = []
    for i in range(count):
        key = f"QX-{9000 + i}"
        noise = " ".join(NOISE_WORDS)
        setup = [
            f"Critical holdout memory {i+1}: the escalation key for incident {i+1} is {key}.",
            f"Noise distractor {i+1}: {noise} should be ignored.",
        ]
        task = f"Ignore noise !!! {noise} ### what is the escalation key for incident {i+1}?"
        cases.append(
            Case(
                scale="generated",
                category="noisy_query",
                name=f"holdout_noisy_incident_{i+1}",
                setup=setup,
                task=task,
                expected_any=[key],
                forbidden=NOISE_WORDS,
                expected_agents=["manager", "evaluator"],
            )
        )
    return cases


def generate_forget_retention_cases(count: int = 10) -> List[Case]:
    cases: List[Case] = []
    for i in range(count):
        suffix = i + 1
        setup = [
            f"Lucas Martin's temporary secret code for holdout {suffix} is TEMP-HOLD-{suffix}.",
            f"Retention rule holdout {suffix}: benchmark summaries must mention limitations and next steps.",
            f"Distractor: Lucas Martin's archived color was blue-{suffix}.",
        ]
        task = "Forget Lucas Martin's temporary secret code, then explain the remaining retention rule."
        cases.append(
            Case(
                scale="generated",
                category="forget_retention",
                name=f"holdout_forget_retention_{suffix}",
                setup=setup,
                task=task,
                expected_all=["limitations", "next steps"],
                forbidden=[f"TEMP-HOLD-{suffix}", f"blue-{suffix}"],
                expected_agents=["manager", "evaluator"],
            )
        )
    return cases


def generate_consolidation_cases(count: int = 8) -> List[Case]:
    rules = [
        ("Consolidation holdout rule A: demos must show selected agents and memory context.", ["selected agents", "memory context"]),
        ("Consolidation holdout rule B: reports must include CSV output and markdown summaries.", ["CSV", "markdown"]),
        ("Consolidation holdout rule C: evaluator must describe limitations and risks.", ["limitations", "risks"]),
        ("Consolidation holdout rule D: DevOps must document PowerShell commands and environment variables.", ["PowerShell", "environment variables"]),
    ]
    cases: List[Case] = []
    for i in range(count):
        rule, expected = rules[i % len(rules)]
        setup = [rule, "Please consolidate the active project rules after storing them."]
        task = "After consolidation, what project rule should be preserved?"
        cases.append(
            Case(
                scale="generated",
                category="consolidation",
                name=f"holdout_consolidation_{i+1}",
                setup=setup,
                task=task,
                expected_all=expected,
                expected_agents=["manager", "evaluator"],
            )
        )
    return cases


def generate_role_routing_cases() -> List[Case]:
    return [
        Case(
            scale="generated",
            category="role_routing",
            name="holdout_fullstack_all_roles",
            setup=["The project uses FastAPI, pytest, PowerShell setup commands, and honest evaluation reports."],
            task="Build a full-stack plan for a protected FastAPI route with tests, installation commands and evaluation risks.",
            expected_all=["FastAPI", "pytest", "Commands", "Evaluation"],
            expected_agents=["manager", "devweb", "devsoft", "devops", "tester", "evaluator"],
        ),
        Case(
            scale="generated",
            category="role_routing",
            name="holdout_devops_provider_setup",
            setup=["Provider configuration should default to Ollama and remain configurable for a future free alternative."],
            task="Prepare the LLM provider setup commands and future provider policy.",
            expected_all=["Ollama", "configurable", "free"],
            expected_agents=["manager", "devops", "evaluator"],
        ),
        Case(
            scale="generated",
            category="role_routing",
            name="holdout_testing_evaluation_only",
            setup=["Validation reports should compare baselines, list failed cases, and summarize risk."],
            task="Create a benchmark evaluation checklist.",
            expected_all=["baseline", "failed", "risk"],
            expected_agents=["manager", "tester", "evaluator"],
        ),
    ]


def generate_profile_cases(count: int = 6) -> List[Case]:
    cases: List[Case] = []
    for i in range(count):
        person = ["Emma Laurent", "Lucas Martin", "Sarah Nguyen", "Noah Bernard", "Mila Bernard", "Hugo Morel"][i % 6]
        setup = [
            f"{person} profile: works on database optimization.",
            f"{person} profile: also owns the mobile accessibility project.",
            f"{person} profile: prefers concise technical reports.",
        ]
        cases.append(
            Case(
                scale="generated",
                category="profile_summary",
                name=f"holdout_profile_summary_{i+1}",
                setup=setup,
                task=f"Summarize the project profile for {person}.",
                expected_all=["database optimization", "mobile accessibility project"],
                expected_any=["concise"],
                expected_agents=["manager", "evaluator"],
            )
        )
    return cases


def build_cases(seed: int, include_generated: int = 0) -> List[Case]:
    rng = random.Random(seed)
    cases: List[Case] = []
    cases.extend(base_holdout_cases())
    cases.extend(generate_project_convention_cases(rng, 16))
    cases.extend(generate_update_cases(20))
    cases.extend(generate_identity_cases(10))
    cases.extend(generate_noisy_cases(12))
    cases.extend(generate_forget_retention_cases(10))
    cases.extend(generate_consolidation_cases(8))
    cases.extend(generate_role_routing_cases())
    cases.extend(generate_profile_cases(6))

    # Optional extra random convention/update cases for heavier runs.
    for i in range(include_generated):
        module = rng.choice(PROJECT_MODULES)
        api = rng.choice(["FastAPI", "REST", "webhook route", "API route"])
        test = rng.choice(["pytest", "unit tests", "integration tests"])
        schema = rng.choice(["typed schemas", "score field", "error contract", "response status"])
        old_value = rng.choice(["legacy", "manual", "temporary"])
        new_value = rng.choice(["current", "validated", "stable"])
        cases.append(
            Case(
                scale="random",
                category="random_project_convention",
                name=f"random_convention_{i+1}_{module}",
                setup=[
                    f"Random distractor {i+1}: {module} parking code is not relevant.",
                    f"Critical random convention for {module}: use {api}, {test}, and {schema}.",
                    f"The {module} mode was {old_value}.",
                    f"The {module} mode is now {new_value}.",
                ],
                task=f"Create the plan for the {module} module and preserve its current convention.",
                expected_all=[api, test],
                expected_any=[schema, new_value],
                forbidden=["parking code", old_value],
                expected_agents=["manager", "devweb", "devsoft", "tester", "evaluator"],
            )
        )
    return cases


def run_case(agent_obj: object, case: Case, strict_agents: bool = False) -> Result:
    agent_obj.reset()
    agent_obj.store_many(case.setup)

    # Consolidation cases explicitly trigger consolidation before answering.
    if case.category == "consolidation" and hasattr(agent_obj, "team"):
        try:
            agent_obj.team.consolidate()
        except Exception:
            pass
    if case.category == "consolidation" and hasattr(agent_obj, "memory"):
        try:
            agent_obj.memory.consolidate()
        except Exception:
            pass

    answer, selected = agent_obj.answer(case.task)
    ok, failure = evaluate_answer(case, answer, selected, strict_agents=strict_agents)
    return Result(
        agent=agent_obj.name,
        scale=case.scale,
        category=case.category,
        name=case.name,
        passed=ok,
        answer=answer,
        selected_agents=list(selected),
        failure=failure,
    )


def summarize(results: Sequence[Result]) -> str:
    lines: List[str] = []
    lines.append("=" * 100)
    lines.append("HOLDOUT / ADVERSARIAL MULTI-AGENT TITAN TEAM COMPARISON")
    lines.append("=" * 100)
    lines.append("")
    lines.append("GLOBAL RESULTS")
    lines.append("-" * 100)
    by_agent: Dict[str, List[Result]] = defaultdict(list)
    for r in results:
        by_agent[r.agent].append(r)
    for agent, rows in by_agent.items():
        good = sum(r.passed for r in rows)
        total = len(rows)
        lines.append(f"{agent:24s} | {good:3d}/{total:<3d} ({good/total*100:5.1f}%)")

    lines.append("")
    lines.append("RESULTS BY SCALE")
    lines.append("-" * 100)
    by_agent_scale: Dict[Tuple[str, str], List[Result]] = defaultdict(list)
    for r in results:
        by_agent_scale[(r.agent, r.scale)].append(r)
    for (agent, scale), rows in sorted(by_agent_scale.items()):
        good = sum(r.passed for r in rows)
        total = len(rows)
        lines.append(f"{agent:24s} | {scale:10s} | {good:3d}/{total:<3d} ({good/total*100:5.1f}%)")

    lines.append("")
    lines.append("RESULTS BY CATEGORY")
    lines.append("-" * 100)
    by_agent_cat: Dict[Tuple[str, str], List[Result]] = defaultdict(list)
    for r in results:
        by_agent_cat[(r.agent, r.category)].append(r)
    for (agent, category), rows in sorted(by_agent_cat.items()):
        good = sum(r.passed for r in rows)
        total = len(rows)
        lines.append(f"{agent:24s} | {category:28s} | {good:3d}/{total:<3d} ({good/total*100:5.1f}%)")

    failed = [r for r in results if not r.passed]
    lines.append("")
    lines.append("FAILURES")
    lines.append("-" * 100)
    if not failed:
        lines.append("No failures.")
    else:
        for r in failed[:120]:
            ans = clean_answer_for_eval(r.answer)
            if len(ans) > 260:
                ans = ans[:260] + "..."
            lines.append(f"{r.agent} | {r.scale} | {r.category} | {r.name} | {r.failure} | answer={ans!r}")
        if len(failed) > 120:
            lines.append(f"... {len(failed) - 120} more failures omitted from console output.")
    return "\n".join(lines)


def save_outputs(results: Sequence[Result], cases: Sequence[Case], out_dir: Path, seed: int) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"multi_agent_team_holdout_{stamp}.csv"
    md_path = out_dir / f"multi_agent_team_holdout_report_{stamp}.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["agent", "scale", "category", "case", "passed", "selected_agents", "failure", "answer"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "agent": r.agent,
                    "scale": r.scale,
                    "category": r.category,
                    "case": r.name,
                    "passed": r.passed,
                    "selected_agents": ",".join(r.selected_agents),
                    "failure": r.failure,
                    "answer": clean_answer_for_eval(r.answer),
                }
            )

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Holdout / adversarial multi-agent Titan team comparison\n\n")
        f.write(f"Cases: {len(cases)} | Agents: 3 | Seed: {seed}\n\n")
        f.write("This benchmark uses different synthetic cases from the previous large benchmark. It is intended as a holdout check, not as training data.\n\n")
        # Compact markdown tables.
        by_agent: Dict[str, List[Result]] = defaultdict(list)
        for r in results:
            by_agent[r.agent].append(r)
        f.write("## Global results\n\n")
        f.write("| Agent | Score |\n|---|---:|\n")
        for agent, rows in by_agent.items():
            good = sum(r.passed for r in rows)
            total = len(rows)
            f.write(f"| {agent} | {good}/{total} ({good/total*100:.1f}%) |\n")
        f.write("\n## Failures\n\n")
        failed = [r for r in results if not r.passed]
        if not failed:
            f.write("No failures.\n")
        else:
            for r in failed:
                ans = clean_answer_for_eval(r.answer)
                if len(ans) > 400:
                    ans = ans[:400] + "..."
                f.write(f"- **{r.agent} / {r.scale} / {r.category} / {r.name}**: {r.failure} — answer: `{ans}`\n")
    return csv_path, md_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a holdout/adversarial multi-agent Titan benchmark.")
    parser.add_argument("--agents", nargs="+", choices=["no_memory", "single", "multi"], default=["no_memory", "single", "multi"])
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--extra-random-cases", type=int, default=0, help="Add extra seed-generated random cases on top of the fixed holdout set.")
    parser.add_argument("--strict-agents", action="store_true", help="Also require expected specialist agents to be selected.")
    parser.add_argument("--show-answers", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--out-dir", default=str(ROOT / "benchmark" / "results"))
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args(argv)

    cases = build_cases(args.seed, include_generated=args.extra_random_cases)
    agent_objects: List[object] = []
    if "no_memory" in args.agents:
        agent_objects.append(NoMemoryAgent())
    if "single" in args.agents:
        agent_objects.append(SingleTitanAgent(top_k=args.top_k))
    if "multi" in args.agents:
        agent_objects.append(MultiAgentWrapper(top_k=args.top_k))

    results: List[Result] = []
    total = len(cases) * len(agent_objects)
    idx = 0
    for case in cases:
        for agent in agent_objects:
            idx += 1
            print(f"[{idx}/{total}] {agent.name} | {case.scale} | {case.category} | {case.name}")
            result = run_case(agent, case, strict_agents=args.strict_agents)
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"  {status}" + (f" | {result.failure}" if result.failure else ""))
            if args.show_answers:
                print("  Answer:", clean_answer_for_eval(result.answer))

    print()
    print(summarize(results))

    if args.save:
        csv_path, md_path = save_outputs(results, cases, Path(args.out_dir), seed=args.seed)
        print()
        print(f"CSV saved to: {csv_path}")
        print(f"Report saved to: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
