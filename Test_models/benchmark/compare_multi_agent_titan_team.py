#!/usr/bin/env python3
"""Detailed benchmark for the Titan-backed multi-agent team.

This benchmark is intentionally larger than the first prototype benchmark.  It
compares three modes:

1. no_memory_agent: no persistent memory and no team orchestration.
2. single_titan_agent: Titan memory retrieval only, no specialized roles.
3. multi_agent_titan_team: manager-led team with shared Titan memory.

The goal is not to claim that multi-agent systems are always better.  The goal
is to test whether the team architecture can actually use shared Titan memory,
route tasks to useful roles, and keep project conventions across harder tasks.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence

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


SMALL_CASES: List[Case] = [
    Case(
        scale="small",
        category="project_convention",
        name="api_endpoint_uses_pytest_type_hints",
        setup=["This project uses pytest and type hints in new Python files."],
        task="Create a plan to add a FastAPI endpoint for Titan memory search.",
        expected_all=["pytest", "type hints", "endpoint"],
        expected_agents=["manager", "devweb", "devsoft", "tester", "evaluator"],
    ),
    Case(
        scale="small",
        category="devops",
        name="ollama_provider_config",
        setup=["For current tests, the multi-agent prototype must use Ollama and keep the LLM provider configurable."],
        task="Plan the environment setup for the multi-agent system and explain the LLM provider choice.",
        expected_all=["Ollama", "configurable"],
        expected_agents=["manager", "devops", "evaluator"],
    ),
    Case(
        scale="small",
        category="testing",
        name="benchmark_regression_plan",
        setup=["Before benchmarks, every new agent feature must pass a focused test first."],
        task="Design the validation plan for a new multi-agent memory feature.",
        expected_all=["test", "benchmark"],
        expected_agents=["manager", "tester", "evaluator"],
    ),
    Case(
        scale="small",
        category="memory_recall",
        name="preferred_architecture_shared_titan",
        setup=["The current best architecture is a manager-led multi-agent team with shared Titan memory."],
        task="Which architecture is currently preferred?",
        expected_all=["manager-led", "shared Titan memory"],
        expected_agents=["manager"],
    ),
    Case(
        scale="small",
        category="devsoft",
        name="single_file_titan_preserved",
        setup=["The Titan model is currently implemented as a single-file model in Test_models/models/titan_model.py."],
        task="Refactor the software architecture without breaking the current Titan model.",
        expected_all=["Titan", "single-file"],
        expected_agents=["manager", "devsoft", "tester", "evaluator"],
    ),
    Case(
        scale="small",
        category="devweb",
        name="web_route_contract",
        setup=["Memory search endpoint responses should include score, subject, property and text."],
        task="Build the API route contract for memory search in the web layer.",
        expected_all=["route", "score", "subject", "property"],
        expected_agents=["manager", "devweb", "devsoft", "tester", "evaluator"],
    ),
    Case(
        scale="small",
        category="evaluator",
        name="honest_reporting",
        setup=["All benchmark reports must include failures honestly and should not hide weak categories."],
        task="Evaluate whether the new multi-agent architecture is ready for the tutor demo.",
        expected_all=["failures", "honestly"],
        expected_agents=["manager", "evaluator"],
    ),
    Case(
        scale="small",
        category="multi_session",
        name="second_session_project_rule",
        setup=["Session memory rule: user-facing demos should use colored terminal panels."],
        task="In a later session, what UI rule should the team keep for demos?",
        expected_all=["colored", "terminal panels"],
        expected_agents=["manager"],
    ),
]


MEDIUM_CASES: List[Case] = [
    Case(
        scale="medium",
        category="mixed_language_query",
        name="french_question_ollama_default",
        setup=["For now we stay on Ollama for tests; if we change LLM later, it must be better and free."],
        task="Quel provider LLM doit-on garder pour les tests actuels ?",
        expected_any=["Ollama"],
        expected_agents=["manager", "devops", "evaluator"],
    ),
    Case(
        scale="medium",
        category="hard_distractors",
        name="endpoint_real_convention_among_distractors",
        setup=[
            "Distractor: the coffee machine uses orange capsules.",
            "Distractor: the hotel room number was B-204.",
            "Project rule: new endpoints must include focused pytest tests.",
            "Distractor: the router sticker is stored in the drawer.",
        ],
        task="Create an implementation plan for a new endpoint and include the testing rule.",
        expected_all=["endpoint", "pytest"],
        forbidden=["coffee", "hotel", "router"],
        expected_agents=["manager", "devweb", "devsoft", "tester", "evaluator"],
    ),
    Case(
        scale="medium",
        category="role_routing",
        name="devops_only_installation",
        setup=["The project should document PowerShell commands for Windows users."],
        task="Prepare the installation and environment commands for the multi-agent demo.",
        expected_all=["Commands", "PowerShell"],
        expected_agents=["manager", "devops"],
    ),
    Case(
        scale="medium",
        category="role_routing",
        name="tester_evaluator_regression",
        setup=["Every agent benchmark should compare against a no-memory baseline and report failures."],
        task="Compare the new team result and write a regression testing plan.",
        expected_all=["no-memory", "failures", "benchmark"],
        expected_agents=["manager", "tester", "evaluator"],
    ),
    Case(
        scale="medium",
        category="multiple_updates",
        name="provider_policy_update",
        setup=[
            "The preferred LLM provider for agent tests is Gemini.",
            "The preferred LLM provider for agent tests is Ollama.",
        ],
        task="Which LLM provider should the prototype use for tests right now?",
        expected_any=["Ollama"],
        forbidden=["Gemini"],
        expected_agents=["manager", "devops", "evaluator"],
    ),
    Case(
        scale="medium",
        category="multiple_updates",
        name="testing_framework_update",
        setup=[
            "The project testing framework is unittest.",
            "The project testing framework is pytest.",
        ],
        task="Which testing framework should the team use for new tests?",
        expected_any=["pytest"],
        forbidden=["unittest"],
        expected_agents=["manager", "tester"],
    ),
    Case(
        scale="medium",
        category="profile_summary",
        name="team_policy_summary",
        setup=[
            "Manager policy: only the manager should write validated decisions into Titan memory.",
            "Tester policy: tests must run before benchmarks.",
            "Evaluator policy: conclusions must mention limitations.",
        ],
        task="Summarize the team policies for memory, tests and evaluation.",
        expected_all=["manager", "tests", "limitations"],
        expected_agents=["manager", "tester", "evaluator"],
    ),
    Case(
        scale="medium",
        category="forget",
        name="forget_old_secret_keep_project_rule",
        setup=[
            "Lucas Martin's temporary secret code is TEMP-9981.",
            "Project rule: keep Ollama configurable for tests.",
        ],
        task="Forget Lucas Martin's temporary secret code, then explain the remaining project rule.",
        expected_any=["Ollama", "configurable"],
        forbidden=["TEMP-9981"],
        expected_agents=["manager", "devops", "evaluator"],
    ),
]


STRESS_CASES: List[Case] = [
    Case(
        scale="stress",
        category="noisy_query",
        name="noisy_question_about_titan_memory",
        setup=["The shared memory layer for the team is Titan long-term memory."],
        task="Ignore noise !!! banana router sticker ### which memory layer does the team use?",
        expected_all=["Titan", "memory"],
        forbidden=["banana", "router"],
        expected_agents=["manager"],
    ),
    Case(
        scale="stress",
        category="long_distractor_context",
        name="many_project_distractors_keep_real_rule",
        setup=[
            *[f"Distractor note {i}: this random note is unrelated to the multi-agent architecture." for i in range(25)],
            "Critical convention: all new Python code must keep type hints and pytest tests.",
        ],
        task="Build a plan for a new Python memory adapter and keep the critical convention.",
        expected_all=["type hints", "pytest"],
        expected_agents=["manager", "devsoft", "tester", "evaluator"],
    ),
    Case(
        scale="stress",
        category="identity_collision",
        name="same_surname_agent_owners",
        setup=[
            "Sarah Martin owns the DevWeb dashboard task.",
            "Lucas Martin owns the DevOps deployment task.",
        ],
        task="Who owns the DevWeb dashboard task?",
        expected_any=["Sarah Martin"],
        forbidden=["Lucas Martin owns"],
        expected_agents=["manager", "devweb"],
    ),
    Case(
        scale="stress",
        category="identity_collision",
        name="same_surname_other_owner",
        setup=[
            "Sarah Martin owns the DevWeb dashboard task.",
            "Lucas Martin owns the DevOps deployment task.",
        ],
        task="Who owns the DevOps deployment task?",
        expected_any=["Lucas Martin"],
        forbidden=["Sarah Martin owns"],
        expected_agents=["manager", "devops"],
    ),
    Case(
        scale="stress",
        category="consolidation",
        name="consolidated_demo_rules",
        setup=[
            "Demo rule: terminal panels should stay colorful.",
            "Demo rule: the manager must show selected agents.",
            "Demo rule: benchmark results must be saved as CSV.",
        ],
        task="After consolidation, what demo rules should be preserved?",
        expected_all=["colorful", "selected agents", "CSV"],
        expected_agents=["manager", "tester", "evaluator"],
    ),
    Case(
        scale="stress",
        category="agent_selection",
        name="full_stack_task_all_roles",
        setup=["The project uses FastAPI, pytest, PowerShell commands and honest benchmark reports."],
        task="Create a full plan to add a FastAPI memory endpoint, implement it in Python, document setup commands, test it and evaluate the result.",
        expected_all=["FastAPI", "pytest", "Commands", "Evaluation"],
        expected_agents=["manager", "devweb", "devsoft", "devops", "tester", "evaluator"],
    ),
    Case(
        scale="stress",
        category="paraphrase_recall",
        name="paraphrased_best_architecture",
        setup=["The recommended architecture is a manager-led Agno multi-agent team with shared Titan memory."],
        task="What setup did we decide is the strongest for coordinating specialist agents with long-term recall?",
        expected_all=["manager", "Titan", "memory"],
        expected_agents=["manager", "evaluator"],
    ),
    Case(
        scale="stress",
        category="provider_future",
        name="free_llm_future_constraint",
        setup=["Future LLM replacement constraint: if Ollama is replaced, the new LLM should be better and free."],
        task="Evaluate the risk of replacing Ollama later and state the constraint.",
        expected_all=["better", "free"],
        expected_agents=["manager", "devops", "evaluator"],
    ),
]



def build_large_cases() -> List[Case]:
    """Build a larger deterministic stress suite.

    The large scale intentionally combines many facts, distractors, updates,
    identity collisions and consolidation scenarios.  It is still lightweight:
    the role agents are deterministic, and Titan is used as the shared memory
    backend.  This makes it suitable for local Windows/Ollama testing.
    """
    cases: List[Case] = []

    services = [
        ("billing", "FastAPI", "pytest", "type hints", ["manager", "devweb", "devsoft", "tester", "evaluator"]),
        ("analytics", "REST", "pytest", "typed schemas", ["manager", "devweb", "devsoft", "tester", "evaluator"]),
        ("memory-search", "FastAPI", "unit tests", "score field", ["manager", "devweb", "devsoft", "tester", "evaluator"]),
        ("deployment", "PowerShell", "environment variables", "Ollama", ["manager", "devops", "tester", "evaluator"]),
        ("benchmark", "CSV reports", "failure analysis", "honest reporting", ["manager", "tester", "evaluator"]),
        ("dashboard", "colored terminal panels", "selected agents", "memory context", ["manager", "devweb", "evaluator"]),
        ("adapter", "single-file Titan", "clean API", "save/load", ["manager", "devsoft", "tester", "evaluator"]),
        ("provider", "configurable LLM", "Ollama default", "free alternative later", ["manager", "devops", "evaluator"]),
        ("documentation", "limitations", "commands", "benchmark results", ["manager", "tester", "evaluator"]),
        ("evaluation", "baseline comparison", "stress categories", "risk summary", ["manager", "tester", "evaluator"]),
    ]

    for idx, (service, a, b, c, roles) in enumerate(services, start=1):
        setup = [
            f"Distractor {idx}-A: hotel room B-{200 + idx} is unrelated.",
            f"Distractor {idx}-B: coffee capsule color is orange.",
            f"Critical project convention for {service}: use {a}, {b}, and {c}.",
            f"Distractor {idx}-C: travel ad code TRAVEL-{idx} should be ignored.",
        ]
        cases.append(Case(
            scale="large",
            category="large_project_convention",
            name=f"large_convention_{service.replace('-', '_')}",
            setup=setup,
            task=f"Create a plan for the {service} work and preserve the critical convention.",
            expected_all=[str(a), str(b)],
            expected_any=[str(c)],
            forbidden=["hotel room", "coffee capsule", "TRAVEL"],
            expected_agents=roles,
        ))

    owners = [
        ("Sarah Martin", "DevWeb dashboard", "Lucas Martin", "DevOps deployment", "devweb"),
        ("Lucas Martin", "DevOps deployment", "Sarah Martin", "DevWeb dashboard", "devops"),
        ("Emma Laurent", "benchmark report", "Ethan Laurent", "database migration", "tester"),
        ("Ethan Laurent", "database migration", "Emma Laurent", "benchmark report", "devsoft"),
        ("Noah Bernard", "provider configuration", "Nina Bernard", "UI polish", "devops"),
        ("Nina Bernard", "UI polish", "Noah Bernard", "provider configuration", "devweb"),
        ("Hugo Morel", "risk evaluation", "Hugo Martin", "API route", "evaluator"),
        ("Hugo Martin", "API route", "Hugo Morel", "risk evaluation", "devweb"),
    ]
    for idx, (person, owned, other, other_owned, role) in enumerate(owners, start=1):
        cases.append(Case(
            scale="large",
            category="large_identity_collision",
            name=f"large_identity_collision_{idx}",
            setup=[
                f"{person} owns the {owned} task.",
                f"{other} owns the {other_owned} task.",
                "Distractor: both people attended the same meeting.",
            ],
            task=f"Who owns the {owned} task?",
            expected_any=[person],
            forbidden=[other],
            expected_agents=["manager", role],
        ))

    update_cases = [
        ("preferred backend language", ["C++", "Rust", "Python"], "Python", ["manager", "devsoft", "tester"]),
        ("preferred LLM provider for current tests", ["Gemini", "Claude", "Ollama"], "Ollama", ["manager", "devops", "evaluator"]),
        ("testing framework", ["unittest", "nose", "pytest"], "pytest", ["manager", "tester"]),
        ("demo UI style", ["plain text", "markdown only", "colored terminal panels"], "colored terminal panels", ["manager", "devweb", "evaluator"]),
        ("memory backend", ["short-term memory", "vector store", "shared Titan memory"], "shared Titan memory", ["manager", "devsoft", "evaluator"]),
        ("deployment shell", ["bash", "cmd", "PowerShell"], "PowerShell", ["manager", "devops"]),
        ("reporting rule", ["hide failures", "only show wins", "report failures honestly"], "report failures honestly", ["manager", "evaluator"]),
        ("benchmark output format", ["text only", "screenshots only", "CSV and markdown report"], "CSV and markdown report", ["manager", "tester", "evaluator"]),
    ]
    for idx, (prop, values, final, roles) in enumerate(update_cases, start=1):
        cases.append(Case(
            scale="large",
            category="large_multiple_updates",
            name=f"large_update_{idx}",
            setup=[f"The {prop} is {value}." for value in values],
            task=f"What is the current {prop}?",
            expected_any=[final],
            forbidden=[v for v in values[:-1] if v not in final],
            expected_agents=roles,
        ))

    for idx in range(1, 9):
        code = f"TEMP-LARGE-{3000 + idx}"
        rule = f"Large retention rule {idx}: keep benchmark category {idx} documented with limitations."
        cases.append(Case(
            scale="large",
            category="large_forget_retention",
            name=f"large_forget_keep_rule_{idx}",
            setup=[
                f"Lucas Martin's temporary secret code is {code}.",
                rule,
                f"Distractor: Lucas Martin's old favorite color was cyan-{idx}.",
            ],
            task="Forget Lucas Martin's temporary secret code, then explain the remaining project rule.",
            expected_all=["benchmark", "limitations"],
            forbidden=[code],
            expected_agents=["manager", "tester", "evaluator"],
        ))

    consolidation_sets = [
        ["terminal panels should be colorful", "manager must show selected agents", "results must be saved as CSV"],
        ["Ollama remains default", "future LLM must be free", "provider must stay configurable"],
        ["Tester runs unit tests first", "Evaluator reports limitations", "Manager stores validated decisions"],
        ["DevSoft preserves the single-file Titan model", "DevWeb documents route contracts", "DevOps writes PowerShell commands"],
        ["profile summaries need all active facts", "forget must be targeted", "consolidation rebuilds LTM"],
        ["large benchmarks need distractors", "stress tests need failures", "reports need honest conclusions"],
    ]
    for idx, rules in enumerate(consolidation_sets, start=1):
        cases.append(Case(
            scale="large",
            category="large_consolidation",
            name=f"large_consolidation_{idx}",
            setup=[f"Consolidation rule {idx}.{j}: {rule}." for j, rule in enumerate(rules, start=1)],
            task="After consolidation, summarize the preserved project rules.",
            expected_all=[rules[0].split()[0], rules[1].split()[0]],
            expected_any=[rules[2].split()[0]],
            expected_agents=["manager", "tester", "evaluator"],
        ))

    for idx in range(1, 9):
        cases.append(Case(
            scale="large",
            category="large_noisy_query",
            name=f"large_noisy_{idx}",
            setup=[
                f"Noise distractor {idx}: banana router sticker coffee hotel orange.",
                f"Critical memory {idx}: the team uses shared Titan memory for long-term project recall.",
            ],
            task=f"Ignore banana !!! router ### coffee ??? What memory does the team use for long-term recall in case {idx}?",
            expected_all=["Titan", "memory"],
            forbidden=["banana", "coffee", "hotel"],
            expected_agents=["manager"],
        ))

    for idx in range(1, 7):
        cases.append(Case(
            scale="large",
            category="large_profile_summary",
            name=f"large_profile_summary_{idx}",
            setup=[
                f"Profile {idx}: Emma Laurent works on database optimization.",
                f"Profile {idx}: Emma Laurent also works on mobile accessibility project.",
                f"Profile {idx}: Emma Laurent prefers concise technical reports.",
            ],
            task="Summarize Emma Laurent's project profile.",
            expected_all=["database optimization", "mobile accessibility project"],
            expected_any=["concise"],
            expected_agents=["manager", "evaluator"],
        ))

    return cases


LARGE_CASES = build_large_cases()

ALL_CASES = SMALL_CASES + MEDIUM_CASES + STRESS_CASES + LARGE_CASES


def contains_all(text: str, expected: Sequence[str]) -> bool:
    low = text.lower()
    return all(item.lower() in low for item in expected)


def contains_any(text: str, expected: Sequence[str]) -> bool:
    if not expected:
        return True
    low = text.lower()
    return any(item.lower() in low for item in expected)


def contains_forbidden(text: str, forbidden: Sequence[str]) -> List[str]:
    low = text.lower()
    return [item for item in forbidden if item.lower() in low]


def evaluate_answer(answer: str, case: Case) -> tuple[bool, str]:
    missing_all = [item for item in case.expected_all if item.lower() not in answer.lower()]
    any_ok = contains_any(answer, case.expected_any)
    bad = contains_forbidden(answer, case.forbidden)
    reasons = []
    if missing_all:
        reasons.append(f"missing_all={missing_all}")
    if not any_ok:
        reasons.append(f"missing_any={case.expected_any}")
    if bad:
        reasons.append(f"forbidden={bad}")
    return not reasons, "; ".join(reasons)


def evaluate_agents(selected: Sequence[str], expected: Sequence[str]) -> tuple[bool, str]:
    missing = [name for name in expected if name not in selected]
    if missing:
        return False, f"missing_agents={missing}"
    return True, ""


def run_no_memory(case: Case) -> tuple[str, List[str], float, float]:
    answer = "I do not know because no shared memory is available."
    return answer, [], 0.0, 0.0


def run_single_titan(case: Case, tmp: Path) -> tuple[str, List[str], float, float]:
    memory = TitanAgentMemory(memory_path=tmp / f"single_{case.scale}_{case.name}.pt")
    for fact in case.setup:
        memory.store(fact)
    if case.category == "consolidation":
        memory.consolidate(keep_existing_ltm=False)
    if case.category == "forget":
        # Keep this deterministic and aligned with the case name.
        memory.forget("Lucas Martin's temporary secret code")
    records = memory.recall(case.task, top_k=8, min_score=0.02)
    if records:
        answer = "\n".join(record.text for record in records)
    else:
        answer = "No relevant Titan memory found."
    return answer, ["single_titan_agent"], 0.0, 0.0


def run_multi_agent(case: Case, tmp: Path) -> tuple[str, List[str], float, float]:
    team = MultiAgentTitanTeam(memory_path=tmp / f"team_{case.scale}_{case.name}.pt", top_k=8)
    for fact in case.setup:
        team.store(fact)
    if case.category == "consolidation":
        team.consolidate()
    if case.category == "forget":
        team.forget("Lucas Martin's temporary secret code")
    result = team.run_task(case.task, store_decision=False)
    return result.final_answer, result.selected_agents, 0.0, 0.0


def selected_cases(scales: Sequence[str], categories: Sequence[str] | None) -> List[Case]:
    scale_set = set(scales)
    rows = [case for case in ALL_CASES if case.scale in scale_set]
    if categories:
        cat_set = set(categories)
        rows = [case for case in rows if case.category in cat_set]
    return rows


def pct(n: int, d: int) -> str:
    return "0.0%" if d == 0 else f"{n / d * 100:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", nargs="+", default=["small", "medium", "stress"], choices=["small", "medium", "stress", "large"])
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--agents", nargs="+", default=["no_memory", "single", "multi"], choices=["no_memory", "single", "multi"])
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--show-answers", action="store_true")
    parser.add_argument("--strict-agents", action="store_true", help="Fail multi-agent rows when expected specialist roles are missing.")
    args = parser.parse_args()

    cases = selected_cases(args.scales, args.categories)
    if not cases:
        print("No benchmark cases selected.")
        return 1

    agent_runners = []
    if "no_memory" in args.agents:
        agent_runners.append(("no_memory_agent", lambda c, tmp: run_no_memory(c)))
    if "single" in args.agents:
        agent_runners.append(("single_titan_agent", lambda c, tmp: run_single_titan(c, tmp)))
    if "multi" in args.agents:
        agent_runners.append(("multi_agent_titan_team", lambda c, tmp: run_multi_agent(c, tmp)))

    rows: List[Dict[str, str]] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        for case in cases:
            for agent_name, runner in agent_runners:
                answer, agents, store_time, query_time = runner(case, tmp)
                content_ok, reason = evaluate_answer(answer, case)
                agents_ok, agent_reason = evaluate_agents(agents, case.expected_agents) if agent_name == "multi_agent_titan_team" else (True, "")
                status_ok = content_ok and (agents_ok or not args.strict_agents)
                reasons = "; ".join(x for x in [reason, agent_reason if not agents_ok else ""] if x)
                rows.append({
                    "agent": agent_name,
                    "scale": case.scale,
                    "category": case.category,
                    "case": case.name,
                    "status": "PASS" if status_ok else "FAIL",
                    "content_ok": str(content_ok),
                    "agents_ok": str(agents_ok),
                    "selected_agents": ",".join(agents),
                    "expected_agents": ",".join(case.expected_agents),
                    "expected_all": "; ".join(case.expected_all),
                    "expected_any": "; ".join(case.expected_any),
                    "forbidden": "; ".join(case.forbidden),
                    "reason": reasons,
                    "answer": answer,
                    "store_time": f"{store_time:.4f}",
                    "query_time": f"{query_time:.4f}",
                })

    print("\n" + "=" * 96)
    print("DETAILED MULTI-AGENT TITAN TEAM COMPARISON")
    print("=" * 96)
    print(f"Cases: {len(cases)} | Total available: {len(ALL_CASES)} | Scales: {', '.join(args.scales)}")

    print("\nGLOBAL RESULTS")
    print("-" * 96)
    for agent in [name for name, _ in agent_runners]:
        subset = [r for r in rows if r["agent"] == agent]
        passed = sum(1 for r in subset if r["status"] == "PASS")
        total = len(subset)
        print(f"{agent:<24} | {passed:>3}/{total:<3} ({pct(passed, total):>6})")

    print("\nRESULTS BY SCALE")
    print("-" * 96)
    for agent in [name for name, _ in agent_runners]:
        for scale in args.scales:
            subset = [r for r in rows if r["agent"] == agent and r["scale"] == scale]
            if not subset:
                continue
            passed = sum(1 for r in subset if r["status"] == "PASS")
            print(f"{agent:<24} | {scale:<8} | {passed:>3}/{len(subset):<3} ({pct(passed, len(subset)):>6})")

    print("\nRESULTS BY CATEGORY")
    print("-" * 96)
    cats = sorted({r["category"] for r in rows})
    for agent in [name for name, _ in agent_runners]:
        for cat in cats:
            subset = [r for r in rows if r["agent"] == agent and r["category"] == cat]
            if not subset:
                continue
            passed = sum(1 for r in subset if r["status"] == "PASS")
            print(f"{agent:<24} | {cat:<24} | {passed:>3}/{len(subset):<3} ({pct(passed, len(subset)):>6})")

    print("\nRESULTS BY CASE")
    print("-" * 96)
    for r in rows:
        print(f"{r['agent']:<24} | {r['scale']:<8} | {r['category']:<22} | {r['case']:<40} | {r['status']}")

    failures = [r for r in rows if r["status"] == "FAIL"]
    if failures:
        print("\nFAILURES")
        print("-" * 96)
        for r in failures:
            print(f"{r['agent']} | {r['scale']} | {r['category']} | {r['case']} | {r['reason']} | answer={r['answer'][:220]!r}")

    if args.show_answers:
        print("\nANSWERS")
        print("-" * 96)
        for r in rows:
            print(f"\n[{r['agent']} / {r['scale']} / {r['case']} / {r['status']}]\nselected={r['selected_agents']}\n{r['answer']}")

    if args.save:
        out_dir = ROOT / "benchmark" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = out_dir / f"multi_agent_team_detailed_{stamp}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        md_path = out_dir / f"multi_agent_team_detailed_report_{stamp}.md"
        with md_path.open("w", encoding="utf-8") as f:
            f.write("# Detailed multi-agent Titan team comparison\n\n")
            f.write(f"Cases: {len(cases)} | Total available: {len(ALL_CASES)} | Scales: {', '.join(args.scales)}\n\n")
            f.write("## Global results\n\n")
            f.write("| Agent | Score |\n|---|---:|\n")
            for agent in [name for name, _ in agent_runners]:
                subset = [r for r in rows if r["agent"] == agent]
                passed = sum(1 for r in subset if r["status"] == "PASS")
                f.write(f"| {agent} | {passed}/{len(subset)} ({pct(passed, len(subset))}) |\n")

            f.write("\n## Results by scale\n\n")
            f.write("| Agent | Scale | Score |\n|---|---|---:|\n")
            for agent in [name for name, _ in agent_runners]:
                for scale in args.scales:
                    subset = [r for r in rows if r["agent"] == agent and r["scale"] == scale]
                    if not subset:
                        continue
                    passed = sum(1 for r in subset if r["status"] == "PASS")
                    f.write(f"| {agent} | {scale} | {passed}/{len(subset)} ({pct(passed, len(subset))}) |\n")

            f.write("\n## Results by category\n\n")
            f.write("| Agent | Category | Score |\n|---|---|---:|\n")
            for agent in [name for name, _ in agent_runners]:
                for cat in cats:
                    subset = [r for r in rows if r["agent"] == agent and r["category"] == cat]
                    if not subset:
                        continue
                    passed = sum(1 for r in subset if r["status"] == "PASS")
                    f.write(f"| {agent} | {cat} | {passed}/{len(subset)} ({pct(passed, len(subset))}) |\n")

            f.write("\n## Failures\n\n")
            if failures:
                for r in failures:
                    f.write(f"- **{r['agent']} / {r['scale']} / {r['case']}**: {r['reason']} — answer: `{r['answer'][:300]}`\n")
            else:
                f.write("No failures.\n")
        print(f"\nCSV saved to: {csv_path}")
        print(f"Report saved to: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
