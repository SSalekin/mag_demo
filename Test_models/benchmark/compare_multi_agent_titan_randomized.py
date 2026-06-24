#!/usr/bin/env python3
"""Randomized adversarial benchmark for the Titan-backed multi-agent team.

This benchmark is intentionally different from the previous large and holdout
benchmarks.  It generates cases at runtime from a seed, so a high score on one
seed does not prove the system is perfect.  Run several seeds to check whether
performance generalizes.

Compared agents:
- no_memory_agent: baseline with no persistent memory.
- single_titan_agent: one Titan memory adapter, no specialist team.
- multi_agent_titan_team: manager-led team with shared Titan memory.

The benchmark focuses on failure modes found earlier:
- latest-value update handling;
- noisy queries and distractors;
- identity collisions;
- forget + retention;
- consolidation-style project rules;
- role routing for web/software/devops/testing/evaluation tasks.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.titan_agent_memory import TitanAgentMemory
from agents.multi_agent_titan_team import MultiAgentTitanTeam


@dataclass
class RandomCase:
    case_id: str
    scale: str
    category: str
    memories: List[str]
    task: str
    expected_all: List[str] = field(default_factory=list)
    expected_any: List[str] = field(default_factory=list)
    forbidden: List[str] = field(default_factory=list)
    required_agents: List[str] = field(default_factory=list)


@dataclass
class RunResult:
    agent: str
    case_id: str
    scale: str
    category: str
    passed: bool
    reason: str
    answer: str


FIRST_NAMES = [
    "Aline", "Axel", "Clara", "Clement", "Mila", "Mathis", "Lina", "Leo", "Nora", "Nathan",
    "Iris", "Ivan", "Yuna", "Yanis", "Elise", "Elias", "Manon", "Marius", "Camille", "Corentin",
]
LAST_NAMES = ["Durand", "Moreau", "Bernard", "Robert", "Garnier", "Lemoine", "Petit", "Roux", "Leroy", "Masson"]
MODULES = [
    "billing", "analytics", "audit-log", "inventory", "notification", "permission", "search", "calendar",
    "fraud", "profile", "reporting", "document", "pricing", "support", "export", "import",
]
CONVENTIONS = [
    ("FastAPI", "pytest", "type hints"),
    ("REST", "pytest", "strict request schemas"),
    ("webhooks", "regression tests", "retry policy"),
    ("RBAC", "unit tests", "typed policies"),
    ("API route", "integration tests", "rank score"),
    ("FastAPI", "unit tests", "error schema"),
    ("typed responses", "pytest", "OpenAPI docs"),
]
UPDATE_PROPS = [
    ("current preferred backend language", ["Go", "Rust", "Python"]),
    ("current LLM provider for tests", ["Gemini", "Claude", "Ollama"]),
    ("current testing framework", ["unittest", "nose", "pytest"]),
    ("current demo UI style", ["plain text", "markdown only", "colored terminal panels"]),
    ("current memory backend", ["vector store", "short-term memory", "shared Titan memory"]),
    ("current deployment shell", ["bash", "cmd", "PowerShell"]),
    ("current reporting rule", ["hide failures", "only show wins", "report failures honestly"]),
    ("current benchmark output format", ["screenshots only", "text only", "CSV and markdown report"]),
    # Properties that were not in the original large benchmark.  If these fail,
    # it suggests the manager update logic is still too hand-coded.
    ("current code formatter", ["black", "yapf", "ruff format"]),
    ("current API auth method", ["basic auth", "API key", "JWT"]),
    ("current cache eviction policy", ["random eviction", "oldest-first", "least recently used"]),
    ("current serialization format", ["XML", "pickle", "JSON"]),
    ("current queue backend", ["RabbitMQ", "Kafka", "Redis streams"]),
]
NOISE = ["banana", "coffee", "hotel", "router", "sticker", "capsule", "orange", "travel ad", "beach", "karaoke"]


def shuffled_copy(rng: random.Random, values: Sequence[str]) -> List[str]:
    values = list(values)
    rng.shuffle(values)
    return values


def person_pairs(rng: random.Random, count: int) -> List[Tuple[str, str, str, str]]:
    pairs = []
    for i in range(count):
        first1 = FIRST_NAMES[(2 * i) % len(FIRST_NAMES)]
        first2 = FIRST_NAMES[(2 * i + 1) % len(FIRST_NAMES)]
        last = LAST_NAMES[i % len(LAST_NAMES)]
        task1 = rng.choice(["release checklist", "schema migration", "load testing", "route review", "UI accessibility", "provider evaluation", "risk summary", "memory adapter", "security audit", "report export"])
        task2 = rng.choice(["database backup", "API contract", "benchmark report", "deployment script", "frontend polish", "monitoring setup", "documentation pass", "regression suite", "auth refactor", "pricing rules"])
        pairs.append((f"{first1} {last}", f"{first2} {last}", task1, task2))
    return pairs


def generate_cases(num_generated: int, seed: int) -> List[RandomCase]:
    rng = random.Random(seed)
    cases: List[RandomCase] = []

    # Fixed policy holdout cases with different wording from previous scripts.
    policy_memories = [
        "For the current prototype, Ollama is the default LLM provider; if we change later, the replacement must be better and free.",
        "Only the manager should write validated decisions into shared Titan memory.",
        "Before running benchmark comparisons, the tester must run focused unit tests.",
        "The demo interface should use colored terminal panels for readability.",
        "The preferred architecture is manager-led with shared Titan memory.",
    ]
    policy_queries = [
        ("Which LLM provider rule should we keep for tests?", ["Ollama", "better", "free"], ["manager", "devops", "evaluator"]),
        ("Who is allowed to write validated decisions into memory?", ["manager", "Titan memory"], ["manager", "evaluator"]),
        ("What must happen before benchmark comparisons?", ["test", "benchmark"], ["tester", "evaluator"]),
        ("What visual style should the demo use?", ["colored", "panels"], ["manager", "devsoft", "evaluator"]),
        ("What architecture is preferred for specialist coordination?", ["manager", "shared Titan memory"], ["manager", "evaluator"]),
    ]
    for i, (task, expected, agents) in enumerate(policy_queries, 1):
        cases.append(RandomCase(
            case_id=f"random_policy_{i}", scale="policy", category="policy_recall",
            memories=policy_memories, task=task, expected_all=expected, required_agents=agents,
        ))

    # Dynamic project conventions.
    n_each = max(1, num_generated // 7)
    for i in range(n_each):
        module = rng.choice(MODULES)
        c1, c2, c3 = rng.choice(CONVENTIONS)
        distractors = [
            f"Distractor for {module}: hotel room B-{rng.randint(100,999)} is unrelated.",
            f"Distractor for {module}: coffee capsule color is {rng.choice(['orange','blue','green'])}.",
            f"Distractor for {module}: travel ad code TRAVEL-{rng.randint(1000,9999)} should be ignored.",
        ]
        memory = f"Critical convention for {module}: use {c1}, {c2}, and {c3}."
        cases.append(RandomCase(
            case_id=f"random_convention_{i+1}_{module}", scale="generated", category="project_convention",
            memories=distractors + [memory],
            task=f"Create a plan for the {module} work and preserve the critical convention.",
            expected_all=[c1, c2], expected_any=[c3], forbidden=["hotel", "coffee", "TRAVEL"],
            required_agents=["manager", "devweb", "devsoft", "tester", "evaluator"],
        ))

    # Latest-value updates, including several properties not explicitly tuned.
    for i in range(n_each):
        prop, values = rng.choice(UPDATE_PROPS)
        values = shuffled_copy(rng, values)
        final = values[-1]
        old = values[:-1]
        memories = [f"The {prop} is {v}." for v in values]
        task = f"What is the {prop} right now?"
        cases.append(RandomCase(
            case_id=f"random_update_{i+1}_{re.sub(r'[^a-z0-9]+','_',prop.lower()).strip('_')}",
            scale="generated", category="latest_value_update",
            memories=memories, task=task, expected_any=[final], forbidden=old,
            required_agents=["manager", "evaluator"],
        ))

    # Identity collision.
    for i, (p1, p2, t1, t2) in enumerate(person_pairs(rng, n_each), 1):
        memories = [
            f"{p1} owns the {t1} task.",
            f"{p2} owns the {t2} task.",
            f"Distractor: {p1} and {p2} attended the same standup meeting.",
        ]
        cases.append(RandomCase(
            case_id=f"random_owner_{i}_a", scale="generated", category="identity_collision",
            memories=memories, task=f"Who owns the {t1} task?", expected_any=[p1], forbidden=[p2, t2, "standup"],
            required_agents=["manager", "evaluator"],
        ))
        cases.append(RandomCase(
            case_id=f"random_owner_{i}_b", scale="generated", category="identity_collision",
            memories=memories, task=f"Who owns the {t2} task?", expected_any=[p2], forbidden=[p1, t1, "standup"],
            required_agents=["manager", "evaluator"],
        ))

    # Noisy query.
    for i in range(n_each):
        key = f"QZ-{seed % 100}-{i:03d}"
        noise_text = " ".join(shuffled_copy(rng, NOISE)[:6])
        memories = [
            f"Noise distractor {i}: {noise_text} should be ignored.",
            f"Critical incident memory {i}: the escalation key is {key}.",
        ]
        task = f"Ignore {' !!! '.join(shuffled_copy(rng, NOISE)[:4])} ### what is the escalation key for incident {i}?"
        cases.append(RandomCase(
            case_id=f"random_noisy_{i+1}", scale="generated", category="noisy_query",
            memories=memories, task=task, expected_any=[key], forbidden=NOISE,
            required_agents=["manager", "evaluator"],
        ))

    # Forget + retention.
    for i in range(n_each):
        secret = f"TMP-RAND-{seed % 100}-{i:03d}"
        memories = [
            f"Lucas Martin's temporary secret code for random case {i} is {secret}.",
            f"Retention rule for random case {i}: benchmark reports must mention limitations and next steps.",
            f"Distractor: Lucas Martin's archived color was cyan-{i}.",
        ]
        cases.append(RandomCase(
            case_id=f"random_forget_retention_{i+1}", scale="generated", category="forget_retention",
            memories=memories, task="Forget Lucas Martin's temporary secret code, then explain the remaining retention rule.",
            expected_all=["limitations", "next steps"], forbidden=[secret, f"cyan-{i}"],
            required_agents=["manager", "tester", "evaluator"],
        ))

    # Consolidation-like cases.
    consolidation_rules = [
        ("terminal", "manager", "results"),
        ("Ollama", "future", "provider"),
        ("Tester", "Evaluator", "Manager"),
        ("DevSoft", "DevWeb", "DevOps"),
        ("profile", "forget", "consolidation"),
        ("large", "stress", "reports"),
    ]
    for i in range(n_each):
        terms = consolidation_rules[i % len(consolidation_rules)]
        memories = [
            "Please consolidate the active project rules after storing them.",
            f"Consolidation random rule {i}: preserve {terms[0]}, {terms[1]}, and {terms[2]}.",
        ]
        cases.append(RandomCase(
            case_id=f"random_consolidation_{i+1}", scale="generated", category="consolidation",
            memories=memories, task="After consolidation, what project rule should be preserved?",
            expected_all=[terms[0], terms[1]], expected_any=[terms[2]], forbidden=["Please consolidate"],
            required_agents=["manager", "devsoft", "tester", "evaluator"],
        ))

    # Role routing tasks.
    role_cases = [
        ("Build a FastAPI route, implement typed Python logic, document PowerShell setup, write pytest tests, and evaluate the risk.",
         ["FastAPI", "pytest", "PowerShell", "risk"], ["manager", "devweb", "devsoft", "devops", "tester", "evaluator"]),
        ("Prepare a provider setup plan for local Ollama now and a better free LLM later.",
         ["Ollama", "free", "provider"], ["manager", "devops", "evaluator"]),
        ("Create a benchmark report comparing baseline failures and summarize limitations.",
         ["benchmark", "failures", "limitations"], ["manager", "tester", "evaluator"]),
    ]
    for i in range(n_each):
        task, expected, agents = rng.choice(role_cases)
        memories = [
            "Team rule: DevWeb handles API routes, DevSoft handles Python internals, DevOps handles commands, Tester handles benchmarks, Evaluator reports limitations.",
            "Provider rule: Ollama is default for current tests; future replacements must be better and free.",
        ]
        cases.append(RandomCase(
            case_id=f"random_role_{i+1}", scale="generated", category="role_routing",
            memories=memories, task=task, expected_all=expected[:2], expected_any=expected[2:], required_agents=agents,
        ))

    rng.shuffle(cases)
    return cases[:num_generated]


def no_memory_answer(case: RandomCase) -> str:
    return "I do not know because no shared memory is available."


def single_titan_answer(case: RandomCase) -> str:
    with tempfile.TemporaryDirectory(prefix="single_titan_random_") as tmp:
        mem = TitanAgentMemory(memory_path=Path(tmp) / "single.pt", top_k=8, min_score=0.0)
        for fact in case.memories:
            mem.store(fact)
        if "forget" in case.task.lower() and "temporary secret code" in case.task.lower():
            mem.forget("Lucas Martin temporary secret code")
        if "consolidat" in case.task.lower():
            mem.consolidate(keep_existing_ltm=False)
        records = mem.recall(case.task, top_k=8, min_score=0.0)
        if not records:
            return "No relevant Titan memory found."
        return "\n".join(r.text for r in records[:5])


def multi_agent_answer(case: RandomCase) -> str:
    with tempfile.TemporaryDirectory(prefix="multi_titan_random_") as tmp:
        team = MultiAgentTitanTeam(memory_path=Path(tmp) / "team.pt", top_k=8)
        for fact in case.memories:
            team.store(fact)
        if "consolidat" in case.task.lower():
            # Keep this explicit to verify consolidated/project-rule retrieval.
            team.consolidate()
        return team.run_task(case.task).final_answer


def check_case(agent: str, case: RandomCase, answer: str, strict_agents: bool = False) -> Tuple[bool, str]:
    low = answer.lower()
    missing_all = [x for x in case.expected_all if x.lower() not in low]
    if missing_all:
        return False, f"missing_all={missing_all}"
    if case.expected_any and not any(x.lower() in low for x in case.expected_any):
        return False, f"missing_any={case.expected_any}"
    bad = [x for x in case.forbidden if x and x.lower() in low]
    if bad:
        return False, f"forbidden={bad}"
    if agent == "multi_agent_titan_team" and strict_agents and case.required_agents:
        missing_agents = [a for a in case.required_agents if a.lower() not in low]
        if missing_agents:
            return False, f"missing_agents={missing_agents}"
    return True, "PASS"


def run_benchmark(cases: Sequence[RandomCase], agents: Sequence[str], strict_agents: bool = False, show_answers: bool = False) -> List[RunResult]:
    runners = {
        "no_memory_agent": no_memory_answer,
        "single_titan_agent": single_titan_answer,
        "multi_agent_titan_team": multi_agent_answer,
    }
    results: List[RunResult] = []
    total = len(cases) * len(agents)
    idx = 0
    for case in cases:
        for agent in agents:
            idx += 1
            answer = runners[agent](case)
            passed, reason = check_case(agent, case, answer, strict_agents=strict_agents)
            results.append(RunResult(agent, case.case_id, case.scale, case.category, passed, reason, answer))
            if show_answers:
                print(f"[{idx}/{total}] {agent} | {case.category} | {case.case_id} | {'PASS' if passed else 'FAIL'} | {reason}")
                print(answer[:700].replace("\n", " "))
                print("-" * 80)
    return results


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({(100*n/d if d else 0):.1f}%)"


def summarize(results: Sequence[RunResult], seed: int, num_cases: int) -> str:
    lines: List[str] = []
    lines.append("=" * 92)
    lines.append("RANDOMIZED ADVERSARIAL MULTI-AGENT TITAN TEAM COMPARISON")
    lines.append("=" * 92)
    lines.append(f"Cases: {num_cases} | Seed: {seed}")
    lines.append("")
    lines.append("GLOBAL RESULTS")
    lines.append("-" * 92)
    by_agent: Dict[str, List[RunResult]] = {}
    for r in results:
        by_agent.setdefault(r.agent, []).append(r)
    for agent, rows in by_agent.items():
        ok = sum(r.passed for r in rows)
        lines.append(f"{agent:<24} | {pct(ok, len(rows))}")
    lines.append("")
    lines.append("RESULTS BY CATEGORY")
    lines.append("-" * 92)
    for agent in by_agent:
        cats: Dict[str, List[RunResult]] = {}
        for r in by_agent[agent]:
            cats.setdefault(r.category, []).append(r)
        for cat, rows in sorted(cats.items()):
            ok = sum(r.passed for r in rows)
            lines.append(f"{agent:<24} | {cat:<24} | {pct(ok, len(rows))}")
    lines.append("")
    lines.append("FAILURES")
    lines.append("-" * 92)
    failures = [r for r in results if not r.passed]
    if not failures:
        lines.append("No failures.")
    else:
        for r in failures[:160]:
            ans = re.sub(r"\s+", " ", r.answer).strip()
            if len(ans) > 360:
                ans = ans[:360] + "..."
            lines.append(f"- {r.agent} / {r.category} / {r.case_id}: {r.reason} — answer=`{ans}`")
        if len(failures) > 160:
            lines.append(f"... {len(failures) - 160} more failures omitted from console report.")
    return "\n".join(lines)


def save_outputs(results: Sequence[RunResult], report: str, seed: int) -> Tuple[Path, Path]:
    out_dir = ROOT / "benchmark" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"multi_agent_team_randomized_{stamp}_seed{seed}.csv"
    md_path = out_dir / f"multi_agent_team_randomized_report_{stamp}_seed{seed}.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["agent", "scale", "category", "case_id", "passed", "reason", "answer"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "agent": r.agent,
                "scale": r.scale,
                "category": r.category,
                "case_id": r.case_id,
                "passed": int(r.passed),
                "reason": r.reason,
                "answer": r.answer,
            })
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Randomized adversarial multi-agent Titan team comparison\n\n")
        f.write(report)
        f.write("\n")
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run randomized adversarial multi-agent Titan benchmark.")
    parser.add_argument("--num-cases", type=int, default=150, help="Number of generated cases. Default: 150")
    parser.add_argument("--seed", type=int, default=20260624, help="Random seed. Change this to generate a new holdout.")
    parser.add_argument("--agents", nargs="+", default=["no_memory_agent", "single_titan_agent", "multi_agent_titan_team"],
                        choices=["no_memory_agent", "single_titan_agent", "multi_agent_titan_team"])
    parser.add_argument("--strict-agents", action="store_true", help="Require the multi-agent answer to include expected specialist names.")
    parser.add_argument("--show-answers", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    cases = generate_cases(num_generated=args.num_cases, seed=args.seed)
    results = run_benchmark(cases, args.agents, strict_agents=args.strict_agents, show_answers=args.show_answers)
    report = summarize(results, seed=args.seed, num_cases=len(cases))
    print(report)
    if args.save:
        csv_path, md_path = save_outputs(results, report, seed=args.seed)
        print(f"\nCSV saved to: {csv_path}")
        print(f"Report saved to: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
