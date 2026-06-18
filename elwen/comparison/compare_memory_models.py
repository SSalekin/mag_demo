#!/usr/bin/env python3
"""
Mamba vs Titan External Memory Comparison - v2
==============================================

Run from the project root:

    python elwen/comparison/compare_memory_models.py --scales small medium
    python elwen/comparison/compare_memory_models.py --scales small medium large

Main corrections compared with v1:
- Avoids false negatives where NEW-CODE-1022 was counted as containing old CODE-1022.
- Avoids false negatives where Java was counted inside JavaScript.
- In retrieval-only forgetting tests, success means the forgotten target fact is not retrieved.
  However, retrieving other secret codes for a secret-code question is still counted as an error.
- Synthetic data now uses full names in every fact, not only first names, to avoid artificial ambiguity.
- Medium/large synthetic first names are unique by default.
- Adds per-category summary and clearer failure details.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class TestCase:
    scale: str
    category: str
    name: str
    store: List[str]
    question: str
    expected_any: List[str]
    forbidden: List[str]
    forget: Optional[str] = None
    target_property: Optional[str] = None


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def contains_keyword(text: str, keyword: str) -> bool:
    """Whole-token keyword matching after normalization."""
    text_n = norm(text)
    key_n = norm(keyword)
    if not key_n:
        return False
    if " " in key_n:
        return key_n in text_n
    return re.search(rf"\b{re.escape(key_n)}\b", text_n) is not None


def remove_expected_values(answer: str, expected_any: List[str]) -> str:
    """
    Remove expected values before checking forbidden values.

    Example:
    - expected: NEW-CODE-1022
    - forbidden: CODE-1022

    Without this, the old code appears as a substring inside the new code.
    """
    cleaned = answer
    for expected in expected_any:
        cleaned = re.sub(re.escape(expected), " ", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace(norm(expected), " ")
    return cleaned


def answer_passes(
    answer: str,
    expected_any: List[str],
    forbidden: List[str],
    category: str,
    target_property: Optional[str] = None,
) -> Tuple[bool, str]:
    answer_n = norm(answer)

    no_memory_expected = any(
        norm(keyword) in {"do not know", "don t know", "not know", "no memory"}
        for keyword in expected_any
    )

    # Retrieval-only forget logic:
    # For a forgotten secret code, it is correct to retrieve no secret-code fact.
    # But retrieving another person's secret code for that question is still a retrieval error.
    if no_memory_expected:
        forbidden_hit = [word for word in forbidden if contains_keyword(answer, word)]
        if forbidden_hit:
            return False, f"forgotten target still retrieved: {forbidden_hit}"

        if target_property == "secret_code":
            if contains_keyword(answer, "secret code"):
                return False, "retrieved an irrelevant secret-code memory after forgetting target"
            return True, "ok (no secret-code memory retrieved)"

        if not answer_n:
            return True, "ok (empty retrieval)"

    expected_ok = any(contains_keyword(answer, keyword) for keyword in expected_any)

    answer_without_expected = remove_expected_values(answer, expected_any)
    forbidden_hit = [word for word in forbidden if contains_keyword(answer_without_expected, word)]

    if not expected_ok:
        return False, f"missing expected keyword(s): {expected_any}"
    if forbidden_hit:
        return False, f"contains forbidden keyword(s): {forbidden_hit}"
    return True, "ok"


# =============================================================================
# Test generation
# =============================================================================


def small_tests() -> List[TestCase]:
    return [
        TestCase(
            scale="small",
            category="simple_recall",
            name="simple_location",
            store=["Sarah Nguyen is a 23-year-old AI student from Hanoi, Vietnam."],
            question="Where is Sarah Nguyen from?",
            expected_any=["hanoi"],
            forbidden=["da nang", "danang", "rennes", "lyon"],
            target_property="location",
        ),
        TestCase(
            scale="small",
            category="atomic_update",
            name="study_retained_after_location_update",
            store=[
                "Sarah Nguyen is a 23-year-old AI student from Hanoi, Vietnam. Sarah Nguyen studies natural language processing and works on a sentiment analysis project.",
                "Actually, Sarah Nguyen is now from Da Nang, Vietnam.",
            ],
            question="What does Sarah Nguyen study?",
            expected_any=["natural language processing", "nlp"],
            forbidden=[],
            target_property="study",
        ),
        TestCase(
            scale="small",
            category="single_value_update",
            name="favorite_color_update",
            store=[
                "Sarah Nguyen's favorite color is yellow.",
                "Actually, Sarah Nguyen's favorite color is now black.",
            ],
            question="What is Sarah Nguyen's favorite color?",
            expected_any=["black"],
            forbidden=["yellow"],
            target_property="favorite_color",
        ),
        TestCase(
            scale="small",
            category="forget_secret_code",
            name="forget_secret_code",
            store=["Lina Moreau's secret code is LA-2026."],
            forget="Lina Moreau secret code",
            question="What is Lina Moreau's secret code?",
            expected_any=["do not know", "don't know", "not know", "no memory"],
            forbidden=["la-2026"],
            target_property="secret_code",
        ),
        TestCase(
            scale="small",
            category="multi_profile",
            name="multi_profile_separation",
            store=[
                "Emma Leclerc is a 21-year-old student from Nantes, France. Emma Leclerc studies computer science and loves robotics.",
                "Maya Dubois is a 23-year-old woman from Marseille, France. Maya Dubois speaks French, English, and Spanish.",
                "Kenji Tanaka is a 26-year-old software engineer from Osaka, Japan.",
            ],
            question="Which languages does Maya Dubois speak?",
            expected_any=["french", "english", "spanish"],
            forbidden=["osaka", "robotics", "nantes"],
            target_property="language",
        ),
        TestCase(
            scale="small",
            category="distractors",
            name="distractor_resistance",
            store=[
                "Sofia Garcia is interested in radiology, medical imaging, and AI-assisted diagnosis.",
                "Sofia Garcia's research group is called MedVision Lab.",
                "The weather in Da Nang is very sunny today.",
                "I ate chicken rice for lunch.",
                "The hotel lobby has a large blue sofa.",
                "There are many motorbikes near the city center.",
            ],
            question="What is Sofia Garcia's research group called?",
            expected_any=["medvision"],
            forbidden=["blue sofa", "chicken rice", "weather"],
            target_property="research_group",
        ),
    ]


FIRST_BASE = [
    "Alex", "Bella", "Carlos", "Diana", "Ethan", "Fatima", "Gabriel", "Hana",
    "Ivan", "Julia", "Karim", "Lena", "Marco", "Nora", "Owen", "Priya",
    "Quentin", "Rosa", "Samuel", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zane",
]
SUFFIXES = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Theta", "Lambda", "Sigma", "Omega"]
LAST_NAMES = [
    "Martin", "Nguyen", "Rossi", "Garcia", "Wilson", "Bernard", "Dubois", "Tanaka",
    "Morel", "Smith", "Brown", "Lopez", "Klein", "Meyer", "Singh", "Khan",
]
CITIES = [
    ("Hanoi", "Vietnam"), ("Da Nang", "Vietnam"), ("Rennes", "France"),
    ("Brest", "France"), ("Lyon", "France"), ("Madrid", "Spain"),
    ("Milan", "Italy"), ("Osaka", "Japan"), ("Manchester", "United Kingdom"),
    ("Toulouse", "France"), ("Nantes", "France"), ("Marseille", "France"),
]
STUDIES = [
    "natural language processing", "cybersecurity", "medical imaging",
    "UI design", "robotics", "cloud infrastructure", "data engineering",
    "sports analytics", "software engineering", "computer vision",
]
PROJECTS = [
    "a sentiment analysis project", "a phishing detection tool",
    "a mobile accessibility project", "a recommendation system",
    "a medical diagnosis assistant", "a cloud monitoring dashboard",
    "a robotic navigation system", "a heart-rate analysis tool",
]
COLORS = ["red", "blue", "green", "yellow", "black", "purple", "orange", "white"]
LANGUAGES = ["Python", "Scala", "Go", "JavaScript", "Rust", "Java", "C++"]


def unique_first(i: int) -> str:
    return FIRST_BASE[i % len(FIRST_BASE)] + SUFFIXES[(i // len(FIRST_BASE)) % len(SUFFIXES)]


def synthetic_profile(i: int) -> Dict[str, str]:
    first = unique_first(i)
    last = LAST_NAMES[(i * 3) % len(LAST_NAMES)]
    city, country = CITIES[(i * 5) % len(CITIES)]
    full = f"{first} {last}"
    return {
        "first": first,
        "last": last,
        "full": full,
        "city": city,
        "country": country,
        "study": STUDIES[(i * 7) % len(STUDIES)],
        "project": PROJECTS[(i * 11) % len(PROJECTS)],
        "color": COLORS[(i * 13) % len(COLORS)],
        "language": LANGUAGES[(i * 17) % len(LANGUAGES)],
        "code": f"CODE-{1000 + i}",
        "room": f"R{100 + i}",
    }


def profile_statement(p: Dict[str, str]) -> str:
    """Use full name in every fact to avoid benchmark-created ambiguity."""
    return (
        f"{p['full']} is a 24-year-old student from {p['city']}, {p['country']}. "
        f"{p['full']} studies {p['study']} and works on {p['project']}. "
        f"{p['full']}'s favorite color is {p['color']}. "
        f"{p['full']}'s favorite programming language is {p['language']}. "
        f"{p['full']}'s secret code is {p['code']}. "
        f"{p['full']}'s office room is {p['room']}."
    )


def distractor_statement(i: int) -> str:
    templates = [
        "The weather in Da Nang is very sunny today.",
        "The hotel lobby has a large blue sofa.",
        "I ate chicken rice for lunch.",
        "There are many motorbikes near the city center.",
        "Tomorrow I might go to the beach with friends.",
        "The coffee shop closes at 9 PM.",
        "A white cat was sleeping near the entrance.",
        "The elevator music was very calm.",
    ]
    return f"{templates[i % len(templates)]} Distractor note number {i}."


def synthetic_tests(scale: str, n_profiles: int, n_distractors: int) -> List[TestCase]:
    profiles = [synthetic_profile(i) for i in range(n_profiles)]
    all_stores = [profile_statement(p) for p in profiles]
    all_stores += [distractor_statement(i) for i in range(n_distractors)]

    target_a = profiles[min(n_profiles - 1, max(0, n_profiles // 4))]
    target_b = profiles[min(n_profiles - 1, max(0, n_profiles // 2))]
    target_c = profiles[min(n_profiles - 1, max(0, (3 * n_profiles) // 4))]
    target_d = profiles[min(n_profiles - 1, max(0, n_profiles - 2))]

    updated_city = "Rome"
    updated_color = "cyan"
    updated_lang = "Python" if target_b["language"].lower() != "python" else "Rust"
    updated_code = f"NEW-{target_c['code']}"

    return [
        TestCase(
            scale=scale,
            category="large_simple_recall",
            name=f"{scale}_location_recall",
            store=list(all_stores),
            question=f"Where is {target_a['full']} from?",
            expected_any=[target_a["city"].lower()],
            forbidden=["blue sofa", "chicken rice"],
            target_property="location",
        ),
        TestCase(
            scale=scale,
            category="large_update",
            name=f"{scale}_location_update",
            store=list(all_stores) + [f"Actually, {target_a['full']} is now from {updated_city}, Italy."],
            question=f"Where is {target_a['full']} from?",
            expected_any=[updated_city.lower()],
            forbidden=[target_a["city"].lower()],
            target_property="location",
        ),
        TestCase(
            scale=scale,
            category="large_update",
            name=f"{scale}_favorite_color_update",
            store=list(all_stores) + [f"Actually, {target_b['full']}'s favorite color is now {updated_color}."],
            question=f"What is {target_b['full']}'s favorite color?",
            expected_any=[updated_color.lower()],
            forbidden=[target_b["color"].lower()],
            target_property="favorite_color",
        ),
        TestCase(
            scale=scale,
            category="large_update",
            name=f"{scale}_favorite_language_update",
            store=list(all_stores) + [f"Actually, {target_b['full']}'s favorite programming language is now {updated_lang}."],
            question=f"What is {target_b['full']}'s favorite programming language?",
            expected_any=[updated_lang.lower()],
            forbidden=[target_b["language"].lower()] if target_b["language"].lower() != updated_lang.lower() else [],
            target_property="favorite_language",
        ),
        TestCase(
            scale=scale,
            category="large_secret_code",
            name=f"{scale}_secret_code_update",
            store=list(all_stores) + [f"Actually, {target_c['full']}'s secret code is now {updated_code}."],
            question=f"What is {target_c['full']}'s secret code?",
            expected_any=[updated_code.lower()],
            forbidden=[target_c["code"].lower()],
            target_property="secret_code",
        ),
        TestCase(
            scale=scale,
            category="large_forget_secret_code",
            name=f"{scale}_forget_secret_code",
            store=list(all_stores),
            forget=f"{target_d['full']} secret code",
            question=f"What is {target_d['full']}'s secret code?",
            expected_any=["do not know", "don't know", "not know", "no memory"],
            forbidden=[target_d["code"].lower()],
            target_property="secret_code",
        ),
        TestCase(
            scale=scale,
            category="large_retention_after_forget",
            name=f"{scale}_forget_target_only_retains_project",
            store=list(all_stores),
            forget=f"{target_d['full']} secret code",
            question=f"What does {target_d['full']} work on?",
            expected_any=[target_d["project"].lower().replace("a ", "")],
            forbidden=[target_d["code"].lower()],
            target_property="project",
        ),
        TestCase(
            scale=scale,
            category="large_distractors",
            name=f"{scale}_distractor_resistance",
            store=list(all_stores),
            question=f"What is {target_c['full']}'s office room?",
            expected_any=[target_c["room"].lower()],
            forbidden=["blue sofa", "chicken rice", "weather"],
            target_property="office_room",
        ),
    ]


def build_tests(args: argparse.Namespace) -> List[TestCase]:
    tests: List[TestCase] = []
    selected = set(args.scales)
    if "small" in selected:
        tests.extend(small_tests())
    if "medium" in selected:
        tests.extend(synthetic_tests("medium", args.medium_profiles, args.medium_distractors))
    if "large" in selected:
        tests.extend(synthetic_tests("large", args.large_profiles, args.large_distractors))
    return tests


# =============================================================================
# Model adapter
# =============================================================================


class ModelAdapter:
    def __init__(self, name: str, module, args: argparse.Namespace, memory_file: Path):
        self.name = name
        self.module = module
        self.args = args
        self.memory_file = memory_file
        self.memory = self._create_memory()

    def _create_memory(self):
        if self.name == "mamba":
            return self.module.MambaVectorMemory(
                hf_model=self.args.mamba_model,
                device=self.args.device,
                memory_path=self.memory_file,
                max_items=self.args.max_items,
            )
        if self.name == "titan":
            return self.module.TitanExternalMemory(
                memory_path=self.memory_file,
                d_model=self.args.titan_d_model,
                hidden_dim=self.args.titan_hidden_dim,
                max_items=self.args.max_items,
                device=self.args.device,
            )
        raise ValueError(f"Unknown model: {self.name}")

    def reset(self) -> None:
        if self.memory_file.exists():
            self.memory_file.unlink()
        self.memory.reset()

    def store_text(self, text: str) -> None:
        if hasattr(self.memory, "store_text"):
            self.memory.store_text(text)
        else:
            self.memory.store(text)

    def forget(self, query: str) -> None:
        candidates = self.memory.search_forget_candidates(query, k=1)
        if candidates:
            best = candidates[0][2]
            self.memory.deactivate_ids([best.id], reason=f"comparison benchmark v2 forget: {query}")

    def retrieve_text(self, question: str, topk: int, min_score: float) -> str:
        results = self.memory.retrieve(question, k=topk, min_score=min_score)
        return "\n".join(item.text for _, _, item in results)

    def ask(self, question: str, topk: int, min_score: float, ollama_model: str, debug: bool) -> str:
        return self.module.ask_with_memory(
            question,
            memory=self.memory,
            ollama_model=ollama_model,
            topk=topk,
            min_score=min_score,
            debug=debug,
            show_citations=False,
        )


# =============================================================================
# Benchmark runner
# =============================================================================


def run_one_test(adapter: ModelAdapter, test: TestCase, args: argparse.Namespace) -> Dict[str, object]:
    adapter.reset()

    store_start = time.perf_counter()
    for statement in test.store:
        adapter.store_text(statement)
    if test.forget:
        adapter.forget(test.forget)
    store_latency = time.perf_counter() - store_start

    query_start = time.perf_counter()
    if args.full_llm:
        answer = adapter.ask(
            test.question,
            topk=args.topk,
            min_score=args.min_score,
            ollama_model=args.ollama_model,
            debug=args.debug,
        )
    else:
        answer = adapter.retrieve_text(test.question, topk=args.topk, min_score=args.min_score)
    query_latency = time.perf_counter() - query_start

    passed, reason = answer_passes(answer, test.expected_any, test.forbidden, test.category, test.target_property)

    return {
        "model": adapter.name,
        "scale": test.scale,
        "category": test.category,
        "test": test.name,
        "passed": passed,
        "reason": reason,
        "store_latency_sec": store_latency,
        "query_latency_sec": query_latency,
        "active_memories": len(adapter.memory.active_items),
        "inactive_memories": len(adapter.memory.inactive_items),
        "question": test.question,
        "expected_any": json.dumps(test.expected_any, ensure_ascii=False),
        "forbidden": json.dumps(test.forbidden, ensure_ascii=False),
        "answer": answer.replace("\n", " | "),
    }


def summarize(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for model in sorted({str(row["model"]) for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        total = len(model_rows)
        passed = sum(1 for row in model_rows if row["passed"])
        summary[model] = {
            "total": total,
            "passed": passed,
            "accuracy": passed / max(1, total),
            "avg_store_latency": statistics.mean(float(row["store_latency_sec"]) for row in model_rows),
            "avg_query_latency": statistics.mean(float(row["query_latency_sec"]) for row in model_rows),
        }
    return summary


def group_summary(rows: List[Dict[str, object]], key: str) -> List[Dict[str, object]]:
    out = []
    for model in sorted({str(row["model"]) for row in rows}):
        for value in sorted({str(row[key]) for row in rows}):
            subset = [row for row in rows if row["model"] == model and str(row[key]) == value]
            if not subset:
                continue
            total = len(subset)
            passed = sum(1 for row in subset if row["passed"])
            out.append({
                "model": model,
                key: value,
                "passed": passed,
                "total": total,
                "accuracy": passed / max(1, total),
                "avg_store_latency": statistics.mean(float(row["store_latency_sec"]) for row in subset),
                "avg_query_latency": statistics.mean(float(row["query_latency_sec"]) for row in subset),
            })
    return out


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: List[Dict[str, object]], path: Path, args: argparse.Namespace) -> None:
    summary = summarize(rows)
    by_scale = group_summary(rows, "scale")
    by_category = group_summary(rows, "category")
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Mamba vs Titan Comparison Report v2")
    lines.append("")
    lines.append(f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Mode: {'full LLM with Ollama/Gemma' if args.full_llm else 'retrieval-only'}")
    lines.append(f"- Scales: {', '.join(args.scales)}")
    lines.append(f"- Top-k: {args.topk}")
    lines.append(f"- Min score: {args.min_score}")
    lines.append("")

    lines.append("## Global results")
    lines.append("")
    lines.append("| Model | Passed | Total | Accuracy | Avg store latency | Avg query latency |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model, s in summary.items():
        lines.append(
            f"| {model} | {int(s['passed'])} | {int(s['total'])} | {s['accuracy']*100:.1f}% | "
            f"{s['avg_store_latency']:.3f}s | {s['avg_query_latency']:.3f}s |"
        )
    lines.append("")

    lines.append("## Results by scale")
    lines.append("")
    lines.append("| Model | Scale | Passed | Total | Accuracy | Avg store latency | Avg query latency |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in by_scale:
        lines.append(
            f"| {row['model']} | {row['scale']} | {row['passed']} | {row['total']} | "
            f"{float(row['accuracy'])*100:.1f}% | {float(row['avg_store_latency']):.3f}s | "
            f"{float(row['avg_query_latency']):.3f}s |"
        )
    lines.append("")

    lines.append("## Results by category")
    lines.append("")
    lines.append("| Model | Category | Passed | Total | Accuracy | Avg store latency | Avg query latency |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in by_category:
        lines.append(
            f"| {row['model']} | {row['category']} | {row['passed']} | {row['total']} | "
            f"{float(row['accuracy'])*100:.1f}% | {float(row['avg_store_latency']):.3f}s | "
            f"{float(row['avg_query_latency']):.3f}s |"
        )
    lines.append("")

    failures = [row for row in rows if not row["passed"]]
    lines.append("## Failures")
    lines.append("")
    if not failures:
        lines.append("No failures.")
    else:
        for row in failures:
            lines.append(f"### {row['model']} - {row['scale']} - {row['test']}")
            lines.append(f"- Category: {row['category']}")
            lines.append(f"- Reason: {row['reason']}")
            lines.append(f"- Question: {row['question']}")
            lines.append(f"- Expected: `{row['expected_any']}`")
            lines.append(f"- Forbidden: `{row['forbidden']}`")
            lines.append(f"- Answer/retrieved text: `{row['answer']}`")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(rows: List[Dict[str, object]]) -> None:
    summary = summarize(rows)
    print("\n" + "=" * 80)
    print("GLOBAL RESULTS")
    print("=" * 80)
    for model, s in summary.items():
        print(
            f"{model.upper():<8} | "
            f"{int(s['passed'])}/{int(s['total'])} "
            f"({s['accuracy']*100:.1f}%) | "
            f"avg store={s['avg_store_latency']:.3f}s | "
            f"avg query={s['avg_query_latency']:.3f}s"
        )

    print("\nRESULTS BY SCALE")
    print("-" * 80)
    for row in group_summary(rows, "scale"):
        print(
            f"{row['model']:<8} | {row['scale']:<6} | "
            f"{row['passed']}/{row['total']} "
            f"({float(row['accuracy'])*100:.1f}%) | "
            f"store={float(row['avg_store_latency']):.3f}s | "
            f"query={float(row['avg_query_latency']):.3f}s"
        )

    print("\nRESULTS BY CATEGORY")
    print("-" * 80)
    for row in group_summary(rows, "category"):
        print(
            f"{row['model']:<8} | {row['category']:<32} | "
            f"{row['passed']}/{row['total']} "
            f"({float(row['accuracy'])*100:.1f}%)"
        )

    failures = [row for row in rows if not row["passed"]]
    if failures:
        print("\nFAILURES")
        print("-" * 80)
        for row in failures:
            print(f"{row['model']} | {row['scale']} | {row['test']} | {row['reason']}")
    else:
        print("\nNo failures.")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]

    parser = argparse.ArgumentParser(description="Compare Mamba and Titan external memory agents.")
    parser.add_argument("--models", nargs="+", default=["mamba", "titan"], choices=["mamba", "titan"])
    parser.add_argument("--scales", nargs="+", default=["small", "medium"], choices=["small", "medium", "large"])
    parser.add_argument("--full-llm", action="store_true", help="Call Ollama/Gemma for final answers. Slower.")
    parser.add_argument("--ollama-model", default="gemma2:2b")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.12)
    parser.add_argument("--max-items", type=int, default=5000)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--mamba-model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--titan-d-model", type=int, default=128)
    parser.add_argument("--titan-hidden-dim", type=int, default=256)
    parser.add_argument("--medium-profiles", type=int, default=30)
    parser.add_argument("--medium-distractors", type=int, default=60)
    parser.add_argument("--large-profiles", type=int, default=120)
    parser.add_argument("--large-distractors", type=int, default=240)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    mamba_path = project_root / "elwen" / "mamba" / "mamba_implementation.py"
    titan_path = project_root / "elwen" / "titan" / "titan_implementation.py"

    modules = {}
    if "mamba" in args.models:
        modules["mamba"] = load_module("mamba_impl_compare_v2", mamba_path)
    if "titan" in args.models:
        modules["titan"] = load_module("titan_impl_compare_v2", titan_path)

    output_dir = script_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    tests = build_tests(args)
    print("\nComparing memory models v2...")
    print(f"Models: {', '.join(args.models)}")
    print(f"Scales: {', '.join(args.scales)}")
    print(f"Mode: {'full LLM' if args.full_llm else 'retrieval-only'}")
    print(f"Total test cases per model: {len(tests)}\n")

    adapters: Dict[str, ModelAdapter] = {}
    for model in args.models:
        memory_file = output_dir / f"{model}_comparison_v2_memory_store.pt"
        adapters[model] = ModelAdapter(model, modules[model], args, memory_file)

    rows: List[Dict[str, object]] = []
    total_runs = len(args.models) * len(tests)
    run_idx = 0

    for test in tests:
        for model in args.models:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] {model} | {test.scale} | {test.name}")
            try:
                row = run_one_test(adapters[model], test, args)
            except Exception as exc:
                row = {
                    "model": model,
                    "scale": test.scale,
                    "category": test.category,
                    "test": test.name,
                    "passed": False,
                    "reason": f"exception: {exc}",
                    "store_latency_sec": 0.0,
                    "query_latency_sec": 0.0,
                    "active_memories": 0,
                    "inactive_memories": 0,
                    "question": test.question,
                    "expected_any": json.dumps(test.expected_any, ensure_ascii=False),
                    "forbidden": json.dumps(test.forbidden, ensure_ascii=False),
                    "answer": "",
                }
            rows.append(row)
            status = "PASS" if row["passed"] else "FAIL"
            print(
                f"  {status} | {row['reason']} | "
                f"store={float(row['store_latency_sec']):.3f}s | "
                f"query={float(row['query_latency_sec']):.3f}s"
            )
            if args.debug or not row["passed"]:
                print(f"  Answer/retrieved: {row['answer']}")
            print()

    csv_path = output_dir / f"comparison_v2_{timestamp}.csv"
    report_path = output_dir / f"comparison_v2_report_{timestamp}.md"
    write_csv(rows, csv_path)
    write_report(rows, report_path, args)

    print_summary(rows)
    print(f"\nCSV saved to: {csv_path}")
    print(f"Report saved to: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
