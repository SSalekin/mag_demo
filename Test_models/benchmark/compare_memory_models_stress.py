#!/usr/bin/env python3
"""
Stress benchmark for Mamba vs Titan external memory models.

Run from the project root:

    python elwen/comparison/compare_memory_models_stress.py --scales small medium
    python elwen/comparison/compare_memory_models_stress.py --scales large --large-profiles 120 --large-distractors 300

This benchmark is harder than the baseline:
- Top-1 / Top-3 / Top-5 retrieval accuracy
- paraphrased questions
- noisy questions
- several successive updates
- people with similar names
- same first-name collisions
- hard distractors that look like useful memories
- forgetting + retention after forgetting
- mixed French/English questions

Use retrieval-only mode for pure memory comparison.
Use --full-llm only for secondary chatbot-quality testing.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import re
import statistics
import sys
import time
import os
from dataclasses import dataclass, field
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def contains(text: str, keyword: str) -> bool:
    t = norm(text)
    k = norm(keyword)
    if not k:
        return False
    if " " in k:
        return k in t
    return re.search(rf"\b{re.escape(k)}\b", t) is not None


def remove_expected(text: str, expected: List[str]) -> str:
    cleaned = str(text)
    for value in expected:
        cleaned = re.sub(re.escape(value), " ", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace(norm(value), " ")
    return cleaned


def evaluate_text(text: str, expected_any: List[str], expected_all: List[str], forbidden: List[str], category: str) -> Tuple[bool, str]:
    no_memory_expected = any(norm(x) in {"do not know", "don t know", "not know", "no memory"} for x in expected_any + expected_all)

    if no_memory_expected:
        bad = [x for x in forbidden if contains(text, x)]
        if bad:
            return False, f"contains forgotten value(s): {bad}"
        if "forget" in category and not contains(text, "secret code"):
            return True, "ok: no secret-code memory retrieved"
        if not norm(text):
            return True, "ok: empty retrieval"

    if expected_any and not any(contains(text, x) for x in expected_any):
        return False, f"missing one of expected_any: {expected_any}"

    missing = [x for x in expected_all if not contains(text, x)]
    if missing:
        return False, f"missing expected_all: {missing}"

    cleaned = remove_expected(text, expected_any + expected_all)
    bad = [x for x in forbidden if contains(cleaned, x)]
    if bad:
        return False, f"contains forbidden value(s): {bad}"

    return True, "ok"


@dataclass
class TestCase:
    scale: str
    category: str
    name: str
    store: List[str]
    question: str
    expected_any: List[str] = field(default_factory=list)
    expected_all: List[str] = field(default_factory=list)
    forbidden: List[str] = field(default_factory=list)
    forget: Optional[str] = None
    notes: str = ""


FIRST = ["Alex", "Bella", "Carlos", "Diana", "Ethan", "Fatima", "Gabriel", "Hana", "Ivan", "Julia", "Karim", "Lena", "Marco", "Nora", "Owen", "Priya", "Quentin", "Rosa", "Samuel", "Tara", "Uma", "Victor", "Wendy", "Xavier", "Yara", "Zane"]
SUFFIX = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Theta", "Lambda", "Sigma", "Omega", "Nova", "Orion", "Pixel", "Vector", "Quantum", "Matrix"]
LAST = ["Martin", "Nguyen", "Rossi", "Garcia", "Wilson", "Bernard", "Dubois", "Tanaka", "Morel", "Smith", "Brown", "Lopez", "Klein", "Meyer", "Singh", "Khan"]
CITIES = [("Hanoi", "Vietnam"), ("Da Nang", "Vietnam"), ("Rennes", "France"), ("Brest", "France"), ("Lyon", "France"), ("Madrid", "Spain"), ("Milan", "Italy"), ("Osaka", "Japan"), ("Manchester", "United Kingdom"), ("Toulouse", "France"), ("Nantes", "France"), ("Marseille", "France"), ("Rome", "Italy"), ("Dublin", "Ireland"), ("Edinburgh", "Scotland")]
STUDIES = ["natural language processing", "cybersecurity", "medical imaging", "UI design", "robotics", "cloud infrastructure", "data engineering", "sports analytics", "software engineering", "computer vision", "database optimization", "human-computer interaction"]
PROJECTS = ["a sentiment analysis project", "a phishing detection tool", "a mobile accessibility project", "a recommendation system", "a medical diagnosis assistant", "a cloud monitoring dashboard", "a robotic navigation system", "a heart-rate analysis tool", "an image classification pipeline", "a multilingual chatbot", "a database indexing experiment", "a gesture recognition prototype"]
COLORS = ["red", "blue", "green", "yellow", "black", "purple", "orange", "white", "cyan", "turquoise"]
LANGS = ["Python", "Scala", "Go", "JavaScript", "Rust", "Java", "C++", "TypeScript"]


def unique_first(i: int) -> str:
    return FIRST[i % len(FIRST)] + SUFFIX[(i // len(FIRST)) % len(SUFFIX)]


def make_profile(i: int) -> Dict[str, str]:
    city, country = CITIES[(i * 5) % len(CITIES)]
    first = unique_first(i)
    last = LAST[(i * 7) % len(LAST)]
    return {
        "first": first,
        "last": last,
        "full": f"{first} {last}",
        "city": city,
        "country": country,
        "study": STUDIES[(i * 11) % len(STUDIES)],
        "project": PROJECTS[(i * 13) % len(PROJECTS)],
        "color": COLORS[(i * 17) % len(COLORS)],
        "lang": LANGS[(i * 19) % len(LANGS)],
        "code": f"CODE-{3000 + i}",
        "room": f"R{400 + i}",
    }


def profile_memory(p: Dict[str, str]) -> str:
    return (
        f"{p['full']} is a 24-year-old student from {p['city']}, {p['country']}. "
        f"{p['full']} studies {p['study']} and works on {p['project']}. "
        f"{p['full']}'s favorite color is {p['color']}. "
        f"{p['full']}'s favorite programming language is {p['lang']}. "
        f"{p['full']}'s secret code is {p['code']}. "
        f"{p['full']}'s office room is {p['room']}."
    )


def hard_distractor(i: int, profiles: List[Dict[str, str]]) -> str:
    p = profiles[i % len(profiles)]
    q = profiles[(i * 7 + 3) % len(profiles)]
    items = [
        f"The hotel has a room named {p['room']} but it is not an office.",
        f"The color {p['color']} appears in a poster near the lab.",
        f"A note says that {q['last']} is a common surname in the dataset.",
        f"The code CODE-{9000+i} is printed on a router sticker.",
        f"{q['first']} saw a blue sofa near the coffee machine.",
        f"The word {p['lang']} appears in an unrelated programming book title.",
        f"{q['city']} is mentioned in a travel advertisement.",
        f"A fake project named {p['project']} was discussed in a meeting, but not by {p['full']}.",
    ]
    return items[i % len(items)]


def base_dataset(n_profiles: int, n_distractors: int) -> Tuple[List[Dict[str, str]], List[str]]:
    profiles = [make_profile(i) for i in range(n_profiles)]
    stores = [profile_memory(p) for p in profiles]
    stores += [hard_distractor(i, profiles) for i in range(n_distractors)]
    return profiles, stores


def same_first_name_case() -> List[str]:
    return [
        "Sarah Nguyen is a 23-year-old AI student from Hanoi, Vietnam.",
        "Sarah Martin is a 25-year-old cybersecurity engineer from Rennes, France.",
        "Sarah Wilson is a 22-year-old design student from Milan, Italy.",
        "Sarah Nguyen's favorite color is purple.",
        "Sarah Martin's favorite color is green.",
        "Sarah Wilson's favorite color is orange.",
    ]


def build_scale_tests(scale: str, n_profiles: int, n_distractors: int) -> List[TestCase]:
    profiles, stores = base_dataset(n_profiles, n_distractors)

    a = profiles[n_profiles // 7]
    b = profiles[n_profiles // 3]
    c = profiles[n_profiles // 2]
    d = profiles[(2 * n_profiles) // 3]
    e = profiles[n_profiles - 3]

    b_color_1, b_color_2 = "magenta", "turquoise"
    c_lang_1 = "Rust" if c["lang"] != "Rust" else "Python"
    c_lang_2 = "Python" if c_lang_1 != "Python" else "Go"
    d_code_1 = f"TEMP-{d['code']}"
    d_code_2 = f"FINAL-{d['code']}"

    return [
        TestCase(scale, "paraphrase_recall", f"{scale}_paraphrased_location", list(stores),
                 f"In which city is {a['full']} currently based?", [a["city"]], [], ["blue sofa", "router sticker", "travel advertisement"]),

        TestCase(scale, "noisy_query", f"{scale}_noisy_office_room", list(stores),
                 f"hey, remind me plz, what's the office room for {a['full']}??", [a["room"]], [], ["hotel has a room"]),

        TestCase(scale, "multiple_updates", f"{scale}_multiple_color_updates",
                 list(stores) + [f"Actually, {b['full']}'s favorite color is now {b_color_1}.", f"Correction: {b['full']}'s favorite color is now {b_color_2}."],
                 f"What is {b['full']}'s favorite color?", [b_color_2], [], [b["color"], b_color_1]),

        TestCase(scale, "multiple_updates", f"{scale}_multiple_language_updates",
                 list(stores) + [f"Actually, {c['full']}'s favorite programming language is now {c_lang_1}.", f"Update: {c['full']}'s favorite programming language is now {c_lang_2}."],
                 f"What programming language does {c['full']} prefer now?", [c_lang_2], [], [c["lang"], c_lang_1]),

        TestCase(scale, "multiple_updates", f"{scale}_multiple_secret_code_updates",
                 list(stores) + [f"Actually, {d['full']}'s secret code is now {d_code_1}.", f"Correction: {d['full']}'s secret code is now {d_code_2}."],
                 f"What is {d['full']}'s latest secret code?", [d_code_2], [], [d["code"], d_code_1]),

        TestCase(scale, "forget_with_nearby_distractors", f"{scale}_forget_secret_code_no_other_secret_code",
                 list(stores), f"What is {e['full']}'s secret code?",
                 ["do not know", "don't know", "not know", "no memory"], [], [e["code"]], forget=f"{e['full']} secret code"),

        TestCase(scale, "retention_after_forget", f"{scale}_retains_project_after_forget",
                 list(stores), f"What does {e['full']} work on?",
                 [e["project"].replace("a ", "")], [], [e["code"]], forget=f"{e['full']} secret code"),

        TestCase(scale, "hard_distractors", f"{scale}_hard_distractor_room",
                 list(stores), f"What is the real office room of {d['full']}?", [d["room"]], [], ["hotel has a room"]),

        TestCase(scale, "mixed_language_query", f"{scale}_mixed_language_location",
                 list(stores), f"En quelle ville does {c['full']} live / is based?", [c["city"]], [], ["travel advertisement", "blue sofa"]),

        TestCase(scale, "profile_summary", f"{scale}_profile_summary",
                 list(stores), f"Tell me the key things you know about {a['full']}.",
                 [], [a["city"], a["study"], a["project"].replace("a ", "")], ["router sticker", "blue sofa"]),

        TestCase(scale, "identity_collision", f"{scale}_same_first_name_full_identity",
                 same_first_name_case() + list(stores[: min(20, len(stores))]),
                 "What is Sarah Nguyen's favorite color?", ["purple"], [], ["green", "orange"]),
    ]


def build_tests(args) -> List[TestCase]:
    tests: List[TestCase] = []
    if "small" in args.scales:
        tests += build_scale_tests("small", args.small_profiles, args.small_distractors)
    if "medium" in args.scales:
        tests += build_scale_tests("medium", args.medium_profiles, args.medium_distractors)
    if "large" in args.scales:
        tests += build_scale_tests("large", args.large_profiles, args.large_distractors)
    if args.shuffle:
        random.Random(args.seed).shuffle(tests)
    return tests



class Adapter:
    def __init__(self, name: str, args):
        self.name = name
        self.args = args
        self.model = self._create_model()

    def _create_model(self):
        if self.name == "transformer":
            from models.transformer_model import TransformerModel
            return TransformerModel(model_name=self.args.ollama_model, max_capacity=self.args.capacity)
        elif self.name == "lstm":
            from models.lstm_model import LstmModel
            return LstmModel(model_name=self.args.ollama_model, max_capacity=self.args.capacity)
        elif self.name == "gru":
            from models.gru_model import GruModel
            return GruModel(model_name=self.args.ollama_model, max_capacity=self.args.capacity)
        elif self.name == "gnn":
            from models.gnn_model import GnnModel
            return GnnModel(model_name=self.args.ollama_model, max_capacity=self.args.capacity)
        elif self.name == "mamba":
            from models.mamba_model import MambaModel
            return MambaModel(model_name=self.args.ollama_model, max_capacity=self.args.capacity)
        elif self.name == "titan":
            from models.titan_model import TitanModel
            return TitanModel(model_name=self.args.ollama_model, max_capacity=self.args.capacity)
        raise ValueError(f"Unknown model: {self.name}")

    def reset(self):
        self.model.clear_memory()

    def store_text(self, text: str):
        self.model.add_user_message(f"Please remember this information: {text}")
        self.model.add_assistant_message("Understood. I will remember that.")

    def forget(self, query: str):
        self.model.add_user_message(f"Please forget this: {query}")
        self.model.add_assistant_message("Acknowledged. I have forgotten it.")

    def ask(self, question: str, topk: int, min_score: float, ollama_model: str, debug: bool) -> str:
        self.model.add_user_message(question)
        answer_chunks = []
        for chunk in self.model.generate_response_stream():
            answer_chunks.append(chunk)
        answer = "".join(answer_chunks)
        self.model.add_assistant_message(answer)
        return answer


def text_top(results, n: int) -> str:
    return "\n".join(item.text for _, _, item in results[:n])


def run_one(adapter: Adapter, test: TestCase, args) -> Dict[str, object]:
    adapter.reset()

    t0 = time.perf_counter()
    for s in test.store:
        adapter.store_text(s)
    if test.forget:
        adapter.forget(test.forget)
    store_latency = time.perf_counter() - t0

    t1 = time.perf_counter()
    if True:
        answer = adapter.ask(test.question, args.topk, args.min_score, args.ollama_model, args.debug)
        query_latency = time.perf_counter() - t1
        ok, reason = evaluate_text(answer, test.expected_any, test.expected_all, test.forbidden, test.category)
        top1_ok = top3_ok = top5_ok = ok
        retrieved = answer
    else:
        results = adapter.retrieve(test.question, args.topk, args.min_score)
        query_latency = time.perf_counter() - t1

        t1_text, t3_text, t5_text = text_top(results, 1), text_top(results, 3), text_top(results, 5)
        top1_ok, r1 = evaluate_text(t1_text, test.expected_any, test.expected_all, test.forbidden, test.category)
        top3_ok, r3 = evaluate_text(t3_text, test.expected_any, test.expected_all, test.forbidden, test.category)
        top5_ok, reason = evaluate_text(t5_text, test.expected_any, test.expected_all, test.forbidden, test.category)
        retrieved = t5_text.replace("\n", " | ")

    return {
        "model": adapter.name,
        "scale": test.scale,
        "category": test.category,
        "test": test.name,
        "top1_passed": bool(top1_ok),
        "top3_passed": bool(top3_ok),
        "top5_passed": bool(top5_ok),
        "passed": bool(top5_ok),
        "reason": reason,
        "store_latency_sec": store_latency,
        "query_latency_sec": query_latency,
        "active_memories": getattr(adapter.model, "get_active_tokens_count", lambda: 0)(),
        "inactive_memories": getattr(adapter.model, "get_dropped_tokens_count", lambda: 0)(),
        "question": test.question,
        "expected_any": json.dumps(test.expected_any, ensure_ascii=False),
        "expected_all": json.dumps(test.expected_all, ensure_ascii=False),
        "forbidden": json.dumps(test.forbidden, ensure_ascii=False),
        "notes": test.notes,
        "retrieved_or_answer": retrieved,
    }


def aggregate(rows: List[Dict[str, object]], keys: List[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, object]]] = {}
    for r in rows:
        groups.setdefault(tuple(str(r[k]) for k in keys), []).append(r)

    out = []
    for key, group in sorted(groups.items()):
        total = len(group)
        row = {keys[i]: key[i] for i in range(len(keys))}
        row.update({
            "total": total,
            "top1": sum(bool(r["top1_passed"]) for r in group),
            "top3": sum(bool(r["top3_passed"]) for r in group),
            "top5": sum(bool(r["top5_passed"]) for r in group),
            "top1_acc": sum(bool(r["top1_passed"]) for r in group) / total,
            "top3_acc": sum(bool(r["top3_passed"]) for r in group) / total,
            "top5_acc": sum(bool(r["top5_passed"]) for r in group) / total,
            "avg_store": statistics.mean(float(r["store_latency_sec"]) for r in group),
            "avg_query": statistics.mean(float(r["query_latency_sec"]) for r in group),
        })
        out.append(row)
    return out


def print_summary(rows: List[Dict[str, object]]):
    print("\n" + "=" * 100)
    print("GLOBAL RESULTS")
    print("=" * 100)
    for r in aggregate(rows, ["model"]):
        print(f"{r['model'].upper():<8} | Top1 {r['top1']}/{r['total']} ({r['top1_acc']*100:.1f}%) | Top3 {r['top3']}/{r['total']} ({r['top3_acc']*100:.1f}%) | Top5 {r['top5']}/{r['total']} ({r['top5_acc']*100:.1f}%) | store={r['avg_store']:.3f}s | query={r['avg_query']:.3f}s")

    print("\nRESULTS BY SCALE")
    print("-" * 100)
    for r in aggregate(rows, ["model", "scale"]):
        print(f"{r['model']:<8} | {r['scale']:<6} | Top1 {r['top1']}/{r['total']} ({r['top1_acc']*100:.1f}%) | Top5 {r['top5']}/{r['total']} ({r['top5_acc']*100:.1f}%) | store={r['avg_store']:.3f}s | query={r['avg_query']:.3f}s")

    print("\nRESULTS BY CATEGORY")
    print("-" * 100)
    for r in aggregate(rows, ["model", "category"]):
        print(f"{r['model']:<8} | {r['category']:<32} | Top1 {r['top1']}/{r['total']} ({r['top1_acc']*100:.1f}%) | Top5 {r['top5']}/{r['total']} ({r['top5_acc']*100:.1f}%)")

    failures = [r for r in rows if not r["top5_passed"]]
    print("\nTOP-5 FAILURES")
    print("-" * 100)
    if not failures:
        print("No Top-5 failures.")
    for r in failures:
        print(f"{r['model']} | {r['scale']} | {r['category']} | {r['test']} | {r['reason']}")

    top1_failures = [r for r in rows if not r["top1_passed"]]
    print("\nTOP-1 FAILURES")
    print("-" * 100)
    if not top1_failures:
        print("No Top-1 failures.")
    for r in top1_failures[:40]:
        print(f"{r['model']} | {r['scale']} | {r['category']} | {r['test']} | {r['reason']}")


def write_csv(rows: List[Dict[str, object]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_report(rows: List[Dict[str, object]], path: Path, args):
    lines = [
        "# Stress Benchmark Report — Mamba vs Titan",
        "",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Mode: {'full LLM' if args.full_llm else 'retrieval-only'}",
        f"- Scales: {', '.join(args.scales)}",
        f"- Top-k: {args.topk}",
        "",
        "## Global results",
        "",
        "| Model | Top1 | Top3 | Top5 | Total | Avg store | Avg query |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for r in aggregate(rows, ["model"]):
        lines.append(f"| {r['model']} | {r['top1']} ({r['top1_acc']*100:.1f}%) | {r['top3']} ({r['top3_acc']*100:.1f}%) | {r['top5']} ({r['top5_acc']*100:.1f}%) | {r['total']} | {r['avg_store']:.3f}s | {r['avg_query']:.3f}s |")

    lines += ["", "## Results by category", "", "| Model | Category | Top1 | Top5 | Total |", "|---|---|---:|---:|---:|"]
    for r in aggregate(rows, ["model", "category"]):
        lines.append(f"| {r['model']} | {r['category']} | {r['top1']} ({r['top1_acc']*100:.1f}%) | {r['top5']} ({r['top5_acc']*100:.1f}%) | {r['total']} |")

    lines += ["", "## Top-5 failures", ""]
    failures = [r for r in rows if not r["top5_passed"]]
    if not failures:
        lines.append("No Top-5 failures.")
    for r in failures:
        lines += [
            f"### {r['model']} - {r['scale']} - {r['test']}",
            f"- Category: {r['category']}",
            f"- Reason: {r['reason']}",
            f"- Question: {r['question']}",
            f"- Retrieved/answer: `{r['retrieved_or_answer']}`",
            "",
        ]

    lines += ["", "## Top-1 failures", ""]
    top1_failures = [r for r in rows if not r["top1_passed"]]
    if not top1_failures:
        lines.append("No Top-1 failures.")
    for r in top1_failures:
        lines += [
            f"### {r['model']} - {r['scale']} - {r['test']}",
            f"- Category: {r['category']}",
            f"- Reason: {r['reason']}",
            f"- Question: {r['question']}",
            f"- Retrieved/answer: `{r['retrieved_or_answer']}`",
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]


    p = argparse.ArgumentParser(description="Stress benchmark for memory models.")
    p.add_argument("--models", nargs="*", default=[], choices=["transformer", "lstm", "gru", "gnn", "mamba", "titan"])
    p.add_argument("--scales", nargs="+", default=["small", "medium"], choices=["small", "medium", "large"])
    p.add_argument("--full-llm", action="store_true", default=True)
    p.add_argument("--ollama-model", default="llama3")
    p.add_argument("--capacity", type=int, default=8192)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--min-score", type=float, default=0.12)
    p.add_argument("--small-profiles", type=int, default=20)
    p.add_argument("--small-distractors", type=int, default=40)
    p.add_argument("--medium-profiles", type=int, default=80)
    p.add_argument("--medium-distractors", type=int, default=180)
    p.add_argument("--large-profiles", type=int, default=250)
    p.add_argument("--large-distractors", type=int, default=700)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    if not args.models:
        print("\n--- Model Selection ---")
        print("Select models to test (comma-separated):")
        print("1. transformer")
        print("2. lstm")
        print("3. gru")
        print("4. gnn")
        print("5. mamba")
        print("6. titan")
        print("7. all")
        choice = input("Choice: ").strip().lower()
        if choice == "7" or "all" in choice:
            args.models = ["transformer", "lstm", "gru", "gnn", "mamba", "titan"]
        else:
            selected = []
            for part in choice.split(","):
                part = part.strip()
                if part in ("1", "transformer"): selected.append("transformer")
                elif part in ("2", "lstm"): selected.append("lstm")
                elif part in ("3", "gru"): selected.append("gru")
                elif part in ("4", "gnn"): selected.append("gnn")
                elif part in ("5", "mamba"): selected.append("mamba")
                elif part in ("6", "titan"): selected.append("titan")
            if not selected:
                print("Invalid choice, defaulting to all models.")
                args.models = ["transformer", "lstm", "gru", "gnn", "mamba", "titan"]
            else:
                args.models = list(set(selected))

    random.seed(args.seed)


    output_dir = script_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    tests = build_tests(args)

    print("\nStress benchmark: Mamba vs Titan")
    print(f"Models: {', '.join(args.models)}")
    print(f"Scales: {', '.join(args.scales)}")
    print(f"Mode: {'full LLM' if args.full_llm else 'retrieval-only'}")
    print(f"Tests per model: {len(tests)}")
    print(f"Top-k: {args.topk}\n")

    adapters = {model: Adapter(model, args) for model in args.models}

    rows: List[Dict[str, object]] = []
    total = len(args.models) * len(tests)
    run = 0

    for test in tests:
        for model in args.models:
            run += 1
            print(f"[{run}/{total}] {model} | {test.scale} | {test.category} | {test.name}")
            try:
                row = run_one(adapters[model], test, args)
            except Exception as exc:
                row = {
                    "model": model, "scale": test.scale, "category": test.category, "test": test.name,
                    "top1_passed": False, "top3_passed": False, "top5_passed": False, "passed": False,
                    "reason": f"exception: {exc}", "store_latency_sec": 0.0, "query_latency_sec": 0.0,
                    "active_memories": 0, "inactive_memories": 0, "question": test.question,
                    "expected_any": json.dumps(test.expected_any), "expected_all": json.dumps(test.expected_all),
                    "forbidden": json.dumps(test.forbidden), "notes": test.notes, "retrieved_or_answer": "",
                }

            rows.append(row)
            print(f"  Top1={'PASS' if row['top1_passed'] else 'FAIL'} | Top3={'PASS' if row['top3_passed'] else 'FAIL'} | Top5={'PASS' if row['top5_passed'] else 'FAIL'} | store={float(row['store_latency_sec']):.3f}s | query={float(row['query_latency_sec']):.3f}s")
            if args.debug or not row["top5_passed"] or not row["top1_passed"]:
                print(f"  Reason: {row['reason']}")
                print(f"  Retrieved/answer: {row['retrieved_or_answer']}")
            print()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"stress_comparison_{ts}.csv"
    report_path = output_dir / f"stress_comparison_report_{ts}.md"

    write_csv(rows, csv_path)
    write_report(rows, report_path, args)
    print_summary(rows)

    print(f"\nCSV saved to: {csv_path}")
    print(f"Report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
