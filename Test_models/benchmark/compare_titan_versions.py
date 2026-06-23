#!/usr/bin/env python3
"""
Compare the old Legacy Titan prototype against the new Test_models Titan.

Run from the project root:

    python Test_models/benchmark/compare_titan_versions.py --scales small medium

The old legacy file did not use llama3 by default. It used RESPONSE_MODEL_ID,
with gemma2:2b as default. This script keeps that separate from the new Titan's
Ollama model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_MODELS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TEST_MODELS_DIR.parent
sys.path.insert(0, str(TEST_MODELS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from compare_memory_models import (  # noqa: E402
    TestCase,
    answer_passes,
    build_tests,
    group_summary,
    summarize,
)


@dataclass
class VersionArgs:
    scales: List[str]
    medium_profiles: int
    medium_distractors: int
    large_profiles: int
    large_distractors: int


class TitanVersionAdapter:
    def __init__(self, name: str, args: argparse.Namespace):
        self.name = name
        self.args = args
        self.model = self._create_model()

    def _create_model(self):
        if self.name == "legacy_titan":
            from models.legacy_titan_model import LegacyTitanModel

            return LegacyTitanModel(
                model_name=self.args.legacy_response_model,
                max_capacity=self.args.capacity,
                mode=self.args.legacy_mode,
                train_steps=self.args.legacy_train_steps,
                max_decode_len=self.args.legacy_max_decode_len,
                embed_dim=self.args.legacy_embed_dim,
                hidden_dim=self.args.legacy_hidden_dim,
                max_seq_len=self.args.legacy_max_seq_len,
            )
        if self.name == "new_titan":
            from models.titan_model import TitanModel

            return TitanModel(
                model_name=self.args.new_ollama_model,
                max_capacity=self.args.capacity,
            )
        raise ValueError(f"Unknown Titan version: {self.name}")

    def reset(self) -> None:
        self.model.clear_memory()

    def store_text(self, text: str) -> None:
        self.model.add_user_message(f"Please remember this information: {text}")
        self.model.add_assistant_message("Understood. I will remember that.")

    def forget(self, query: str) -> None:
        self.model.add_user_message(f"Please forget this: {query}")
        self.model.add_assistant_message("Acknowledged. I have forgotten it.")

    def ask(self, question: str) -> str:
        self.model.add_user_message(question)
        chunks = []
        for chunk in self.model.generate_response_stream():
            chunks.append(chunk)
        answer = "".join(chunks)
        self.model.add_assistant_message(answer)
        return answer


def run_one_test(adapter: TitanVersionAdapter, test: TestCase) -> Dict[str, object]:
    adapter.reset()

    store_start = time.perf_counter()
    for statement in test.store:
        adapter.store_text(statement)
    if test.forget:
        adapter.forget(test.forget)
    store_latency = time.perf_counter() - store_start

    query_start = time.perf_counter()
    answer = adapter.ask(test.question)
    query_latency = time.perf_counter() - query_start

    passed, reason = answer_passes(
        answer,
        test.expected_any,
        test.forbidden,
        test.category,
        test.target_property,
    )

    return {
        "model": adapter.name,
        "scale": test.scale,
        "category": test.category,
        "test": test.name,
        "passed": passed,
        "reason": reason,
        "store_latency_sec": store_latency,
        "query_latency_sec": query_latency,
        "active_memories": getattr(adapter.model, "get_active_tokens_count", lambda: 0)(),
        "inactive_memories": getattr(adapter.model, "get_dropped_tokens_count", lambda: 0)(),
        "question": test.question,
        "expected_any": json.dumps(test.expected_any, ensure_ascii=False),
        "forbidden": json.dumps(test.forbidden, ensure_ascii=False),
        "answer": answer.replace("\n", " | "),
    }


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
    failures = [row for row in rows if not row["passed"]]

    lines = [
        "# Old Titan vs New Titan Comparison", "",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Scales: {', '.join(args.scales)}",
        f"- Legacy source: `Legacy/titan_implementation.py`",
        f"- Legacy mode: `{args.legacy_mode}`",
        f"- Legacy response model id: `{args.legacy_response_model}`",
        f"- New Titan Ollama model: `{args.new_ollama_model}`",
        "",
        "## Interpretation note", "",
        "The legacy prototype originally combined a char-level neural memory with an Agno/Ollama response layer.",
        "By default this benchmark uses `legacy-mode=facts` to compare the memory content deterministically and quickly.",
        "Use `--legacy-mode neural` for the slower experimental decoder path.", "",
        "## Global results", "",
        "| Model | Passed | Total | Accuracy | Avg store latency | Avg query latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, s in summary.items():
        lines.append(
            f"| {model} | {int(s['passed'])} | {int(s['total'])} | "
            f"{s['accuracy']*100:.1f}% | {s['avg_store_latency']:.3f}s | {s['avg_query_latency']:.3f}s |"
        )

    lines.extend([
        "", "## Results by scale", "",
        "| Model | Scale | Passed | Total | Accuracy | Avg store latency | Avg query latency |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in by_scale:
        lines.append(
            f"| {row['model']} | {row['scale']} | {row['passed']} | {row['total']} | "
            f"{float(row['accuracy'])*100:.1f}% | {float(row['avg_store_latency']):.3f}s | "
            f"{float(row['avg_query_latency']):.3f}s |"
        )

    lines.extend([
        "", "## Results by category", "",
        "| Model | Category | Passed | Total | Accuracy |",
        "|---|---|---:|---:|---:|",
    ])
    for row in by_category:
        lines.append(
            f"| {row['model']} | {row['category']} | {row['passed']} | {row['total']} | "
            f"{float(row['accuracy'])*100:.1f}% |"
        )

    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("No failures.")
    else:
        for row in failures:
            lines.extend([
                f"### {row['model']} - {row['scale']} - {row['test']}",
                f"- Category: {row['category']}",
                f"- Reason: {row['reason']}",
                f"- Question: {row['question']}",
                f"- Expected: `{row['expected_any']}`",
                f"- Forbidden: `{row['forbidden']}`",
                f"- Answer/retrieved text: `{row['answer']}`",
                "",
            ])

    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(rows: List[Dict[str, object]]) -> None:
    summary = summarize(rows)
    print("\n" + "=" * 88)
    print("OLD TITAN VS NEW TITAN - GLOBAL RESULTS")
    print("=" * 88)
    for model, s in summary.items():
        print(
            f"{model:<14} | {int(s['passed'])}/{int(s['total'])} "
            f"({s['accuracy']*100:.1f}%) | avg store={s['avg_store_latency']:.3f}s | "
            f"avg query={s['avg_query_latency']:.3f}s"
        )

    print("\nRESULTS BY SCALE")
    print("-" * 88)
    for row in group_summary(rows, "scale"):
        print(
            f"{row['model']:<14} | {row['scale']:<6} | {row['passed']}/{row['total']} "
            f"({float(row['accuracy'])*100:.1f}%) | store={float(row['avg_store_latency']):.3f}s | "
            f"query={float(row['avg_query_latency']):.3f}s"
        )

    print("\nRESULTS BY CATEGORY")
    print("-" * 88)
    for row in group_summary(rows, "category"):
        print(f"{row['model']:<14} | {row['category']:<32} | {row['passed']}/{row['total']} ({float(row['accuracy'])*100:.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Legacy Titan and new Titan.")
    parser.add_argument("--versions", nargs="*", default=["legacy_titan", "new_titan"], choices=["legacy_titan", "new_titan"])
    parser.add_argument("--scales", nargs="+", default=["small", "medium"], choices=["small", "medium", "large"])
    parser.add_argument("--new-ollama-model", default=os.getenv("OLLAMA_MODEL", "llama3.2:1b"))
    parser.add_argument("--legacy-response-model", default=os.getenv("RESPONSE_MODEL_ID", "gemma2:2b"))
    parser.add_argument("--legacy-mode", choices=["facts", "neural"], default="facts")
    parser.add_argument("--legacy-train-steps", type=int, default=1)
    parser.add_argument("--legacy-max-decode-len", type=int, default=512)
    parser.add_argument("--legacy-embed-dim", type=int, default=128)
    parser.add_argument("--legacy-hidden-dim", type=int, default=256)
    parser.add_argument("--legacy-max-seq-len", type=int, default=1024)
    parser.add_argument("--capacity", type=int, default=8192)
    parser.add_argument("--medium-profiles", type=int, default=30)
    parser.add_argument("--medium-distractors", type=int, default=60)
    parser.add_argument("--large-profiles", type=int, default=120)
    parser.add_argument("--large-distractors", type=int, default=240)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    test_args = VersionArgs(
        scales=args.scales,
        medium_profiles=args.medium_profiles,
        medium_distractors=args.medium_distractors,
        large_profiles=args.large_profiles,
        large_distractors=args.large_distractors,
    )
    tests = build_tests(test_args)  # type: ignore[arg-type]

    print("\nComparing old Legacy Titan against new Titan...")
    print(f"Versions: {', '.join(args.versions)}")
    print(f"Scales: {', '.join(args.scales)}")
    print(f"Legacy source: {PROJECT_ROOT / 'Legacy' / 'titan_implementation.py'}")
    print(f"Legacy response model id: {args.legacy_response_model}")
    print(f"Legacy mode: {args.legacy_mode}")
    print(f"New Titan Ollama model: {args.new_ollama_model}")
    print(f"Total test cases per version: {len(tests)}\n")

    adapters = {version: TitanVersionAdapter(version, args) for version in args.versions}

    rows: List[Dict[str, object]] = []
    total_runs = len(args.versions) * len(tests)
    run_idx = 0
    for test in tests:
        for version in args.versions:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] {version} | {test.scale} | {test.name}")
            try:
                row = run_one_test(adapters[version], test)
            except Exception as exc:
                row = {
                    "model": version,
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
                f"  {status} | {row['reason']} | store={float(row['store_latency_sec']):.3f}s | "
                f"query={float(row['query_latency_sec']):.3f}s"
            )
            if args.debug or not row["passed"]:
                print(f"  Answer: {row['answer']}")
            print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = SCRIPT_DIR / "results"
    csv_path = output_dir / f"titan_versions_{timestamp}.csv"
    report_path = output_dir / f"titan_versions_report_{timestamp}.md"
    write_csv(rows, csv_path)
    write_report(rows, report_path, args)
    print_summary(rows)
    print(f"\nCSV saved to: {csv_path}")
    print(f"Report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
