#!/usr/bin/env python3
"""
Stress comparison between the old Legacy Titan prototype and the new Titan model.

Run from the project root:

    python Test_models/benchmark/compare_titan_versions_stress.py --scales small medium

The old Legacy Titan did not use llama3 by default. Its original response layer
used RESPONSE_MODEL_ID, defaulting to gemma2:2b. For the main memory-only
comparison this script defaults to --legacy-mode facts, which compares the
legacy memory content without depending on an Ollama answer generator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_MODELS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TEST_MODELS_DIR.parent
sys.path.insert(0, str(TEST_MODELS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from compare_memory_models_stress import (  # noqa: E402
    aggregate,
    build_tests,
    evaluate_text,
)


class TitanVersionStressAdapter:
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


def run_one(adapter: TitanVersionStressAdapter, test, args: argparse.Namespace) -> Dict[str, object]:
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

    ok, reason = evaluate_text(
        answer,
        test.expected_any,
        test.expected_all,
        test.forbidden,
        test.category,
    )

    # This script evaluates the produced/retrieved answer text. Therefore Top1,
    # Top3 and Top5 are identical here. This matches the current full-answer mode
    # used by compare_memory_models_stress.py in this project.
    return {
        "model": adapter.name,
        "scale": test.scale,
        "category": test.category,
        "test": test.name,
        "top1_passed": bool(ok),
        "top3_passed": bool(ok),
        "top5_passed": bool(ok),
        "passed": bool(ok),
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
        "retrieved_or_answer": answer.replace("\n", " | "),
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
    lines = [
        "# Stress Comparison — Old Titan vs New Titan",
        "",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Scales: {', '.join(args.scales)}",
        f"- Legacy source: `Legacy/titan_implementation.py`",
        f"- Legacy mode: `{args.legacy_mode}`",
        f"- Legacy response model id: `{args.legacy_response_model}`",
        f"- New Titan Ollama model: `{args.new_ollama_model}`",
        f"- Capacity: `{args.capacity}`",
        "",
        "## Interpretation note",
        "",
        "This script compares the old and new Titan versions on the same stress cases used for the Mamba/Titan stress benchmark.",
        "By default, the legacy version runs in `facts` mode to compare memory content directly and avoid depending on its old LLM response layer.",
        "Because the current project evaluates the generated/retrieved answer text, Top1, Top3 and Top5 are identical in this script.",
        "",
        "## Global results",
        "",
        "| Model | Top1 | Top3 | Top5 | Total | Avg store | Avg query |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for r in aggregate(rows, ["model"]):
        lines.append(
            f"| {r['model']} | {r['top1']} ({r['top1_acc']*100:.1f}%) | "
            f"{r['top3']} ({r['top3_acc']*100:.1f}%) | "
            f"{r['top5']} ({r['top5_acc']*100:.1f}%) | {r['total']} | "
            f"{r['avg_store']:.3f}s | {r['avg_query']:.3f}s |"
        )

    lines += [
        "",
        "## Results by scale",
        "",
        "| Model | Scale | Top1 | Top5 | Total | Avg store | Avg query |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in aggregate(rows, ["model", "scale"]):
        lines.append(
            f"| {r['model']} | {r['scale']} | {r['top1']} ({r['top1_acc']*100:.1f}%) | "
            f"{r['top5']} ({r['top5_acc']*100:.1f}%) | {r['total']} | "
            f"{r['avg_store']:.3f}s | {r['avg_query']:.3f}s |"
        )

    lines += [
        "",
        "## Results by category",
        "",
        "| Model | Category | Top1 | Top5 | Total |",
        "|---|---|---:|---:|---:|",
    ]
    for r in aggregate(rows, ["model", "category"]):
        lines.append(
            f"| {r['model']} | {r['category']} | {r['top1']} ({r['top1_acc']*100:.1f}%) | "
            f"{r['top5']} ({r['top5_acc']*100:.1f}%) | {r['total']} |"
        )

    failures = [r for r in rows if not r["top5_passed"]]
    lines += ["", "## Top-5 failures", ""]
    if not failures:
        lines.append("No Top-5 failures.")
    for r in failures:
        lines += [
            f"### {r['model']} - {r['scale']} - {r['test']}",
            f"- Category: {r['category']}",
            f"- Reason: {r['reason']}",
            f"- Question: {r['question']}",
            f"- Expected any: `{r['expected_any']}`",
            f"- Expected all: `{r['expected_all']}`",
            f"- Forbidden: `{r['forbidden']}`",
            f"- Retrieved/answer: `{r['retrieved_or_answer']}`",
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(rows: List[Dict[str, object]]) -> None:
    print("\n" + "=" * 100)
    print("OLD TITAN VS NEW TITAN - STRESS RESULTS")
    print("=" * 100)
    for r in aggregate(rows, ["model"]):
        print(
            f"{r['model']:<14} | Top1 {r['top1']}/{r['total']} ({r['top1_acc']*100:.1f}%) | "
            f"Top3 {r['top3']}/{r['total']} ({r['top3_acc']*100:.1f}%) | "
            f"Top5 {r['top5']}/{r['total']} ({r['top5_acc']*100:.1f}%) | "
            f"store={r['avg_store']:.3f}s | query={r['avg_query']:.3f}s"
        )

    print("\nRESULTS BY SCALE")
    print("-" * 100)
    for r in aggregate(rows, ["model", "scale"]):
        print(
            f"{r['model']:<14} | {r['scale']:<6} | Top1 {r['top1']}/{r['total']} ({r['top1_acc']*100:.1f}%) | "
            f"Top5 {r['top5']}/{r['total']} ({r['top5_acc']*100:.1f}%) | "
            f"store={r['avg_store']:.3f}s | query={r['avg_query']:.3f}s"
        )

    print("\nRESULTS BY CATEGORY")
    print("-" * 100)
    for r in aggregate(rows, ["model", "category"]):
        print(
            f"{r['model']:<14} | {r['category']:<32} | Top1 {r['top1']}/{r['total']} "
            f"({r['top1_acc']*100:.1f}%) | Top5 {r['top5']}/{r['total']} ({r['top5_acc']*100:.1f}%)"
        )

    failures = [r for r in rows if not r["top5_passed"]]
    print("\nTOP-5 FAILURES")
    print("-" * 100)
    if not failures:
        print("No Top-5 failures.")
    for r in failures:
        print(f"{r['model']} | {r['scale']} | {r['category']} | {r['test']} | {r['reason']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stress compare Legacy Titan and new Titan.")
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
    parser.add_argument("--small-profiles", type=int, default=20)
    parser.add_argument("--small-distractors", type=int, default=40)
    parser.add_argument("--medium-profiles", type=int, default=80)
    parser.add_argument("--medium-distractors", type=int, default=180)
    parser.add_argument("--large-profiles", type=int, default=250)
    parser.add_argument("--large-distractors", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    tests = build_tests(args)

    print("\nStress comparison: old Legacy Titan vs new Titan")
    print(f"Versions: {', '.join(args.versions)}")
    print(f"Scales: {', '.join(args.scales)}")
    print(f"Legacy source: {PROJECT_ROOT / 'Legacy' / 'titan_implementation.py'}")
    print(f"Legacy mode: {args.legacy_mode}")
    print(f"Legacy response model id: {args.legacy_response_model}")
    print(f"New Titan Ollama model: {args.new_ollama_model}")
    print(f"Tests per version: {len(tests)}\n")

    adapters = {version: TitanVersionStressAdapter(version, args) for version in args.versions}

    rows: List[Dict[str, object]] = []
    total = len(args.versions) * len(tests)
    run_idx = 0
    for test in tests:
        for version in args.versions:
            run_idx += 1
            print(f"[{run_idx}/{total}] {version} | {test.scale} | {test.category} | {test.name}")
            try:
                row = run_one(adapters[version], test, args)
            except Exception as exc:
                row = {
                    "model": version,
                    "scale": test.scale,
                    "category": test.category,
                    "test": test.name,
                    "top1_passed": False,
                    "top3_passed": False,
                    "top5_passed": False,
                    "passed": False,
                    "reason": f"exception: {exc}",
                    "store_latency_sec": 0.0,
                    "query_latency_sec": 0.0,
                    "active_memories": 0,
                    "inactive_memories": 0,
                    "question": test.question,
                    "expected_any": json.dumps(test.expected_any, ensure_ascii=False),
                    "expected_all": json.dumps(test.expected_all, ensure_ascii=False),
                    "forbidden": json.dumps(test.forbidden, ensure_ascii=False),
                    "notes": test.notes,
                    "retrieved_or_answer": "",
                }
            rows.append(row)
            print(
                f"  Top1={'PASS' if row['top1_passed'] else 'FAIL'} | "
                f"Top3={'PASS' if row['top3_passed'] else 'FAIL'} | "
                f"Top5={'PASS' if row['top5_passed'] else 'FAIL'} | "
                f"store={float(row['store_latency_sec']):.3f}s | query={float(row['query_latency_sec']):.3f}s"
            )
            if args.debug or not row["top5_passed"]:
                print(f"  Reason: {row['reason']}")
                print(f"  Retrieved/answer: {row['retrieved_or_answer']}")
            print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = SCRIPT_DIR / "results"
    csv_path = output_dir / f"titan_versions_stress_{timestamp}.csv"
    report_path = output_dir / f"titan_versions_stress_report_{timestamp}.md"
    write_csv(rows, csv_path)
    write_report(rows, report_path, args)
    print_summary(rows)
    print(f"\nCSV saved to: {csv_path}")
    print(f"Report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
