#!/usr/bin/env python3
"""Focused retrieval-only checks for the Titan V2 architecture.

This script intentionally does not call Ollama. It validates the memory module
itself: identity collision handling, updates, forgetting, paraphrases and noisy
large memory. Run from `mag_demo/`:

    python Test_models/benchmark/test_titan_architecture.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.titan_config import TitanMemoryConfig
from models.titan_model import TitanModel


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def contains(text: str, expected: str) -> bool:
    return norm(expected) in norm(text)


def teach(model: TitanModel, fact: str) -> None:
    model.add_user_message(f"Please remember this information: {fact}")
    model.add_assistant_message("Stored.")


def forget(model: TitanModel, query: str) -> None:
    model.add_user_message(f"Please forget this: {query}")
    model.add_assistant_message("Forgotten.")


def retrieved_text(model: TitanModel, question: str, topk: int = 5) -> str:
    results = model.memory.retrieve(question, k=topk, min_score=0.0)
    return " | ".join(item.text for _, _, item in results)


def assert_contains(label: str, text: str, expected: str) -> bool:
    ok = contains(text, expected)
    print(f"{'PASS' if ok else 'FAIL'} | {label} | expected={expected!r} | retrieved={text}")
    return ok


def assert_not_contains(label: str, text: str, forbidden: Iterable[str]) -> bool:
    bad = [x for x in forbidden if contains(text, x)]
    ok = not bad
    print(f"{'PASS' if ok else 'FAIL'} | {label} | forbidden={bad!r} | retrieved={text}")
    return ok


def build_model() -> TitanModel:
    # Small shape keeps the test fast on CPU. The production defaults remain in
    # TitanMemoryConfig and can be overridden with TITAN_* environment variables.
    cfg = TitanMemoryConfig(
        d_model=int(os.getenv("TITAN_TEST_D_MODEL", "32")),
        hidden_dim=int(os.getenv("TITAN_TEST_HIDDEN_DIM", "64")),
        max_items=1000,
        device="cpu",
        top_k=5,
        min_score=0.0,
        min_train_steps=1,
        max_train_steps=1,
        replay_items=2,
        ollama_model="llama3.2:1b",
    )
    model = TitanModel(config=cfg)
    model.clear_memory()
    return model


def run() -> int:
    model = build_model()
    checks: List[bool] = []

    # 1) Identity collision: same first name, different last names.
    for fact in [
        "Sarah Nguyen is a 23-year-old AI student from Hanoi, Vietnam.",
        "Sarah Martin is a 25-year-old cybersecurity engineer from Rennes, France.",
        "Sarah Wilson is a 22-year-old design student from Milan, Italy.",
        "Sarah Nguyen's favorite color is purple.",
        "Sarah Martin's favorite color is green.",
        "Sarah Wilson's favorite color is orange.",
    ]:
        teach(model, fact)
    text = retrieved_text(model, "What is Sarah Nguyen's favorite color?")
    checks.append(assert_contains("identity_collision_exact_subject", text, "purple"))
    checks.append(assert_not_contains("identity_collision_no_other_sarah", text, ["green", "orange"]))

    # 2) Multiple updates: only the newest single-value property should remain active.
    teach(model, "Actually, Sarah Nguyen's favorite color is now black.")
    teach(model, "Correction: Sarah Nguyen's favorite color is now cyan.")
    text = retrieved_text(model, "What is Sarah Nguyen's favorite color?")
    checks.append(assert_contains("multiple_updates_latest_value", text, "cyan"))
    checks.append(assert_not_contains("multiple_updates_old_values", text, ["purple", "black"]))

    # 3) Forgetting with nearby distractors.
    teach(model, "Lina Moreau's secret code is LA-2026.")
    teach(model, "Lina Martin's secret code is LM-9999.")
    forget(model, "Lina Moreau secret code")
    text = retrieved_text(model, "What is Lina Moreau's secret code?")
    checks.append(assert_not_contains("forget_specific_secret", text, ["LA-2026", "LM-9999"]))

    # 4) Paraphrased retrieval.
    text = retrieved_text(model, "In which city is Sarah Nguyen currently based?")
    checks.append(assert_contains("paraphrased_location", text, "Hanoi"))

    # 5) Large noisy memory with distractors.
    for i in range(40):
        teach(model, f"Noise note {i}: the hotel lobby has a blue sofa and code CODE-{9000+i} on a router sticker.")
    teach(model, "Maya Dubois speaks French, English, and Spanish.")
    teach(model, "Maya Dubois's office room is R512.")
    text = retrieved_text(model, "Which languages does Maya Dubois speak?")
    checks.append(assert_contains("spoken_languages", text, "French"))
    checks.append(assert_contains("spoken_languages_all", text, "Spanish"))
    text = retrieved_text(model, "What is the real office room of Maya Dubois?")
    checks.append(assert_contains("large_noisy_room", text, "R512"))
    checks.append(assert_not_contains("large_noisy_no_router", text, ["router sticker", "blue sofa"]))

    passed = sum(bool(x) for x in checks)
    total = len(checks)
    print(f"\nTitan V2 focused checks: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(run())
