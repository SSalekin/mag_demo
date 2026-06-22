#!/usr/bin/env python3
"""Focused integration checks for TitanExternalMemory + standalone Titans LTM.

This test does not call Ollama. It validates that the current Titan model API can
store, update, forget, retrieve and save/load memories while using the
Aedelon/Titans-inspired NeuralLongTermMemory backend.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.titan_model import TitanExternalMemory  # noqa: E402


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def top_texts(results):
    return [item.text.lower() for _, _, item in results]


def main() -> int:
    os.environ.setdefault("PYTHONHASHSEED", "0")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "titan_ltm_memory.pt"
        memory = TitanExternalMemory(
            memory_path=path,
            d_model=64,
            hidden_dim=128,
            max_items=100,
            device="cpu",
            learning_rate=0.03,
            weight_decay=0.001,
            replay_items=4,
            use_aedelon_ltm=True,
        )

        assert_true(memory.stats()["ltm_enabled"] is True, "LTM backend should be enabled")
        assert_true(memory.stats()["ltm_parameters"] > 0, "LTM parameter count should be positive")

        memory.store_text("Lucas Martin's secret code is 8392.")
        res = memory.retrieve("What is Lucas Martin's secret code?", k=3, min_score=-1.0)
        assert_true(any("8392" in t for t in top_texts(res)), "secret code should be retrieved")

        # Multiple updates should deactivate the older single-value property.
        memory.store_text("Lucas Martin's secret code is 1245.")
        res = memory.retrieve("What is Lucas Martin's secret code now?", k=3, min_score=-1.0)
        texts = top_texts(res)
        assert_true(any("1245" in t for t in texts), "updated secret code should be retrieved")
        assert_true(any(not item.active and "8392" in item.text for item in memory.items), "old code should be inactive")

        # Identity collision: same last name but different first name should not dominate.
        memory.store_text("Sarah Martin's favorite color is green.")
        memory.store_text("Lucas Martin's favorite color is violet.")
        res = memory.retrieve("What is Sarah Martin's favorite color?", k=3, min_score=-1.0)
        assert_true("sarah martin" in res[0][2].text.lower(), "Sarah query should retrieve Sarah memory first")
        assert_true("green" in res[0][2].text.lower(), "Sarah color should be green")

        # Forgetting should deactivate safe candidates.
        candidates = memory.search_forget_candidates("Sarah Martin favorite color", k=3)
        memory.deactivate_ids([candidates[0][2].id], reason="test forget")
        res = memory.retrieve("What is Sarah Martin's favorite color?", k=3, min_score=-1.0)
        assert_true(not any("green" in t and "sarah" in t for t in top_texts(res)), "forgotten Sarah color should not be active")

        # Save/load should preserve explicit memory items and LTM state.
        memory.save()
        loaded = TitanExternalMemory(memory_path=path, d_model=64, hidden_dim=128, device="cpu", use_aedelon_ltm=True)
        res = loaded.retrieve("What is Lucas Martin's secret code?", k=3, min_score=-1.0)
        assert_true(any("1245" in t for t in top_texts(res)), "loaded memory should retrieve updated code")

    print("Titan model + Aedelon LTM integration checks: 6/6 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
