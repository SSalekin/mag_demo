#!/usr/bin/env python3
"""Focused checks for the Titans LTM consolidation method.

This test does not call Ollama. It verifies that the new memory-only
consolidation method can replay active key/value associations into the LTM,
that it is exposed through TitanExternalMemory, and that forgotten/inactive
items are not reactivated by consolidation.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.titan_model import (  # noqa: E402
    NeuralLongTermMemory,
    TitanExternalMemory,
    TitansMemoryConfig,
)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def top_texts(results):
    return [item.text.lower() for _, _, item in results]


def main() -> int:
    # 1) Unit-level LTM consolidation reduces explicit pair loss.
    torch.manual_seed(7)
    config = TitansMemoryConfig(dim=32, num_memory_layers=2, memory_lr=0.08, memory_momentum=0.80, memory_decay=0.0005)
    ltm = NeuralLongTermMemory(config)
    state = ltm.init_state("cpu")
    keys = torch.randn(1, 6, 32)
    values = torch.randn(1, 6, 32)
    loss_before = ltm.associative_loss_for_pairs(keys, values, state)
    state = ltm.consolidate(keys, values, state, steps=6, reset_state=True)
    loss_after = ltm.associative_loss_for_pairs(keys, values, state)
    assert_true(loss_after < loss_before, "LTM consolidation should reduce associative loss")

    # 2) Integration-level consolidation works on active Titan memories.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "titan_consolidation.pt"
        memory = TitanExternalMemory(
            memory_path=path,
            d_model=64,
            hidden_dim=128,
            max_items=100,
            device="cpu",
            learning_rate=0.04,
            weight_decay=0.001,
            replay_items=4,
            use_aedelon_ltm=True,
        )
        memory.store_text("Lucas Martin's secret code is 8392.")
        memory.store_text("Sarah Martin's favorite color is green.")
        memory.store_text("Lucas Martin's project is a cloud monitoring dashboard.")

        result = memory.consolidate(steps=5, reset_ltm=True)
        assert_true(result["consolidated"] == 3, "all active items should be consolidated")
        assert_true(result["loss_after"] <= result["loss_before"], "consolidation should not increase pair loss")
        assert_true(memory.last_consolidated_at is not None, "consolidation timestamp should be set")

        res = memory.retrieve("What is Lucas Martin's secret code?", k=3, min_score=-1.0)
        assert_true(any("8392" in t for t in top_texts(res)), "retrieval should still find code after consolidation")

        # 3) Forget + active-only consolidation should keep inactive facts inactive.
        candidates = memory.search_forget_candidates("Sarah Martin favorite color", k=3)
        assert_true(candidates, "forget candidate should exist")
        memory.deactivate_ids([candidates[0][2].id], reason="test forget before consolidation")
        result = memory.consolidate(steps=5, reset_ltm=True, include_inactive=False)
        assert_true(result["consolidated"] == 2, "only active memories should be replayed after forget")
        res = memory.retrieve("What is Sarah Martin's favorite color?", k=3, min_score=-1.0)
        assert_true(not any("sarah" in t and "green" in t for t in top_texts(res)), "inactive fact should not be retrieved after consolidation")

        # 4) Save/load preserves explicit memory and consolidation metadata.
        memory.save()
        loaded = TitanExternalMemory(memory_path=path, d_model=64, hidden_dim=128, device="cpu", use_aedelon_ltm=True)
        assert_true(loaded.last_consolidated_at is not None, "loaded memory should preserve consolidation timestamp")
        res = loaded.retrieve("What is Lucas Martin's project?", k=3, min_score=-1.0)
        assert_true(any("cloud monitoring dashboard" in t for t in top_texts(res)), "loaded memory should retain consolidated active facts")

    print("Titan LTM consolidation checks: 4/4 passed")
    print(f"loss_before={loss_before:.6f} | loss_after={loss_after:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
