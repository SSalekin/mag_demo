#!/usr/bin/env python3
"""Focused tests for the standalone Titans long-term memory module.

Run from the repository root:
    python Test_models/benchmark/test_titans_ltm.py
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models.titans_ltm import NeuralLongTermMemory, TitansMemoryConfig


def main() -> int:
    torch.manual_seed(7)

    config = TitansMemoryConfig(
        dim=32,
        num_memory_layers=1,
        memory_hidden_mult=2.0,
        memory_lr=0.35,
        memory_momentum=0.0,
        memory_decay=0.0,
        activation="gelu",
    )
    memory = NeuralLongTermMemory(config)

    x = torch.randn(1, 8, config.dim)

    # 1. Shape check.
    out, state = memory(x)
    assert out.shape == x.shape
    assert state is not None
    assert len(state.weights) == config.num_memory_layers
    assert len(state.momentum) == config.num_memory_layers

    # 2. Retrieve-only mode should not update the state.
    state_before = state.clone()
    _ = memory.retrieve(x, state)
    for before, after in zip(state_before.weights, state.weights, strict=True):
        assert torch.allclose(before, after)

    # 3. Update mode should change memory weights.
    _, state_after = memory(x, state=state)
    changed = any(
        not torch.allclose(before, after)
        for before, after in zip(state.weights, state_after.weights, strict=True)
    )
    assert changed

    # 4. Repeated test-time learning should reduce associative loss.
    state_train = memory.init_state("cpu")
    loss_before = memory.associative_loss_for_input(x, state_train)
    for _ in range(20):
        _, state_train = memory(x, state=state_train, update=True)
    loss_after = memory.associative_loss_for_input(x, state_train)
    assert loss_after < loss_before, (loss_before, loss_after)

    # 5. Tiny parameter budget sanity check.
    assert memory.count_parameters() < 100_000

    print("Titans LTM focused checks: 5/5 passed")
    print(f"loss_before={loss_before:.6f} | loss_after={loss_after:.6f}")
    print(f"params={memory.count_parameters():,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
