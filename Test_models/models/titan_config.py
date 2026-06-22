"""Configuration helpers for the Titan-inspired memory backend.

The goal is to keep the memory module small enough for a normal PC while
making the important architectural knobs explicit and reproducible.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class TitanMemoryConfig:
    """Runtime configuration for the lightweight Titan memory module."""

    # Neural memory shape. 128/256 is intentionally tiny: useful for CPU demos.
    d_model: int = 128
    hidden_dim: int = 256

    # Runtime / capacity.
    max_items: int = 5000
    device: str = "auto"
    ollama_model: str = "llama3.2:1b"

    # Retrieval.
    top_k: int = 5
    min_score: float = 0.12
    max_context_memories: int = 5

    # Test-time memory learning.
    learning_rate: float = 0.01
    weight_decay: float = 0.001
    surprise_momentum: float = 0.90
    surprise_lr_scale: float = 8.0
    min_train_steps: int = 4
    max_train_steps: int = 32
    replay_items: int = 8

    # Memory management.
    forgetting_decay: float = 0.001

    @classmethod
    def from_env(cls, **overrides: Any) -> "TitanMemoryConfig":
        cfg = cls(
            d_model=_env_int("TITAN_D_MODEL", cls.d_model),
            hidden_dim=_env_int("TITAN_HIDDEN_DIM", cls.hidden_dim),
            max_items=_env_int("TITAN_MAX_ITEMS", cls.max_items),
            device=os.getenv("TITAN_DEVICE", cls.device),
            ollama_model=os.getenv("OLLAMA_MODEL", os.getenv("TITAN_OLLAMA_MODEL", cls.ollama_model)),
            top_k=_env_int("TITAN_TOP_K", cls.top_k),
            min_score=_env_float("TITAN_MIN_SCORE", cls.min_score),
            max_context_memories=_env_int("TITAN_MAX_CONTEXT_MEMORIES", cls.max_context_memories),
            learning_rate=_env_float("TITAN_LR", cls.learning_rate),
            weight_decay=_env_float("TITAN_WEIGHT_DECAY", cls.weight_decay),
            surprise_momentum=_env_float("TITAN_SURPRISE_MOMENTUM", cls.surprise_momentum),
            surprise_lr_scale=_env_float("TITAN_SURPRISE_LR_SCALE", cls.surprise_lr_scale),
            min_train_steps=_env_int("TITAN_MIN_TRAIN_STEPS", cls.min_train_steps),
            max_train_steps=_env_int("TITAN_MAX_TRAIN_STEPS", cls.max_train_steps),
            replay_items=_env_int("TITAN_REPLAY_ITEMS", cls.replay_items),
            forgetting_decay=_env_float("TITAN_FORGETTING_DECAY", cls.forgetting_decay),
        )
        for key, value in overrides.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def resolved_device(self, cuda_available: bool) -> str:
        if self.device == "auto":
            return "cuda" if cuda_available else "cpu"
        return self.device

    def memory_parameter_count(self) -> int:
        """Approximate trainable + frozen parameter count of TitanMemoryNetwork."""
        # Three dxd projections, two MLP layers and biases.
        return (3 * self.d_model * self.d_model) + (self.d_model * self.hidden_dim + self.hidden_dim) + (self.hidden_dim * self.d_model + self.d_model)

    def total_parameter_budget_estimate(self, llm_params: Optional[int] = None) -> Dict[str, int]:
        memory_params = self.memory_parameter_count()
        return {
            "memory_module_params": memory_params,
            "llm_params": int(llm_params or 0),
            "estimated_total_params": memory_params + int(llm_params or 0),
            "budget_limit_params": 2_000_000_000,
        }

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
