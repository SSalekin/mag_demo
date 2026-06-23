#!/usr/bin/env python3
"""Titan memory adapter for AI agents.

This module intentionally keeps the agent-facing API small and stable.
It wraps the current TitanExternalMemory implementation so that any agent
(Agno, a custom CLI agent, or a coding agent) can use Titan as a long-term
memory layer without knowing the internals of titan_model.py.

The adapter exposes four core operations:
- store(text): write a memory
- recall(query): retrieve relevant memories
- forget(query): deactivate a targeted memory
- consolidate(): replay active memories into the Titan LTM
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Allow running this file both as a module from Test_models and directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.titan_model import TitanExternalMemory, MemoryItem  # type: ignore


@dataclass
class AgentMemoryRecord:
    """Agent-safe representation of a Titan memory item."""

    id: int
    text: str
    score: float = 1.0
    subject: Optional[str] = None
    property: Optional[str] = None
    active: bool = True
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_item(cls, item: MemoryItem, score: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> "AgentMemoryRecord":
        return cls(
            id=int(item.id),
            text=str(item.text),
            score=float(score),
            subject=getattr(item, "subject", None),
            property=getattr(item, "property", None),
            active=bool(getattr(item, "active", True)),
            metadata=metadata or {},
        )


class TitanAgentMemory:
    """Agent-facing adapter around TitanExternalMemory.

    The purpose is not to build a complete agent here. The purpose is to make
    the Titan memory usable by any agent framework through a clean API.
    """

    def __init__(
        self,
        memory_path: str | Path = "agent_titan_memory.pt",
        d_model: int = 128,
        hidden_dim: int = 256,
        max_items: int = 5000,
        device: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.12,
    ) -> None:
        self.memory_path = Path(memory_path)
        self.top_k = int(top_k)
        self.min_score = float(min_score)
        if device is None:
            # Keep default CPU-friendly unless caller explicitly asks for CUDA.
            device = os.getenv("TITAN_DEVICE", "cpu")
        self.memory = TitanExternalMemory(
            memory_path=self.memory_path,
            d_model=d_model,
            hidden_dim=hidden_dim,
            max_items=max_items,
            device=device,
            use_aedelon_ltm=True,
        )

    # ------------------------------------------------------------------
    # Core operations expected by an AI agent.
    # ------------------------------------------------------------------
    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[AgentMemoryRecord]:
        """Store one or more facts in Titan memory.

        metadata is currently kept at adapter level only. The existing Titan
        memory item does not have a metadata field, so this method does not
        mutate the model internals. It can be extended later if needed.
        """
        results = self.memory.store_text(text)
        records: List[AgentMemoryRecord] = []
        for _action, item, _deactivated in results:
            records.append(AgentMemoryRecord.from_item(item, metadata=metadata))
        return records

    def recall(self, query: str, top_k: Optional[int] = None, min_score: Optional[float] = None) -> List[AgentMemoryRecord]:
        """Retrieve memories relevant to a query."""
        k = self.top_k if top_k is None else int(top_k)
        score_threshold = self.min_score if min_score is None else float(min_score)
        retrieved = self.memory.retrieve(query, k=k, min_score=score_threshold)
        return [AgentMemoryRecord.from_item(item, score=score, metadata={"details": details}) for score, details, item in retrieved]

    def forget(self, query: str, safe_top_k: int = 5) -> List[AgentMemoryRecord]:
        """Deactivate the best safe forget candidate.

        For an agent tool, forgetting should be targeted and conservative. This
        method deactivates only the top candidate selected by Titan's own forget
        candidate search.
        """
        candidates = self.memory.search_forget_candidates(query, k=safe_top_k)
        if not candidates:
            return []
        best = candidates[0][2]
        changed = self.memory.deactivate_ids([best.id], reason=f"agent forget query: {query}")
        return [AgentMemoryRecord.from_item(item) for item in changed]

    def consolidate(self, keep_existing_ltm: bool = False) -> Dict[str, Any]:
        """Consolidate active memories into Titan long-term memory.

        Adapter option:
        - keep_existing_ltm=False rebuilds the LTM from active memories only.
        - keep_existing_ltm=True reinforces the existing LTM without resetting it.

        The single-file Titan model uses the inverse parameter name: reset_ltm.

        The current single-file Titan model should expose memory.consolidate().
        This adapter also provides a safe fallback for older branches where that
        method is not available yet.
        """
        if hasattr(self.memory, "consolidate"):
            result = self.memory.consolidate(reset_ltm=not keep_existing_ltm)  # type: ignore[attr-defined]
            if isinstance(result, dict):
                return result
            return {"status": "ok", "result": result}
        return {
            "status": "not_available",
            "message": "TitanExternalMemory.consolidate() is not available in this branch.",
            "active_items": len(self.memory.active_items),
        }

    # ------------------------------------------------------------------
    # Utilities for prompt injection and persistence.
    # ------------------------------------------------------------------
    def build_context(self, query: str, top_k: Optional[int] = None) -> str:
        """Build a compact memory block that can be injected into an agent prompt."""
        records = self.recall(query, top_k=top_k)
        if not records:
            return "No relevant long-term memory found."
        lines = ["Relevant long-term Titan memories:"]
        for idx, record in enumerate(records, start=1):
            subject = f"subject={record.subject}" if record.subject else "subject=?"
            prop = f"property={record.property}" if record.property else "property=?"
            lines.append(f"{idx}. [score={record.score:.3f}; {subject}; {prop}] {record.text}")
        return "\n".join(lines)

    def save(self, path: str | Path | None = None) -> None:
        if path is not None:
            self.memory.memory_path = Path(path)
        self.memory.save()

    def load(self, path: str | Path | None = None) -> bool:
        if path is not None:
            self.memory.memory_path = Path(path)
        if not self.memory.memory_path or not self.memory.memory_path.exists():
            return False
        self.memory.load()
        return True

    def clear(self) -> None:
        self.memory.reset()

    def stats(self) -> Dict[str, Any]:
        base = self.memory.stats() if hasattr(self.memory, "stats") else {}
        base.update({
            "adapter": "TitanAgentMemory",
            "memory_path": str(self.memory.memory_path),
            "top_k": self.top_k,
            "min_score": self.min_score,
        })
        return base
