"""
Compatibility wrapper around Legacy/titan_implementation.py.

Goal
----
Expose the old Titan prototype through the same minimal API as the new
Test_models.models.titan_model.TitanModel so we can benchmark old vs new Titan
without modifying the legacy file.

Important notes
---------------
The legacy implementation used RESPONSE_MODEL_ID=gemma2:2b by default, not
llama3. This wrapper does not force llama3. It keeps the legacy response-model
name configurable for reporting, but the default fast comparison mode evaluates
the legacy memory content directly instead of calling an Ollama response model.

Modes
-----
- facts: fast and deterministic. It returns the most relevant raw facts kept by
  the old implementation. This shows the old memory behaviour without the noisy
  LLM response layer.
- neural: slow and experimental. It calls the legacy character-level neural
  decoder after test-time training. This is closer to the old demo but can be
  very slow on medium/large benchmarks.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import types
from pathlib import Path
from typing import List, Optional, Tuple


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _tokens(text: str) -> set[str]:
    return {tok for tok in _norm(text).split() if len(tok) > 1}


def _strip_benchmark_prefix(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"(?is)^please\s+remember\s+this\s+information\s*:\s*", "", text).strip()
    text = re.sub(r"(?is)^remember\s+this\s+information\s*:\s*", "", text).strip()
    return text


def _install_dummy_agno_modules() -> None:
    """Let us import Legacy/titan_implementation.py even if agno is unavailable."""
    if "agno.agent" in sys.modules and "agno.models.ollama" in sys.modules:
        return

    agno = sys.modules.setdefault("agno", types.ModuleType("agno"))
    agent_mod = types.ModuleType("agno.agent")
    models_mod = sys.modules.setdefault("agno.models", types.ModuleType("agno.models"))
    ollama_mod = types.ModuleType("agno.models.ollama")

    class DummyAgent:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def print_response(self, prompt: str):
            print(prompt)

    class DummyOllama:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    agent_mod.Agent = DummyAgent
    ollama_mod.Ollama = DummyOllama
    agno.agent = agent_mod
    models_mod.ollama = ollama_mod

    sys.modules["agno.agent"] = agent_mod
    sys.modules["agno.models.ollama"] = ollama_mod


def _load_legacy_module():
    _install_dummy_agno_modules()
    root = Path(__file__).resolve().parents[2]
    legacy_path = root / "Legacy" / "titan_implementation.py"
    if not legacy_path.exists():
        raise FileNotFoundError(f"Legacy Titan implementation not found: {legacy_path}")

    module_name = "legacy_titan_implementation_for_benchmark"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load legacy module from {legacy_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class LegacyTitanModel:
    """Old Titan adapter with a TitanModel-like API."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_capacity: int = 5000,
        mode: str = "facts",
        train_steps: int = 1,
        max_decode_len: int = 512,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        max_seq_len: int = 1024,
    ):
        self.response_model_id = model_name or os.getenv("RESPONSE_MODEL_ID", "gemma2:2b")
        self.max_capacity = max_capacity
        self.mode = mode
        self.train_steps = max(0, int(train_steps))
        self.max_decode_len = max_decode_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.history: List[dict] = []
        self._legacy = _load_legacy_module()
        self.brain = self._new_brain()
        self._set_system_prompt()

    def _new_brain(self):
        return self._legacy.NeuralMemory(
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            max_seq_len=self.max_seq_len,
        )

    def _set_system_prompt(self) -> None:
        self.history.append({
            "role": "system",
            "content": (
                "Legacy Titan prototype. The original response layer used "
                f"{self.response_model_id} via Ollama by default."
            ),
        })

    def clear_memory(self) -> None:
        self.history = []
        self._set_system_prompt()
        self.brain = self._new_brain()

    def get_active_tokens_count(self) -> int:
        return len(getattr(self.brain, "facts", []))

    def get_dropped_tokens_count(self) -> int:
        # The legacy prototype has no active/inactive memory state.
        return 0

    def get_memory_dump(self) -> str:
        facts = list(getattr(self.brain, "facts", []))
        lines = [
            "Architecture: Legacy Titan implementation",
            f"Mode: {self.mode}",
            f"Response model id: {self.response_model_id}",
            f"Facts: {len(facts)} / {self.max_capacity}",
            "",
        ]
        for i, fact in enumerate(facts[-10:], start=max(1, len(facts) - 9)):
            lines.append(f"  • #{i}: {fact}")
        return "\n".join(lines)

    def _should_store(self, message: str) -> bool:
        lower = message.lower().strip()
        if "please remember this information" in lower:
            return True
        if lower.startswith(("/store", "remember ")):
            return True
        if "?" in lower:
            return False
        fact_patterns = [
            r"\b.+\b\s+is\s+.+",
            r"\b.+\b\s+studies\s+.+",
            r"\b.+\b\s+works\s+on\s+.+",
            r"\b.+\b\s+speaks\s+.+",
        ]
        return any(re.search(pattern, message, flags=re.IGNORECASE) for pattern in fact_patterns)

    def _store_fact(self, message: str) -> None:
        fact = _strip_benchmark_prefix(message)
        fact = re.sub(r"(?is)^/store\s+", "", fact).strip()
        if not fact:
            return
        if len(getattr(self.brain, "facts", [])) >= self.max_capacity:
            self.brain.facts.pop(0)
        if self.mode == "neural":
            # This is faithful to the old demo, but slow.
            self.brain.update(fact, train_steps=max(1, self.train_steps))
        else:
            # Fast mode: use the old explicit fact list as the memory content.
            if fact not in self.brain.facts:
                self.brain.facts.append(fact)
            self.brain.training_text = self.brain._compose_memory_text()

    def _forget(self, message: str) -> None:
        # The old legacy implementation did not implement targeted forgetting.
        # We keep it as a no-op so forget benchmarks reveal this limitation.
        _ = message

    def add_user_message(self, message: str) -> None:
        self.history.append({"role": "user", "content": message})
        lower = message.lower().strip()
        if lower.startswith("/ask"):
            return
        if "please forget this" in lower or lower.startswith(("/forget", "forget ")):
            self._forget(message)
        elif self._should_store(message):
            self._store_fact(message)

    def add_assistant_message(self, message: str) -> None:
        self.history.append({"role": "assistant", "content": message})

    def _facts_answer(self, question: str, k: int = 8) -> str:
        question = re.sub(r"(?is)^/ask\s+", "", question).strip()
        q_tokens = _tokens(question)
        facts = list(getattr(self.brain, "facts", []))
        scored: List[Tuple[float, str]] = []
        for idx, fact in enumerate(facts):
            f_tokens = _tokens(fact)
            overlap = len(q_tokens & f_tokens)
            score = overlap + (idx / max(1, len(facts))) * 0.01
            if overlap > 0:
                scored.append((score, fact))
        if not scored:
            return "I do not know based on legacy memory."
        scored.sort(key=lambda x: x[0], reverse=True)
        return "Legacy memory snapshot: " + " | ".join(fact for _, fact in scored[:k])

    def _neural_answer(self, question: str) -> str:
        _ = question
        try:
            recalled = self.brain.recall(max_len=self.max_decode_len)
        except Exception as exc:
            return f"Legacy neural decode failed: {exc}"
        return recalled or "Legacy neural memory returned empty text."

    def generate_response_stream(self):
        last_user_message = self.history[-1]["content"] if self.history else ""
        if self.mode == "neural":
            answer = self._neural_answer(last_user_message)
        else:
            answer = self._facts_answer(last_user_message)
        yield answer

    def save_memory(self, filepath: str = "legacy_titan_memory.pt"):
        # Kept for API compatibility; benchmarks do not need persistence.
        return filepath

    def load_memory(self, filepath: str = "legacy_titan_memory.pt") -> bool:
        return False
