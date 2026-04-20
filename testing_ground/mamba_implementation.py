#!/usr/bin/env python3
"""
Memory Augmented Generation demo:
- Stores user statements ONLY in an in-memory Mamba-backed vector memory (no DB/files/json).
- Answers questions by retrieving relevant stored statements, then calling Ollama (gemma2:2b).

Notes:
- This does not persist your statements anywhere. They live only in RAM until the process exits.
- Model weights/tokenizer may be cached by Hugging Face/transformers (separate from "memory").
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps))


@dataclass
class MemoryItem:
    text: str
    embedding: torch.Tensor  # shape: (d,)


class MambaVectorMemory:
    """
    In-memory vector store where embeddings come from a Mamba LM's hidden states.
    """

    def __init__(self, hf_model: str, device: str) -> None:
        self.hf_model = hf_model
        self.device = device

        self._tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=True)
        # Mamba models are causal LMs in HF; we use hidden states to derive an embedding.
        self._model = AutoModelForCausalLM.from_pretrained(hf_model)
        self._model.to(device)
        self._model.eval()

        self._items: List[MemoryItem] = []

    @torch.no_grad()
    def embed(self, text: str) -> torch.Tensor:
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs, output_hidden_states=True, use_cache=False)
        # hidden_states[-1]: (batch=1, seq, hidden)
        last = outputs.hidden_states[-1][0]
        attn = inputs.get("attention_mask", torch.ones(last.shape[0], device=last.device)).to(last.dtype)
        attn = attn[0].unsqueeze(-1)  # (seq, 1)
        pooled = (last * attn).sum(dim=0) / attn.sum().clamp_min(1.0)
        return _l2_normalize(pooled.detach().float().cpu())

    def store(self, text: str) -> None:
        emb = self.embed(text)
        self._items.append(MemoryItem(text=text, embedding=emb))

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[float, str]]:
        if not self._items:
            return []

        q = self.embed(query)  # cpu normalized
        scored: List[Tuple[float, str]] = []
        for it in self._items:
            score = float(torch.dot(q, it.embedding))
            scored.append((score, it.text))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[: max(1, k)]


def _ollama_chat(model: str, system: str, user: str) -> str:
    """
    Uses the `ollama` python package if available; falls back to a helpful error otherwise.
    """
    try:
        import ollama  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import `ollama`. Make sure Ollama is installed and `pip install ollama` succeeded."
        ) from e

    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    msg = resp.get("message", {}) if isinstance(resp, dict) else getattr(resp, "message", {})
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    return (content or "").strip()


def _build_prompt(question: str, retrieved: List[Tuple[float, str]]) -> Tuple[str, str]:
    system = (
        "You are a helpful assistant. You MUST ground your answer only in the provided MEMORY SNIPPETS. "
        "If the memory does not contain the answer, say you don't know based on memory."
    )

    if not retrieved:
        user = f"Question: {question}\n\nMEMORY SNIPPETS:\n(none)\n\nAnswer:"
        return system, user

    lines = []
    for i, (score, text) in enumerate(retrieved, start=1):
        # Keep prompt compact but include a relevance hint.
        lines.append(f"[{i}] (relevance={score:.3f}) {text}")
    mem = "\n".join(lines)
    user = f"Question: {question}\n\nMEMORY SNIPPETS:\n{mem}\n\nAnswer:"
    return system, user


def _print_help() -> None:
    print(
        "\nCommands:\n"
        "  /store <text>   Store a statement in Mamba memory\n"
        "  /ask <question> Ask a question; retrieves from memory then uses gemma2:2b via Ollama\n"
        "  /stats          Show memory size\n"
        "  /help           Show this help\n"
        "  /exit           Quit\n"
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Mamba + Ollama + (optional) agno Memory Augmented Generation demo")
    p.add_argument("--mamba-model", default="state-spaces/mamba-130m-hf", help="Hugging Face model id")
    p.add_argument("--ollama-model", default="gemma2:2b", help="Ollama model name (must be pulled locally)")
    p.add_argument("--topk", type=int, default=5, help="How many memory items to retrieve")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Torch device for Mamba",
    )
    args = p.parse_args(argv)

    print("Loading Mamba model (this can take a bit on first run)...")
    memory = MambaVectorMemory(hf_model=args.mamba_model, device=args.device)
    print("Ready.")
    _print_help()

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if not line:
            continue

        if line in {"/exit", "exit", "quit"}:
            return 0
        if line in {"/help", "help"}:
            _print_help()
            continue
        if line == "/stats":
            print(f"Memory items in RAM: {len(memory._items)}")
            continue

        if line.startswith("/store "):
            text = line[len("/store ") :].strip()
            if not text:
                print("Nothing to store.")
                continue
            memory.store(text)
            print("Stored in memory (RAM).")
            continue

        if line.startswith("/ask "):
            q = line[len("/ask ") :].strip()
            if not q:
                print("Ask a non-empty question.")
                continue
            retrieved = memory.retrieve(q, k=args.topk)
            system, user = _build_prompt(q, retrieved)
            try:
                answer = _ollama_chat(model=args.ollama_model, system=system, user=user)
            except Exception as e:
                print(f"Failed to call Ollama: {e}")
                print("Make sure Ollama is running and the model is available, e.g.:")
                print(f"  ollama pull {args.ollama_model}")
                continue

            if not answer:
                answer = "(no response)"
            print("\n" + answer)
            continue

        # Convenience: bare input defaults to /store, bare question defaults to /ask.
        # Heuristic: treat lines ending with '?' as questions.
        if line.endswith("?"):
            q = line
            retrieved = memory.retrieve(q, k=args.topk)
            system, user = _build_prompt(q, retrieved)
            answer = _ollama_chat(model=args.ollama_model, system=system, user=user)
            print("\n" + (answer or "(no response)"))
            continue

        memory.store(line)
        print("Stored in memory (RAM). Tip: use /ask ... to query it.")


if __name__ == "__main__":
    raise SystemExit(main())

