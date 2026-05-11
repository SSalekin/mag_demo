"""
Neural memory demo using a Mamba-style selective state-space sequence model
(Mamba: Linear-Time Sequence Modeling with Selective State Spaces, Gu & Dao).

Core: causal depthwise conv + selective SSM scan (no cross-token attention).
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from agno.agent import Agent
from agno.models.ollama import Ollama

CPU_THREADS = os.cpu_count() or 1
MAX_MEMORY_CHARS = 4096
RESPONSE_MODEL_ID = os.getenv("RESPONSE_MODEL_ID", "gemma2:2b")
DEFAULT_TRAIN_STEPS = 50
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_CHECKPOINT_PATH = "mamba_neural_memory.pt"
TRAINING_LOG_PATH = "training_worker.log"

_training_logger: Optional[logging.Logger] = None


def _get_training_logger() -> logging.Logger:
    global _training_logger
    if _training_logger is not None:
        return _training_logger
    log = logging.getLogger("mamba_implementation.training_worker")
    log.setLevel(logging.INFO)
    if not log.handlers:
        file_handler = logging.FileHandler(
            TRAINING_LOG_PATH, encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        log.addHandler(file_handler)
    log.propagate = False
    _training_logger = log
    return log


torch.set_num_threads(CPU_THREADS)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(CPU_THREADS)


class SelectiveSSM(nn.Module):
    """
    Diagonal selective SSM with per-channel state (Mamba-style discretization).
    Sequential scan: O(L * E * N); fine for demo sequence lengths.
    """

    def __init__(self, d_inner: int, d_state: int = 16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        A = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, -1).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A.clamp(min=1e-4)))
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.B_proj = nn.Linear(d_inner, d_inner * d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_inner * d_state, bias=False)
        self.D_skip = nn.Parameter(torch.ones(d_inner))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, L, d_inner) -> (B, L, d_inner)"""
        b, seq_len, e = u.shape
        A = -torch.exp(self.A_log)
        delta = F.softplus(self.dt_proj(u))
        B_all = self.B_proj(u).view(b, seq_len, e, self.d_state)
        C_all = self.C_proj(u).view(b, seq_len, e, self.d_state)

        h = torch.zeros(b, e, self.d_state, device=u.device, dtype=u.dtype)
        outs: List[torch.Tensor] = []
        for t in range(seq_len):
            d = delta[:, t, :]
            dA = torch.exp(d.unsqueeze(-1) * A.unsqueeze(0))
            dB = d.unsqueeze(-1) * B_all[:, t]
            h = dA * h + dB * u[:, t, :].unsqueeze(-1)
            y = (h * C_all[:, t]).sum(dim=-1) + self.D_skip * u[:, t, :]
            outs.append(y)
        return torch.stack(outs, dim=1)


class MambaBlock(nn.Module):
    """One Mamba-style block: in_proj -> SiLU -> causal depthwise conv -> SSM -> gated out."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            bias=True,
            padding=d_conv - 1,
        )
        self.ssm = SelectiveSSM(self.d_inner, d_state=d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        x_inner = F.silu(x_inner)
        b, l, _ = x_inner.shape
        xc = self.conv1d(x_inner.transpose(1, 2))[:, :, :l].transpose(1, 2)
        x_inner = F.silu(xc)
        y = self.ssm(x_inner)
        y = y * F.silu(z)
        return res + self.out_proj(y)


class NeuralMemory(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    ):
        super().__init__()
        _ = hidden_dim

        self.special_tokens = ["<BOS>", "<EOS>", "<UNK>"]
        self.ascii_tokens = [chr(i) for i in range(32, 127)]
        self.vocab = self.special_tokens + self.ascii_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

        self.bos_idx = self.token_to_idx["<BOS>"]
        self.eos_idx = self.token_to_idx["<EOS>"]
        self.unk_idx = self.token_to_idx["<UNK>"]
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(len(self.vocab), embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        d_state = max(8, min(32, embed_dim // 4))
        self.mamba_blocks = nn.ModuleList(
            [
                MambaBlock(embed_dim, d_state=d_state, d_conv=4, expand=2),
                MambaBlock(embed_dim, d_state=d_state, d_conv=4, expand=2),
            ]
        )
        self.readout = nn.Linear(embed_dim, len(self.vocab))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.facts: List[str] = []
        self.training_text = ""

    def _text_to_ids(self, text: str) -> torch.Tensor:
        ids = [self.token_to_idx.get(ch, self.unk_idx) for ch in text]
        ids.append(self.eos_idx)
        return torch.tensor([ids], dtype=torch.long)

    def _ids_to_text(self, ids: List[int]) -> str:
        chars = []
        for idx in ids:
            token = self.idx_to_token.get(idx, "")
            if token not in self.special_tokens:
                chars.append(token)
        return "".join(chars)

    def _forward_hidden(self, decoder_input: torch.Tensor) -> torch.Tensor:
        seq_len = decoder_input.size(1)
        positions = torch.arange(
            seq_len, dtype=torch.long, device=decoder_input.device
        ).unsqueeze(0)
        hidden = self.embedding(decoder_input) + self.pos_embedding(positions)
        for block in self.mamba_blocks:
            hidden = block(hidden)
        return hidden

    def _compose_memory_text(self) -> str:
        return " | ".join(self.facts)

    def _trim_memory_to_seq_limit(self, merged_memory: str) -> str:
        target = self._text_to_ids(merged_memory)
        while target.size(1) > self.max_seq_len and len(self.facts) > 1:
            self.facts.pop(0)
            merged_memory = self._compose_memory_text()
            target = self._text_to_ids(merged_memory)
        return merged_memory

    def update(self, fact: str, train_steps: int = DEFAULT_TRAIN_STEPS) -> str:
        clean_fact = fact.strip()
        if not clean_fact:
            return "Memory update skipped: empty fact."

        if clean_fact not in self.facts:
            self.facts.append(clean_fact)

        merged_memory = self._trim_memory_to_seq_limit(self._compose_memory_text())
        target = self._text_to_ids(merged_memory)

        bos = torch.tensor([[self.bos_idx]], dtype=torch.long)
        decoder_input = torch.cat([bos, target[:, :-1]], dim=1)
        seq_len = decoder_input.size(1)
        if seq_len > self.max_seq_len:
            return "Memory update skipped: sequence exceeds max sequence length."

        self.training_text = merged_memory
        self.train()

        loss = None
        for _ in range(train_steps):
            self.optimizer.zero_grad(set_to_none=True)
            hidden = self._forward_hidden(decoder_input)
            logits = self.readout(hidden)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

        return (
            f"Memory Updated. Loss: {loss.item():.4f}"
            if loss is not None
            else "Memory update skipped."
        )

    def recall(
        self,
        user_message: str = "",
        max_len: int = 100000,
        deterministic: bool = True,
    ) -> str:
        _ = user_message
        _ = deterministic

        self.eval()
        generated_ids: List[int] = []
        token = torch.tensor([[self.bos_idx]], dtype=torch.long)

        with torch.no_grad():
            for _ in range(max_len):
                seq_len = token.size(1)
                if seq_len > self.max_seq_len:
                    break
                hidden = self._forward_hidden(token)
                logits = self.readout(hidden[:, -1, :])
                next_idx = int(torch.argmax(logits, dim=-1).item())

                if next_idx == self.eos_idx:
                    break
                generated_ids.append(next_idx)
                token = torch.cat(
                    [token, torch.tensor([[next_idx]], dtype=torch.long)],
                    dim=1,
                )

        decoded = self._ids_to_text(generated_ids).strip()
        return decoded if decoded else "Neural memory decode returned empty text."


brain = NeuralMemory()
brain_lock = threading.Lock()

latest_training_status = "No training runs yet."
latest_recalled_text = "No Mamba memory encoded yet."


def write_to_neural_memory(fact: str) -> str:
    """Synchronously train Mamba memory on the new fact."""
    global latest_training_status, latest_recalled_text

    with brain_lock:
        latest_training_status = brain.update(fact)
        latest_recalled_text = brain.recall()
    return latest_training_status


def read_from_neural_memory(user_message: str = "") -> str:
    """Decode current Mamba memory (no background queue required)."""
    global latest_recalled_text

    with brain_lock:
        latest_recalled_text = brain.recall(user_message=user_message)
        print("Latest MEMORY:", latest_recalled_text)

    memory_text = latest_recalled_text[-MAX_MEMORY_CHARS:]
    return f"Mamba Memory Text: {memory_text if memory_text else '[empty]'}\n"


response_agent = Agent(
    model=Ollama(id=RESPONSE_MODEL_ID),
    description="Fast memory-grounded assistant.",
    instructions=[
        "Answer in 1-3 concise sentences.",
        "Use only the provided Mamba memory snapshot for profile facts.",
        "If the snapshot has no fact, say you do not have that information yet.",
    ],
    markdown=True,
)


def _should_memorize(user_message: str) -> bool:
    lower = user_message.lower()
    blocked_markers = [
        "?",
        "don't save",
        "dont save",
        "forget",
        "don't remember",
        "dont remember",
    ]
    if any(marker in lower for marker in blocked_markers):
        return False

    fact_markers = [
        "i am ",
        "i'm ",
        "i live",
        "i study",
        "my ",
        "remember",
        "currently",
    ]
    return any(marker in lower for marker in fact_markers)


def _queue_memorize_if_needed(user_message: str) -> None:
    if _should_memorize(user_message):
        threading.Thread(
            target=write_to_neural_memory,
            args=(user_message,),
            daemon=True,
            name="memorize-dispatch",
        ).start()


def _answer_with_memory_context(user_message: str) -> None:
    memory_snapshot = read_from_neural_memory(user_message=user_message)

    grounded_prompt = (
        "Answer briefly using this memory snapshot.\n\n"
        f"{memory_snapshot}\n\n"
        f"User message: {user_message}"
    )
    response_agent.print_response(grounded_prompt)
    _queue_memorize_if_needed(user_message)


def run_seed_demo() -> None:
    seed_facts = [
        "My name is Salekin",
        "I was born in December 1991",
        "I am a student of Greenwich Danang",
        "I am living in Danang",
        "Danang is the most beautiful city of Vietnam.",
    ]
    for fact in seed_facts:
        write_to_neural_memory(fact)


def _start_seed_demo() -> None:
    thread = threading.Thread(target=run_seed_demo, daemon=True, name="seed-demo")
    thread.start()


def save_neural_memory_checkpoint(path: str = DEFAULT_CHECKPOINT_PATH) -> str:
    checkpoint = {
        "model_state": brain.state_dict(),
        "optimizer_state": brain.optimizer.state_dict(),
        "facts": list(brain.facts),
        "training_text": brain.training_text,
        "max_seq_len": brain.max_seq_len,
        "vocab": list(brain.vocab),
    }
    torch.save(checkpoint, path)
    return path


def load_neural_memory_checkpoint(path: str = DEFAULT_CHECKPOINT_PATH) -> str:
    global latest_recalled_text

    checkpoint = torch.load(path, map_location="cpu")
    brain.load_state_dict(checkpoint["model_state"])

    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer_state:
        brain.optimizer.load_state_dict(optimizer_state)

    brain.facts = list(checkpoint.get("facts", []))
    brain.training_text = checkpoint.get("training_text", "")
    latest_recalled_text = brain.recall()
    return path


def run_chat() -> None:
    print("Neural memory chat started. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_message = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            with brain_lock:
                checkpoint_path = save_neural_memory_checkpoint()
            print(f"Saved neural memory checkpoint to {checkpoint_path}.")
            print("Exiting chat.")
            break

        _answer_with_memory_context(user_message)


if __name__ == "__main__":
    if os.path.exists(DEFAULT_CHECKPOINT_PATH):
        with brain_lock:
            checkpoint_path = load_neural_memory_checkpoint()
        print(f"Loaded neural memory checkpoint from {checkpoint_path}.")
    else:
        _start_seed_demo()
    run_chat()
