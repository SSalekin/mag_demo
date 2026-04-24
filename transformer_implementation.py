import logging
import os
import queue
import threading
from typing import List, Optional

import torch
import torch.nn as nn
from agno.agent import Agent
from agno.models.ollama import Ollama

CPU_THREADS = os.cpu_count() or 1
MAX_MEMORY_CHARS = 10000000
RESPONSE_MODEL_ID = os.getenv("RESPONSE_MODEL_ID", "gemma2:2b")
DEFAULT_TRAIN_STEPS = 100
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_CHECKPOINT_PATH = "neural_memory.pt"
TRAINING_LOG_PATH = "training_worker.log"

_training_logger: Optional[logging.Logger] = None


def _get_training_logger() -> logging.Logger:
    global _training_logger
    if _training_logger is not None:
        return _training_logger
    log = logging.getLogger("transformer_implementation.training_worker")
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


class NeuralMemory(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, max_seq_len: int = DEFAULT_MAX_SEQ_LEN):
        super().__init__()

        self.special_tokens = ["<BOS>", "<EOS>", "<UNK>"]
        self.ascii_tokens = [chr(i) for i in range(32, 127)]
        self.vocab = self.special_tokens + self.ascii_tokens
        self.token_to_idx = {token: idx for idx,
                             token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for token,
                             idx in self.token_to_idx.items()}

        self.bos_idx = self.token_to_idx["<BOS>"]
        self.eos_idx = self.token_to_idx["<EOS>"]
        self.unk_idx = self.token_to_idx["<UNK>"]
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(len(self.vocab), embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=2)
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

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

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

        merged_memory = self._trim_memory_to_seq_limit(
            self._compose_memory_text())
        target = self._text_to_ids(merged_memory)

        bos = torch.tensor([[self.bos_idx]], dtype=torch.long)
        decoder_input = torch.cat([bos, target[:, :-1]], dim=1)
        seq_len = decoder_input.size(1)
        if seq_len > self.max_seq_len:
            return "Memory update skipped: sequence exceeds max sequence length."

        positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        self.training_text = merged_memory
        self.train()

        loss = None
        for _ in range(train_steps):
            self.optimizer.zero_grad()
            dec_emb = self.embedding(decoder_input) + \
                self.pos_embedding(positions)
            decoded = self.decoder(dec_emb, mask=self._causal_mask(seq_len))
            logits = self.readout(decoded)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

        return f"Memory Updated. Loss: {loss.item():.4f}" if loss is not None else "Memory update skipped."

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
                positions = torch.arange(
                    seq_len, dtype=torch.long).unsqueeze(0)
                dec_emb = self.embedding(token) + self.pos_embedding(positions)
                decoded = self.decoder(
                    dec_emb, mask=self._causal_mask(seq_len))
                logits = self.readout(decoded[:, -1, :])
                next_idx = int(torch.argmax(logits, dim=-1).item())

                if next_idx == self.eos_idx:
                    break
                generated_ids.append(next_idx)
                token = torch.cat([token, torch.tensor(
                    [[next_idx]], dtype=torch.long)], dim=1)

        decoded = self._ids_to_text(generated_ids).strip()
        return decoded if decoded else "Neural memory decode returned empty text."


brain = NeuralMemory()
brain_lock = threading.Lock()
training_queue: "queue.Queue[str]" = queue.Queue()

latest_training_status = "No training runs yet."
latest_memory_text = ""
latest_recalled_text = "No transformer memory encoded yet."


def write_to_neural_memory(fact: str) -> str:
    training_queue.put(fact)
    return "Transformer training queued in background."


def read_from_neural_memory(user_message: str = "") -> str:
    global latest_memory_text, latest_recalled_text

    if brain_lock.acquire(blocking=False):
        try:
            latest_memory_text = brain._compose_memory_text(
            ) if brain.facts else brain.training_text
            latest_recalled_text = brain.recall(user_message=user_message)
            print("Latest MEMORY:", latest_recalled_text)
        finally:
            brain_lock.release()

    memory_text = latest_recalled_text[-MAX_MEMORY_CHARS:]
    return (
        f"Transformer Memory Text: {memory_text if memory_text else '[empty]'}\n"

    )


def _training_worker() -> None:
    global latest_training_status, latest_memory_text, latest_recalled_text

    log = _get_training_logger()
    log.info("Training worker started.")

    while True:
        fact = training_queue.get()
        log.info("Dequeued training item: %r", fact)
        if fact is None:
            log.info("Received shutdown sentinel; exiting training worker loop.")
            training_queue.task_done()
            break

        try:
            log.info("Starting neural memory update for fact.")
            with brain_lock:
                latest_training_status = brain.update(fact)
            log.info("Training update finished: %s", latest_training_status)
        except Exception as err:
            latest_training_status = f"Background training failed: {err}"
            log.exception("Background training failed: %s", err)
        finally:
            training_queue.task_done()
            log.info("Marked training queue item as done.")


def _start_training_worker() -> None:
    thread = threading.Thread(target=_training_worker,
                              daemon=True, name="memory-trainer")
    thread.start()


response_agent = Agent(
    model=Ollama(id=RESPONSE_MODEL_ID),
    description="Fast memory-grounded assistant.",
    instructions=[
        "Answer in 1-3 concise sentences.",
        "Use only the provided transformer memory snapshot for profile facts.",
        "If the snapshot has no fact, say you do not have that information yet.",
    ],
    markdown=True,
)


def _should_memorize(user_message: str) -> bool:
    lower = user_message.lower()
    blocked_markers = ["?", "don't save", "dont save",
                       "forget", "don't remember", "dont remember"]
    if any(marker in lower for marker in blocked_markers):
        return False

    fact_markers = ["i am ", "i'm ", "i live",
                    "i study", "my ", "remember", "currently"]
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
    thread = threading.Thread(target=run_seed_demo,
                              daemon=True, name="seed-demo")
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
    global latest_memory_text, latest_recalled_text

    checkpoint = torch.load(path, map_location="cpu")
    brain.load_state_dict(checkpoint["model_state"])

    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer_state:
        brain.optimizer.load_state_dict(optimizer_state)

    brain.facts = list(checkpoint.get("facts", []))
    brain.training_text = checkpoint.get("training_text", "")
    latest_memory_text = brain._compose_memory_text(
    ) if brain.facts else brain.training_text
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

            training_queue.join()
            with brain_lock:
                checkpoint_path = save_neural_memory_checkpoint()
            print(f"Saved neural memory checkpoint to {checkpoint_path}.")
            print("Exiting chat.")
            break

        _answer_with_memory_context(user_message)


if __name__ == "__main__":
    _start_training_worker()
    if os.path.exists(DEFAULT_CHECKPOINT_PATH):
        with brain_lock:
            checkpoint_path = load_neural_memory_checkpoint()
        print(f"Loaded neural memory checkpoint from {checkpoint_path}.")
    else:
        _start_seed_demo()
    run_chat()
