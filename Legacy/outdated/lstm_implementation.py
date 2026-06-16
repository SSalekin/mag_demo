import os
import queue
import threading
from typing import List

import torch
import torch.nn as nn
from agno.agent import Agent
from agno.models.ollama import Ollama


# -----------------------------
# Runtime configuration
# -----------------------------
CPU_THREADS = os.cpu_count() or 1
MAX_MEMORY_CHARS = 500
# RESPONSE_MODEL_ID = os.getenv("RESPONSE_MODEL_ID", "gemma2:2b")
RESPONSE_MODEL_ID = os.getenv("RESPONSE_MODEL_ID", "gemma4:e2b")
DEFAULT_TRAIN_STEPS = 5
DEFAULT_MAX_SEQ_LEN = 512

# Use all available CPU threads for LSTM training/inference when possible.
torch.set_num_threads(CPU_THREADS)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(CPU_THREADS)


class NeuralMemory(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, max_seq_len: int = DEFAULT_MAX_SEQ_LEN):
        super().__init__()

        # ASCII + control tokens for sequence generation.
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
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_dim, len(self.vocab))
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
        """Append fact to LSTM memory and retrain next-token prediction."""
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
            lstm_out, _ = self.lstm(dec_emb)
            logits = self.readout(lstm_out)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

        return f"Memory Updated. Loss: {loss.item():.4f}" if loss is not None else "Memory update skipped."

    def recall(self, max_len: int = 220, deterministic: bool = True) -> str:
        """Recall memory text from facts or decode with LSTM generation."""
        if self.facts:
            return self._compose_memory_text()

        if deterministic:
            return self.training_text or "No LSTM memory encoded yet."

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
                lstm_out, _ = self.lstm(dec_emb)
                logits = self.readout(lstm_out[:, -1, :])
                next_idx = int(torch.argmax(logits, dim=-1).item())

                if next_idx == self.eos_idx:
                    break
                generated_ids.append(next_idx)
                token = torch.cat([token, torch.tensor(
                    [[next_idx]], dtype=torch.long)], dim=1)

        decoded = self._ids_to_text(generated_ids).strip()
        return decoded if decoded else "Neural memory decode returned empty text."


# -----------------------------
# App state
# -----------------------------
brain = NeuralMemory()
brain_lock = threading.Lock()
training_queue: "queue.Queue[str]" = queue.Queue()

latest_training_status = "No training runs yet."
latest_memory_text = ""
latest_recalled_text = "No LSTM memory encoded yet."


def write_to_neural_memory(fact: str) -> str:
    """Queue a fact for async LSTM memory training."""
    training_queue.put(fact)
    return "LSTM training queued in background."


def read_from_neural_memory() -> str:
    """Read cached neural memory without blocking user response path."""
    global latest_memory_text, latest_recalled_text

    # Try refresh snapshot opportunistically; skip if trainer holds the lock.
    if brain_lock.acquire(blocking=False):
        try:
            latest_memory_text = brain._compose_memory_text(
            ) if brain.facts else brain.training_text
            latest_recalled_text = brain.recall()
        finally:
            brain_lock.release()

    memory_text = latest_memory_text[-MAX_MEMORY_CHARS:]
    recalled_text = latest_recalled_text[-MAX_MEMORY_CHARS:]
    return (
        f"LSTM Memory Text: {memory_text if memory_text else '[empty]'}\n"
        f"Recalled Fact From LSTM: {recalled_text}"
    )


def _training_worker() -> None:
    global latest_training_status, latest_memory_text, latest_recalled_text

    while True:
        fact = training_queue.get()
        if fact is None:
            training_queue.task_done()
            break

        try:
            with brain_lock:
                latest_training_status = brain.update(fact)
                latest_memory_text = brain._compose_memory_text(
                ) if brain.facts else brain.training_text
                latest_recalled_text = brain.recall()
        except Exception as err:
            latest_training_status = f"Background training failed: {err}"
        finally:
            training_queue.task_done()


def _start_training_worker() -> None:
    thread = threading.Thread(target=_training_worker,
                              daemon=True, name="memory-trainer")
    thread.start()


response_agent = Agent(
    model=Ollama(id=RESPONSE_MODEL_ID),
    description="Fast memory-grounded assistant.",
    instructions=[
        "Answer in 1-3 concise sentences.",
        "Use only the provided LSTM memory snapshot for profile facts.",
        "If the snapshot has no fact, say you do not have that information yet.",
    ],
    markdown=True,
)


def _should_memorize(user_message: str) -> bool:
    """Simple heuristic for autobiographical facts worth writing to memory."""
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
        # Keep response path strictly non-blocking.
        threading.Thread(
            target=write_to_neural_memory,
            args=(user_message,),
            daemon=True,
            name="memorize-dispatch",
        ).start()


def _answer_with_memory_context(user_message: str) -> None:
    memory_snapshot = read_from_neural_memory()
    training_status = f"Queue length: {training_queue.qsize()} | Last training status: {latest_training_status}"
    grounded_prompt = (
        "Answer briefly using this memory snapshot.\n\n"
        f"Training Status: {training_status}\n"
        f"{memory_snapshot}\n\n"
        f"User message: {user_message}"
    )
    response_agent.print_response(grounded_prompt)
    _queue_memorize_if_needed(user_message)


def run_seed_demo() -> None:
    """Seed neural memory in background without LLM response calls."""
    seed_facts = [
        "My name is Salekin",
        "I was born in December 1991",
        "I am a student of Greenwich Danang",
        "I am working on a reasearch paper about Memory Augmented Generation",
        "Remember, I live in Danang city and Better Call Saul is the best tv show ever.",
        "Danang is the most beautiful city of Vietnam.",
    ]
    for fact in seed_facts:
        write_to_neural_memory(fact)


def _start_seed_demo() -> None:
    thread = threading.Thread(target=run_seed_demo,
                              daemon=True, name="seed-demo")
    thread.start()


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
            print("Exiting chat.")
            break

        _answer_with_memory_context(user_message)


if __name__ == "__main__":
    _start_training_worker()
    _start_seed_demo()
    run_chat()
