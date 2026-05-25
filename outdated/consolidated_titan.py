"""
True Neural Memory System using Google's Titans Architecture.
Reference: "Titans: Learning to Memorize at Test Time" (Behrouz et al., Google Research)

Fixed in-place modification bug by ensuring out-of-place parameter updates during forward passes.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
from typing import List
from agno.agent import Agent
from agno.models.ollama import Ollama

# System Configurations
CPU_THREADS = os.cpu_count() or 1
torch.set_num_threads(CPU_THREADS)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(CPU_THREADS)

RESPONSE_MODEL_ID = os.getenv("RESPONSE_MODEL_ID", "gemma2:2b")
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_CHECKPOINT_PATH = "true_titans_memory.pt"


class TrueTitansLongTermMemory(nn.Module):
    """
    Associative neural memory network M.
    Updates its underlying parameters dynamically via structural gates.
    """
    def __init__(self, d_model: int, lr_inner: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.lr_inner = lr_inner

        # Projection Matrices (Titans §3.1)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        self.proj_q = nn.Linear(d_model, d_model, bias=False)

        # Stable Memory Neural Network M(k) -> v
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Architectural Gates: Evaluated against the hidden representations
        self.gate_surprise = nn.Linear(d_model, 1)
        self.gate_decay = nn.Linear(d_model, 1)

        # Stability normalization
        self.norm = nn.LayerNorm(d_model)

    def retrieve(self, x: torch.Tensor) -> torch.Tensor:
        """Read state from neural weights: y = M(x W_Q)"""
        q = self.proj_q(x)
        return self.norm(self.mlp(q))

    def consolidate_step(self, x_t: torch.Tensor):
        """
        Calculates inline surprise-gated gradients and applies them
        directly to the weight matrices along with localized decay factors.
        """
        k = self.proj_k(x_t)
        v = self.proj_v(x_t)

        # 1. Track gradients locally for the associative objective ||M(k) - v||^2
        with torch.enable_grad():
            pred = self.mlp(k)
            loss = F.mse_loss(pred, v)

            # Extract explicit partial derivatives relative ONLY to the internal MLP layers
            grads = torch.autograd.grad(
                loss, self.mlp.parameters(), retain_graph=False, create_graph=False
            )

        # 2. Evaluate current step gating metrics as simple scalar floats
        surprise = torch.sigmoid(self.gate_surprise(x_t)).item()
        decay = torch.sigmoid(self.gate_decay(x_t)).item() * 0.01

        # 3. Direct Parameter Gradient Injection via out-of-place graph operations
        with torch.no_grad():
            for param, grad in zip(self.mlp.parameters(), grads):
                # Apply localized Gradient Clipping to prevent weight destruction
                clipped_grad = torch.clamp(grad, min=-0.5, max=0.5)

                # Out-of-place parameter creation to bypass PyTorch in-place guardrails
                decayed_weight = param * (1.0 - (decay * self.lr_inner))
                updated_weight = decayed_weight - (self.lr_inner * surprise * clipped_grad)

                # Assign the tracking reference directly
                param.copy_(updated_weight)


class CausalAttentionBlock(nn.Module):
    """Standard sliding window core attention architecture for handling short-term data."""
    def __init__(self, d_model: int, nhead: int = 8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class TrueTitansMemorySystem(nn.Module):
    """The master end-to-end framework managing the long-term and short-term branches."""
    def __init__(self, embed_dim: int = 128, max_seq_len: int = DEFAULT_MAX_SEQ_LEN):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Tokenizer setup (Character-level for self-contained execution)
        self.special_tokens = ["<BOS>", "<EOS>", "<UNK>"]
        self.ascii_tokens = [chr(i) for i in range(32, 127)]
        self.vocab = self.special_tokens + self.ascii_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

        self.bos_idx = self.token_to_idx["<BOS>"]
        self.eos_idx = self.token_to_idx["<EOS>"]
        self.unk_idx = self.token_to_idx["<UNK>"]

        # Structural Layers
        self.embedding = nn.Embedding(len(self.vocab), embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.long_term = TrueTitansLongTermMemory(embed_dim)
        self.short_term_core = nn.ModuleList([
            CausalAttentionBlock(embed_dim),
            CausalAttentionBlock(embed_dim)
        ])
        self.readout = nn.Linear(embed_dim, len(self.vocab))

        # Outer loop optimizer (Handles persistent structures, not volatile memory network weights)
        outer_params = (
            list(self.embedding.parameters())
            + list(self.pos_embedding.parameters())
            + list(self.long_term.proj_k.parameters())
            + list(self.long_term.proj_v.parameters())
            + list(self.long_term.proj_q.parameters())
            + list(self.long_term.gate_surprise.parameters())
            + list(self.long_term.gate_decay.parameters())
            + list(self.short_term_core.parameters())
            + list(self.readout.parameters())
        )
        self.optimizer = torch.optim.Adam(outer_params, lr=0.001)

    def text_to_ids(self, text: str) -> torch.Tensor:
        ids = [self.token_to_idx.get(ch, self.unk_idx) for ch in text]
        return torch.tensor([ids], dtype=torch.long)

    def ids_to_text(self, ids: List[int]) -> str:
        return "".join([self.idx_to_token.get(idx, "") for idx in ids if idx not in [self.bos_idx, self.eos_idx]])

    def forward_and_consolidate(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Executes sequential forward passes, decoding context from historical neural paths
        while updating the memory network weights on surprising steps.
        """
        batch_size, seq_len = token_ids.size()
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        embeddings = self.embedding(token_ids) + self.pos_embedding(positions)

        consolidated_outputs = []

        # Iterate over sequence steps to allow memory states to mutate progressively
        for t in range(seq_len):
            x_t = embeddings[:, t, :] # Shape: [Batch, Embed_Dim]

            # 1. Look up long-term context from the current state of the weights
            # Keep recall detached from the outer CE graph because long-term
            # memory weights are updated online during this same forward pass.
            mem_recall = self.long_term.retrieve(x_t).detach()

            # 2. Blend context into the tracking stream
            combined_representation = x_t + mem_recall
            consolidated_outputs.append(combined_representation.unsqueeze(1))

            # 3. Test-time Consolidation Gate execution
            if self.training:
                # Isolate computational graph tracking to prevent cyclic gradient leakages
                self.long_term.consolidate_step(x_t.detach())

        # Pass aggregated representation through the causal core blocks
        hidden = torch.cat(consolidated_outputs, dim=1)
        for block in self.short_term_core:
            hidden = block(hidden)

        return self.readout(hidden)

    def memorize_fact(self, raw_text: str, epochs: int = 15):
        """Trains the framework structure on sequence dynamics."""
        self.train()
        tokens = self.text_to_ids(raw_text)
        if tokens.size(1) > self.max_seq_len:
            tokens = tokens[:, :self.max_seq_len]

        for _ in range(epochs):
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.forward_and_consolidate(tokens)

            # Standard auto-regressive target shifting
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, len(self.vocab)),
                tokens[:, 1:].reshape(-1)
            )
            loss.backward()
            # Clip system outer gradients for ultimate initialization safety
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def generate_memory_recall(self, max_len: int = 64) -> str:
        """Decodes knowledge stored natively inside the neural weight structures."""
        self.eval()
        generated_ids = []
        token = torch.tensor([[self.bos_idx]], dtype=torch.long)

        with torch.no_grad():
            for _ in range(max_len):
                if token.size(1) > self.max_seq_len:
                    break
                embeddings = self.embedding(token)

                # Fetch output using tracking sequence state
                outputs = []
                for t in range(embeddings.size(1)):
                    x_t = embeddings[:, t, :]
                    mem_recall = self.long_term.retrieve(x_t)
                    outputs.append((x_t + mem_recall).unsqueeze(1))

                hidden = torch.cat(outputs, dim=1)
                for block in self.short_term_core:
                    hidden = block(hidden)

                logits = self.readout(hidden[:, -1, :])
                next_idx = int(torch.argmax(logits, dim=-1).item())

                if next_idx == self.eos_idx:
                    break
                generated_ids.append(next_idx)
                token = torch.cat([token, torch.tensor([[next_idx]], dtype=torch.long)], dim=1)

        return self.ids_to_text(generated_ids).strip()


# Thread-safe application state
brain = TrueTitansMemorySystem()
brain_lock = threading.Lock()

response_agent = Agent(
    model=Ollama(id=RESPONSE_MODEL_ID),
    description="True Titans memory-grounded engine.",
    instructions=[
        "Answer in 1-2 sharp, highly concise sentences.",
        "Rely on the provided Neural Recall facts to address profile declarations.",
        "If the snapshot context is blank, messy, or lacks the answer, state that you do not know.",
    ],
    markdown=True,
)


def save_checkpoint(path: str = DEFAULT_CHECKPOINT_PATH):
    with brain_lock:
        torch.save({"model_state": brain.state_dict()}, path)
    print(f"\n[System] Serialized neural weights directly to {path}.")


def load_checkpoint(path: str = DEFAULT_CHECKPOINT_PATH):
    if os.path.exists(path):
        try:
            brain.load_state_dict(torch.load(path, map_location="cpu")["model_state"])
            print(f"[System] Initialized architecture using weights from {path}.")
            return
        except Exception:
            print("[System] Outdated/corrupted file framework located. Regenerating baseline model layers.")

    print("[System] No historical checkpoint located. Initializing baseline demonstration profile.")
    seed_facts = [
        "My name is Salekin.",
        "I was born in December 1991.",
        "I am a student of Greenwich Danang.",
        "I live in Danang, Vietnam.",
    ]
    for fact in seed_facts:
        brain.memorize_fact(fact)


def handle_user_interaction(user_message: str):
    """Evaluates facts, applies test-time parameter steps, and routes to LLM."""
    is_profile_fact = any(marker in user_message.lower() for marker in ["i am", "i'm", "my name", "i live", "remember"])

    if is_profile_fact and not any(neg in user_message.lower() for neg in ["?", "forget", "dont save"]):
        print("[Neural Memory] Surprise-Gating active. Consolidating sequence to neural weights...")
        with brain_lock:
            brain.memorize_fact(user_message)

    # Read tracking context natively out of weights
    with brain_lock:
        neural_recall_context = brain.generate_memory_recall()

    print(f"\033[94m[Neural Weights Retrieval]: {neural_recall_context or '[Empty Vector Space]'}\033[0m")

    prompt = (
        f"Neural Recall: {neural_recall_context}\n\n"
        f"User Input: {user_message}"
    )
    response_agent.print_response(prompt)


def main():
    load_checkpoint()
    print("\nTrue Titans Chat Interface initialized. Type 'exit' to terminate session.")
    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            save_checkpoint()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            save_checkpoint()
            break

        handle_user_interaction(user_input)


if __name__ == "__main__":
    main()
