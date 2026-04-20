import torch
import torch.nn as nn
import torch.nn.functional as F

from agno.agent import Agent
from agno.models.ollama import Ollama


class TitanDecoder(nn.Module):
  """Lightweight TITAN-style recurrent decoder (no Transformer blocks)."""

  def __init__(self, embed_dim: int, hidden_dim: int):
    super().__init__()
    self.input_proj = nn.Linear(embed_dim, hidden_dim)
    self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
    self.output_proj = nn.Linear(hidden_dim, embed_dim)
    self.gate = nn.Linear(hidden_dim, hidden_dim)

  def forward(self, x: torch.Tensor, context: torch.Tensor):
    context_seq = context.unsqueeze(1).expand(-1, x.size(1), -1)
    projected = self.input_proj(x)
    mixed = torch.tanh(projected + context_seq)
    recurrent, _ = self.rnn(mixed)
    gated = torch.sigmoid(self.gate(context_seq)) * recurrent + recurrent
    return self.output_proj(gated)


class NeuralMemory(nn.Module):
  def __init__(self, embed_dim=128, hidden_dim=256, num_slots=32, max_seq_len=512):
    super().__init__()

    # ASCII plus control tokens for sequence generation.
    self.special_tokens = ["<BOS>", "<EOS>", "<UNK>"]
    self.ascii_tokens = [chr(i) for i in range(32, 127)]
    self.vocab = self.special_tokens + self.ascii_tokens
    self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
    self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

    self.bos_idx = self.token_to_idx["<BOS>"]
    self.eos_idx = self.token_to_idx["<EOS>"]
    self.unk_idx = self.token_to_idx["<UNK>"]
    self.hidden_dim = hidden_dim
    self.max_seq_len = max_seq_len

    self.embedding = nn.Embedding(len(self.vocab), embed_dim)
    self.encoder_proj = nn.Linear(embed_dim, hidden_dim)
    self.key_proj = nn.Linear(hidden_dim, hidden_dim)
    self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    self.memory_slots = nn.Parameter(torch.zeros(num_slots, hidden_dim))
    nn.init.xavier_uniform_(self.memory_slots)

    self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
    self.decoder = TitanDecoder(embed_dim=embed_dim, hidden_dim=hidden_dim)
    self.context_proj = nn.Linear(hidden_dim, embed_dim)
    self.readout = nn.Linear(embed_dim, len(self.vocab))

    self.state = torch.zeros(1, hidden_dim)
    self.facts = []
    self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

  def _text_to_ids(self, text: str):
    ids = [self.token_to_idx.get(ch, self.unk_idx) for ch in text]
    ids.append(self.eos_idx)
    return torch.tensor([ids], dtype=torch.long)

  def _ids_to_text(self, ids):
    chars = []
    for idx in ids:
      token = self.idx_to_token.get(idx, "")
      if token in self.special_tokens:
        continue
      chars.append(token)
    return "".join(chars)

  def _causal_mask(self, seq_len: int):
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

  def _compose_memory_text(self):
    return " | ".join(self.facts)

  def _read_memory(self, query: torch.Tensor):
    keys = self.key_proj(self.memory_slots)
    scores = torch.matmul(query, keys.T) / (self.hidden_dim ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    values = self.value_proj(self.memory_slots)
    context = torch.matmul(weights, values)
    return context

  def _write_memory(self, query: torch.Tensor):
    with torch.no_grad():
      keys = self.key_proj(self.memory_slots)
      scores = torch.matmul(query, keys.T) / (self.hidden_dim ** 0.5)
      weights = torch.softmax(scores, dim=-1).squeeze(0)
      candidate = torch.tanh(self.value_proj(query)).squeeze(0)
      self.memory_slots.mul_(0.995)
      self.memory_slots.add_(weights.unsqueeze(-1) * candidate.unsqueeze(0))

  def update(self, fact: str, train_steps: int = 40):
    """Append fact to Titan memory and retrain decode target from memory state."""
    clean_fact = fact.strip()
    if not clean_fact:
      return "Memory update skipped: empty fact."

    if clean_fact not in self.facts:
      self.facts.append(clean_fact)

    merged_memory = self._compose_memory_text()

    target = self._text_to_ids(merged_memory)
    while target.size(1) > self.max_seq_len and len(self.facts) > 1:
      self.facts.pop(0)
      merged_memory = self._compose_memory_text()
      target = self._text_to_ids(merged_memory)

    bos = torch.tensor([[self.bos_idx]], dtype=torch.long)
    decoder_input = torch.cat([bos, target[:, :-1]], dim=1)
    seq_len = decoder_input.size(1)
    positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    if seq_len > self.max_seq_len:
      return "Memory update skipped: sequence exceeds max sequence length."

    self.train()
    for _ in range(train_steps):
      self.optimizer.zero_grad()

      target_emb = self.embedding(target)
      query = self.encoder_proj(target_emb.mean(dim=1))
      self._write_memory(query.detach())
      context = self._read_memory(query)

      dec_emb = self.embedding(decoder_input) + self.pos_embedding(positions)
      conditioned = dec_emb + self.context_proj(context).unsqueeze(1)
      decoded = self.decoder(conditioned, context)
      logits = self.readout(decoded)

      loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
      loss.backward()
      nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
      self.optimizer.step()

    # Refresh neural state after training.
    with torch.no_grad():
      target_emb = self.embedding(target)
      query = self.encoder_proj(target_emb.mean(dim=1))
      self.state = self._read_memory(query)

    return f"Memory Updated. Loss: {loss.item():.4f}"

  def recall(self, max_len: int = 220):
    """Decode remembered text from Titan memory state only."""
    if self.facts:
      return self._compose_memory_text()

    if torch.allclose(self.state, torch.zeros_like(self.state)):
      return "No neural memory encoded yet."

    self.eval()
    generated_ids = []
    context = self.state.clone()
    token = torch.tensor([[self.bos_idx]], dtype=torch.long)

    with torch.no_grad():
      for _ in range(max_len):
        seq_len = token.size(1)
        if seq_len > self.max_seq_len:
          break
        positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        dec_emb = self.embedding(token) + self.pos_embedding(positions)
        conditioned = dec_emb + self.context_proj(context).unsqueeze(1)
        decoded = self.decoder(conditioned, context)
        logits = self.readout(decoded[:, -1, :])
        next_idx = int(torch.argmax(logits, dim=-1).item())

        if next_idx == self.eos_idx:
          break

        generated_ids.append(next_idx)
        token = torch.cat([token, torch.tensor([[next_idx]], dtype=torch.long)], dim=1)

    decoded = self._ids_to_text(generated_ids).strip()
    return decoded if decoded else "Neural memory decode returned empty text."

# Initialize our little Titan brain.
brain = NeuralMemory()

def write_to_neural_memory(fact: str):
  """Encodes a fact into Titan memory slots and state only."""
  status = brain.update(fact)
  return f"Neural write complete: {status}"

def read_from_neural_memory():
  """Reads from Titan-only memory (no JSON or database)."""
  state_sample = brain.state.flatten().tolist()[:5]
  recalled_text = brain.recall()
  return (
    f"Neural State Vector (Sample): {state_sample}\n"
    f"Recalled Fact From Titan: {recalled_text}"
  )

agent = Agent(
  model=Ollama(id="gemma4:e2b"),
  tools=[write_to_neural_memory, read_from_neural_memory],
  description="You are a MAG-enhanced Research Assistant.",
  instructions=[
      "1. When the user tells you something important, use 'write_to_neural_memory'.",
      "2. Before answering complex questions, use 'read_from_neural_memory' to inspect Titan memory.",
      "3. Store and recall memory from Titan state only; do not use a database or JSON."
  ],
  markdown=True
)

def _should_memorize(user_message: str) -> bool:
  """Simple heuristic for autobiographical facts worth writing to memory."""
  lower = user_message.lower()
  blocked_markers = ["?", "don't save", "dont save", "forget", "don't remember", "dont remember"]
  if any(marker in lower for marker in blocked_markers):
    return False
  fact_markers = [
    "i am ",
    "i'm ",
    "i live",
    "i study",
    "my ",
    "remember",
    "currently"
  ]
  return any(marker in lower for marker in fact_markers)

def _answer_with_memory_context(user_message: str):
  # Write first when message looks like a personal fact.
  if _should_memorize(user_message):
    write_to_neural_memory(user_message)

  memory_snapshot = read_from_neural_memory()
  grounded_prompt = (
    "Use the neural memory snapshot below as your primary source for user-profile facts. "
    "If the question asks about identity/location/preferences, answer from the snapshot.\n\n"
    f"{memory_snapshot}\n\n"
    f"User message: {user_message}"
  )
  agent.print_response(grounded_prompt)

def run_seed_demo():
  """Optional demo prompts for quick manual testing."""
  agent.print_response("My name is Salekin")
  agent.print_response("I was born in December 1991")
  agent.print_response("I am a student of Greenwich Danang")
  agent.print_response("I am working on a reasearch paper about Memory Augmented Generation")
  agent.print_response("Remember, I live in Danang city and Better Call Saul is the best tv show ever.")
  agent.print_response("Danang is the most beautiful city of Vietnam.")

def run_chat():
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
  run_seed_demo()
  run_chat()