import torch
import torch.nn as nn

from agno.agent import Agent
from agno.models.ollama import Ollama


class NeuralMemory(nn.Module):
  def __init__(self, embed_dim=128, hidden_dim=256, max_seq_len=512):
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
    self.max_seq_len = max_seq_len

    self.embedding = nn.Embedding(len(self.vocab), embed_dim)
    self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
    decoder_layer = nn.TransformerEncoderLayer(
      d_model=embed_dim,
      nhead=8,
      dim_feedforward=hidden_dim * 2,
      batch_first=True,
    )
    self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)
    self.readout = nn.Linear(embed_dim, len(self.vocab))

    self.facts = []
    self.training_text = ""
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

  def update(self, fact: str, train_steps: int = 40):
    """Append fact to transformer memory and retrain next-token prediction."""
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

    self.training_text = merged_memory
    self.train()
    loss = None
    for _ in range(train_steps):
      self.optimizer.zero_grad()

      dec_emb = self.embedding(decoder_input) + self.pos_embedding(positions)
      decoded = self.decoder(dec_emb, mask=self._causal_mask(seq_len))
      logits = self.readout(decoded)

      loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
      loss.backward()
      nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
      self.optimizer.step()

    return f"Memory Updated. Loss: {loss.item():.4f}" if loss is not None else "Memory update skipped."

  def recall(self, max_len: int = 220, deterministic: bool = True):
    """Recall memory text from facts or decode with transformer generation."""
    if self.facts:
      return self._compose_memory_text()

    if deterministic:
      return self.training_text if self.training_text else "No transformer memory encoded yet."

    self.eval()
    generated_ids = []
    token = torch.tensor([[self.bos_idx]], dtype=torch.long)

    with torch.no_grad():
      for _ in range(max_len):
        seq_len = token.size(1)
        if seq_len > self.max_seq_len:
          break
        positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        dec_emb = self.embedding(token) + self.pos_embedding(positions)
        decoded = self.decoder(dec_emb, mask=self._causal_mask(seq_len))
        logits = self.readout(decoded[:, -1, :])
        next_idx = int(torch.argmax(logits, dim=-1).item())

        if next_idx == self.eos_idx:
          break

        generated_ids.append(next_idx)
        token = torch.cat([token, torch.tensor([[next_idx]], dtype=torch.long)], dim=1)

    decoded = self._ids_to_text(generated_ids).strip()
    return decoded if decoded else "Neural memory decode returned empty text."

# Initialize our transformer memory module.
brain = NeuralMemory()

def write_to_neural_memory(fact: str):
  """Encodes a fact into transformer-based memory."""
  status = brain.update(fact)
  return f"Transformer write complete: {status}"

def read_from_neural_memory():
  """Reads from transformer-based memory (no JSON or database)."""
  memory_text = brain._compose_memory_text() if brain.facts else brain.training_text
  recalled_text = brain.recall()
  return (
    f"Transformer Memory Text: {memory_text if memory_text else '[empty]'}\n"
    f"Recalled Fact From Transformer: {recalled_text}"
  )

agent = Agent(
  model=Ollama(id="gemma4:e2b"),
  tools=[write_to_neural_memory, read_from_neural_memory],
  description="You are a MAG-enhanced Research Assistant.",
  instructions=[
      "1. When the user tells you something important, use 'write_to_neural_memory'.",
      "2. Before answering complex questions, use 'read_from_neural_memory' to inspect transformer memory.",
      "3. Store and recall memory from transformer context only; do not use a database or JSON."
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
    "Use the transformer memory snapshot below as your primary source for user-profile facts. "
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