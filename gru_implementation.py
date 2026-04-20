import torch
import torch.nn as nn
import torch.nn.functional as F

from agno.agent import Agent
from agno.models.ollama import Ollama


class NeuralMemory(nn.Module):
  def __init__(self, embed_dim=128, hidden_dim=256):
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

    self.embedding = nn.Embedding(len(self.vocab), embed_dim)
    self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
    self.decoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
    self.readout = nn.Linear(hidden_dim, len(self.vocab))

    self.state = torch.zeros(1, 1, hidden_dim)
    self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

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

  def update(self, fact: str, train_steps: int = 120):
    """Append fact to neural memory and retrain decode target from GRU state."""
    existing = self.recall(max_len=400)
    if existing.startswith("No neural memory") or existing.startswith("Neural memory decode"):
      merged_memory = fact
    elif fact in existing:
      merged_memory = existing
    else:
      merged_memory = f"{existing} | {fact}"

    target = self._text_to_ids(merged_memory)
    bos = torch.tensor([[self.bos_idx]], dtype=torch.long)
    decoder_input = torch.cat([bos, target[:, :-1]], dim=1)

    self.train()
    for _ in range(train_steps):
      self.optimizer.zero_grad()

      _, hidden = self.encoder(self.embedding(target))
      decoder_output, _ = self.decoder(self.embedding(decoder_input), hidden)
      logits = self.readout(decoder_output)

      loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
      loss.backward()
      self.optimizer.step()

    # Refresh neural state after training.
    with torch.no_grad():
      _, self.state = self.encoder(self.embedding(target))

    return f"Memory Updated. Loss: {loss.item():.4f}"

  def recall(self, max_len: int = 220):
    """Decode remembered text from GRU state only."""
    if torch.allclose(self.state, torch.zeros_like(self.state)):
      return "No neural memory encoded yet."

    self.eval()
    generated_ids = []
    hidden = self.state.clone()
    token = torch.tensor([[self.bos_idx]], dtype=torch.long)

    with torch.no_grad():
      for _ in range(max_len):
        out, hidden = self.decoder(self.embedding(token), hidden)
        logits = self.readout(out[:, -1, :])
        next_idx = int(torch.argmax(logits, dim=-1).item())

        if next_idx == self.eos_idx:
          break

        generated_ids.append(next_idx)
        token = torch.tensor([[next_idx]], dtype=torch.long)

    decoded = self._ids_to_text(generated_ids).strip()
    return decoded if decoded else "Neural memory decode returned empty text."

# Initialize our little brain
brain = NeuralMemory()

def write_to_neural_memory(fact: str):
  """Encodes a fact into GRU parameters and hidden state only."""
  status = brain.update(fact)
  return f"Neural write complete: {status}"

def read_from_neural_memory():
  """Reads from GRU-only memory (no JSON or database)."""
  state_sample = brain.state.flatten().tolist()[:5]
  recalled_text = brain.recall()
  return (
    f"Neural State Vector (Sample): {state_sample}\n"
    f"Recalled Fact From GRU: {recalled_text}"
  )

agent = Agent(
  model=Ollama(id="gemma4:e2b"),
  tools=[write_to_neural_memory, read_from_neural_memory],
  description="You are a MAG-enhanced Research Assistant.",
  instructions=[
      "1. When the user tells you something important, use 'write_to_neural_memory'.",
      "2. Before answering complex questions, use 'read_from_neural_memory' to inspect GRU memory.",
      "3. Store and recall memory from GRU state only; do not use a database or JSON."
  ],
  markdown=True
)

def _should_memorize(user_message: str) -> bool:
  """Simple heuristic for autobiographical facts worth writing to memory."""
  lower = user_message.lower()
  blocked_markers = ["?", "don't save"]
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