import ollama
# pyrefly: ignore [missing-import]
import tiktoken
import numpy as np
from typing import List, Dict, Generator

class GruModel:
    """
    A unified model class that uses Ollama for intelligence
    while mathematically simulating a GRU's amnesia using a probabilistic
    Token-Forgetting mechanism over the context window.
    GRU tends to forget slightly faster than LSTM in some contexts due to reset gates.
    """
    def __init__(self, model_name: str = "llama3", max_capacity: int = 4096):
        self.model_name = model_name
        self.max_capacity = max_capacity
        
        # Token-based GRU Memory Simulation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.memory_tokens: List[int] = []
        self.total_tokens_processed = 0
        
        # GRU specific states (more aggressive forgetting)
        self.forget_rate = 0.9995  # 99.95% retention per token step
        
        # Ollama Conversation History (for UI only)
        self.history: List[Dict[str, str]] = []
        self._set_system_prompt()

    def _set_system_prompt(self):
        """Initializes the system prompt defining the AI's behavior."""
        system_prompt = (
            "You are a highly intelligent, concise, and helpful AI assistant. "
            "You always communicate in English. "
            "Your underlying architecture relies on a simulated GRU hidden state."
        )
        self.history.append({"role": "system", "content": system_prompt})

    def _apply_forget_gate(self, num_new_tokens: int):
        """
        Simulates the GRU reset/update gates mathematically.
        Every existing token in memory decays exponentially based on how many
        new tokens are ingested.
        """
        if not self.memory_tokens:
            return
            
        survival_prob = self.forget_rate ** num_new_tokens
        
        # Apply probabilistic dropout mask
        mask = np.random.rand(len(self.memory_tokens)) < survival_prob
        self.memory_tokens = [t for i, t in enumerate(self.memory_tokens) if mask[i]]

    def get_active_tokens_count(self) -> int:
        """Returns the total ingested tokens, so the benchmark graph X-axis aligns correctly."""
        return self.total_tokens_processed
        
    def get_dropped_tokens_count(self) -> int:
        """Returns the number of tokens that have been forgotten (dropped)."""
        return max(0, self.total_tokens_processed - len(self.memory_tokens))

    def get_memory_dump(self) -> str:
        """Returns the actual memory structure currently stored."""
        tokens_str = str(self.memory_tokens[:50])
        if len(self.memory_tokens) > 50:
            tokens_str = tokens_str[:-1] + ", ...]"
            
        dump = (
            "[bold cyan]Architecture:[/bold cyan] [bold]GRU (Gated Recurrent Unit)[/bold]\n"
            f"[bold yellow]Tokens Processed:[/bold yellow] {self.total_tokens_processed}\n"
            f"[bold red]Tokens Forgotten:[/bold red] {self.get_dropped_tokens_count()} [italic](due to Reset/Update Gates)[/italic]\n\n"
            "[bold green]Internal State:[/bold green]\n"
            f"  • [bold]Hidden State (h_t):[/bold] 1 vector of continuous values\n"
            f"  [italic]* GRU does not have a separate Cell State like LSTM[/italic]\n\n"
            "[bold magenta]Retained Token IDs (Simulated Vector):[/bold magenta]\n"
            f"[dim]{tokens_str}[/dim]"
        )
        return dump

    def add_user_message(self, message: str):
        """Appends user message to history and simulates GRU logic."""
        self.history.append({"role": "user", "content": message})
        
        new_tokens = self.tokenizer.encode(f"\nUser: {message}\nAssistant: ")
        num_new = len(new_tokens)
        
        # 1. Decay existing memory
        self._apply_forget_gate(num_new)
        
        # 2. Add new tokens
        self.memory_tokens.extend(new_tokens)
        self.total_tokens_processed += num_new
        
    def add_assistant_message(self, message: str):
        """Appends assistant message to history and simulates GRU logic."""
        self.history.append({"role": "assistant", "content": message})
        
        new_tokens = self.tokenizer.encode(f"{message}\n")
        num_new = len(new_tokens)
        
        # 1. Decay existing memory
        self._apply_forget_gate(num_new)
        
        # 2. Add new tokens
        self.memory_tokens.extend(new_tokens)
        self.total_tokens_processed += num_new

    def generate_response_stream(self) -> Generator[str, None, None]:
        """
        Calls Ollama to generate a response stream based on the degraded memory tokens.
        Yields chunks of the response text.
        """
        # Decode the degraded memory tokens
        try:
            corrupted_context = self.tokenizer.decode(self.memory_tokens, errors="replace")
        except Exception:
            corrupted_context = self.tokenizer.decode(self.memory_tokens)

        inference_messages = [
            self.history[0], # System prompt
            {
                "role": "user", 
                "content": f"Here is the degraded content of your memory buffer:\n\n{corrupted_context}\n\nPlease continue the conversation naturally based on this memory. Do not mention the memory corruption."
            }
        ]
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=inference_messages,
                stream=True,
                options={
                    "num_ctx": self.max_capacity
                }
            )
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
        except Exception as e:
            yield f"\n[Error communicating with Ollama: {str(e)}]"

    def save_memory(self, filepath: str = "memory_gru.pt"):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                "history": self.history,
                "memory_tokens": self.memory_tokens,
                "total_tokens_processed": self.total_tokens_processed
            }, f)

    def load_memory(self, filepath: str = "memory_gru.pt") -> bool:
        import os
        import pickle
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.history = data["history"]
                self.memory_tokens = data.get("memory_tokens", [])
                self.total_tokens_processed = data.get("total_tokens_processed", 0)
            return True
        except Exception:
            return False
            
    def clear_memory(self):
        self.history = []
        self._set_system_prompt()
        self.memory_tokens = []
        self.total_tokens_processed = 0
