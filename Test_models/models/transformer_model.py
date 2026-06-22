import ollama
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer
import numpy as np
from typing import List, Dict, Generator

class TransformerModel:
    """
    A single unified model class that uses Ollama for intelligence
    while mathematically simulating a Transformer KV-Cache constraint locally.
    """
    def __init__(self, model_name: str = "llama3", max_capacity: int = 8192, embedding_dim: int = 4096):
        self.model_name = model_name
        self.max_capacity = max_capacity
        self.embedding_dim = embedding_dim
        
        # Memory Simulation (KV-Cache)
        self.tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")
        self.token_ids = []
        self.total_tokens_processed = 0
        self.keys = []
        self.values = []
        
        # Ollama Conversation History
        self.history: List[Dict[str, str]] = []
        self._set_system_prompt()

    def _set_system_prompt(self):
        """Initializes the system prompt defining the AI's behavior."""
        system_prompt = (
            "You are a highly intelligent, concise, and helpful AI assistant, integrated into the file 'transformer_model.py'. "
            "You always communicate exclusively in English. Your model is 'llama3', running locally via Ollama. "
            "Your underlying architecture mathematically simulates the real constraints of the Llama 3 model: "
            "a strict maximum context window of 8192 tokens, and embedding dimensions of 4096. "
            "You must act knowing that your distant past context is continuously erased due to a sliding window (FIFO) mechanism once the 8192 token limit is reached. "
            "If a previous session is detected (memory.pt restored), you must immediately reconnect with the saved context and history. "
            "If the user asks how you work, explicitly explain that you locally simulate the real tensor structures of Llama 3 "
            "(8192 context tokens, 4096 dimension embeddings) while delegating the actual inference to Ollama. "
            "Be direct and get straight to the point to preserve your 8192 token context window."
        )
        self.history.append({"role": "system", "content": system_prompt})

    def _simulate_kv_cache(self):
        """
        Recalculates all cache token IDs on each update 
        to avoid any asymmetric index drift with the text history.
        """
        active_messages = self.history[1:]
        if not active_messages:
            self.token_ids = []
            self.keys = []
            self.values = []
            return
            
        # 1. Recalculate the entire token sequence
        full_token_ids = self.tokenizer.apply_chat_template(active_messages, add_generation_prompt=False)
        
        # For interface statistics (total processed)
        latest_message_tokens = self.tokenizer.apply_chat_template([self.history[-1]], add_generation_prompt=False)
        self.total_tokens_processed += len(latest_message_tokens)
        
        # 2. Full cache replacement
        self.token_ids = full_token_ids
        
        # Ultra-fast vectorized tensor generation
        self.keys = list(np.random.randn(len(self.token_ids), self.embedding_dim))
        self.values = list(np.random.randn(len(self.token_ids), self.embedding_dim))
            
        # 3. Enforce max capacity (Sliding Window / FIFO) locally
        if len(self.token_ids) > self.max_capacity:
            overflow = len(self.token_ids) - self.max_capacity
            self.token_ids = self.token_ids[overflow:]
            self.keys = self.keys[overflow:]
            self.values = self.values[overflow:]

        # 4. Call pruning to synchronize self.history with the truncated size
        self._prune_history()

    def get_active_tokens_count(self) -> int:
        """Returns the number of tokens currently active in the simulated cache."""
        return len(self.token_ids)
        
    def get_dropped_tokens_count(self) -> int:
        """Returns the number of tokens that have been forgotten (dropped) from the sliding window."""
        return max(0, self.total_tokens_processed - len(self.token_ids))

    def get_memory_dump(self) -> str:
        """Returns the actual memory structure currently stored."""
        tokens_str = str(self.token_ids[:30])
        if len(self.token_ids) > 30:
            tokens_str = tokens_str[:-1] + ", ...]"
            
        dump = (
            "[bold cyan]Architecture:[/bold cyan] [bold]Transformer (KV-Cache)[/bold]\n"
            f"[bold yellow]Tokens in Cache:[/bold yellow] {len(self.token_ids)} / {self.max_capacity} [italic](Sliding Window FIFO)[/italic]\n\n"
            "[bold green]Tensor State:[/bold green]\n"
            f"  • [bold]Keys Matrix :[/bold] {len(self.keys)} vectors of dimension {self.embedding_dim}\n"
            f"  • [bold]Values Matrix:[/bold] {len(self.values)} vectors of dimension {self.embedding_dim}\n\n"
            "[bold magenta]Raw Token IDs (Vector):[/bold magenta]\n"
            f"[dim]{tokens_str}[/dim]"
        )
        return dump

    def _prune_history(self):
        """
        Forces self.history to forget the oldest messages (excluding system)
        by strictly synchronizing with the number of active tokens in the KV-Cache.
        """
        while len(self.history) > 1:
            # Extract active discussion messages (excluding system prompt at index 0)
            active_messages = self.history[1:]
            
            # SOLUTION: Measure the REAL size by applying the Llama 3 chat template
            formatted_tokens = self.tokenizer.apply_chat_template(active_messages, add_generation_prompt=False)
            text_tokens = len(formatted_tokens)
            
            # If the formatted text size exceeds the actual cache (after max_capacity restriction), evict the oldest.
            # Keep the safety check (len == 2) to avoid purging the entire context at once.
            if text_tokens > len(self.token_ids):
                if len(self.history) == 2:
                    break
                self.history.pop(1)  # Removes the oldest user or assistant message
            else:
                break

    def add_user_message(self, message: str):
        """Appends user message to history and simulates KV allocation."""
        self.history.append({"role": "user", "content": message})
        self._simulate_kv_cache()
        
    def add_assistant_message(self, message: str):
        """Appends assistant message to history and simulates KV allocation."""
        self.history.append({"role": "assistant", "content": message})
        self._simulate_kv_cache()

    def generate_response_stream(self) -> Generator[str, None, None]:
        """
        Calls Ollama to generate a response stream based on the current history.
        Yields chunks of the response text.
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=self.history,
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

    def save_memory(self, filepath: str = "memory.pt"):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                "history": self.history,
                "token_ids": self.token_ids,
                "total_tokens_processed": self.total_tokens_processed,
                "keys": self.keys,
                "values": self.values
            }, f)

    def load_memory(self, filepath: str = "memory.pt") -> bool:
        import os
        import pickle
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.history = data["history"]
                self.token_ids = data["token_ids"]
                self.total_tokens_processed = data["total_tokens_processed"]
                self.keys = data["keys"]
                self.values = data["values"]
            return True
        except Exception:
            return False
            
    def clear_memory(self):
        self.history = []
        self._set_system_prompt()
        self.token_ids = []
        self.keys = []
        self.values = []
        self.total_tokens_processed = 0
