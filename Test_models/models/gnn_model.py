import ollama
# pyrefly: ignore [missing-import]
import tiktoken
import numpy as np
from typing import List, Dict, Generator

class GnnModel:
    """
    A unified model class that uses Ollama for intelligence
    while mathematically simulating a Graph Neural Network (GNN) based memory.
    It treats tokens as nodes in a graph and simulates amnesia through 
    probabilistic Node Pruning (dropping nodes/edges as context grows).
    """
    def __init__(self, model_name: str = "llama3", max_capacity: int = 4096):
        self.model_name = model_name
        self.max_capacity = max_capacity
        
        # Token-based GNN Memory Simulation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.memory_nodes: List[int] = []  # Tokens treated as graph nodes
        self.total_tokens_processed = 0
        
        # GNN specific states
        self.node_survival_rate = 0.9997  # 99.97% node retention per token step
        
        # Ollama Conversation History (for UI only)
        self.history: List[Dict[str, str]] = []
        self._set_system_prompt()

    def _set_system_prompt(self):
        """Initializes the system prompt defining the AI's behavior."""
        system_prompt = (
            "You are a highly intelligent, concise, and helpful AI assistant. "
            "You always communicate in English. "
            "Your underlying architecture relies on a simulated Graph Neural Network (GNN) memory structure."
        )
        self.history.append({"role": "system", "content": system_prompt})

    def _apply_node_pruning(self, num_new_nodes: int):
        """
        Simulates GNN Message Passing Decay and Edge Pruning mathematically.
        Existing nodes (tokens) in the graph have a probability of becoming
        isolated and dropped based on how many new nodes are ingested.
        """
        if not self.memory_nodes:
            return
            
        survival_prob = self.node_survival_rate ** num_new_nodes
        
        # Apply probabilistic dropout mask (Node Isolation)
        mask = np.random.rand(len(self.memory_nodes)) < survival_prob
        self.memory_nodes = [t for i, t in enumerate(self.memory_nodes) if mask[i]]

    def get_active_tokens_count(self) -> int:
        """Returns the total ingested tokens, so the benchmark graph X-axis aligns correctly."""
        return self.total_tokens_processed
        
    def get_dropped_tokens_count(self) -> int:
        """Returns the number of nodes/tokens that have been pruned."""
        return max(0, self.total_tokens_processed - len(self.memory_nodes))

    def get_memory_dump(self) -> str:
        """Returns the actual memory structure currently stored."""
        tokens_str = str(self.memory_nodes[:50])
        if len(self.memory_nodes) > 50:
            tokens_str = tokens_str[:-1] + ", ...]"
            
        dump = (
            "[bold cyan]Architecture:[/bold cyan] [bold]GNN (Graph Neural Network)[/bold]\n"
            f"[bold yellow]Nodes Processed:[/bold yellow] {self.total_tokens_processed}\n"
            f"[bold red]Nodes Pruned:[/bold red] {self.get_dropped_tokens_count()} [italic](due to Message Passing Decay)[/italic]\n\n"
            "[bold green]Internal Graph State:[/bold green]\n"
            f"  • [bold]Active Nodes (V):[/bold] {len(self.memory_nodes)} token nodes\n"
            f"  • [bold]Simulated Edges (E):[/bold] ~{len(self.memory_nodes) * 2} connections\n\n"
            "[bold magenta]Retained Node IDs (Tokens):[/bold magenta]\n"
            f"[dim]{tokens_str}[/dim]"
        )
        return dump

    def add_user_message(self, message: str):
        """Appends user message to history and simulates GNN logic."""
        self.history.append({"role": "user", "content": message})
        
        new_tokens = self.tokenizer.encode(f"\nUser: {message}\nAssistant: ")
        num_new = len(new_tokens)
        
        # 1. Decay existing graph memory
        self._apply_node_pruning(num_new)
        
        # 2. Add new nodes
        self.memory_nodes.extend(new_tokens)
        self.total_tokens_processed += num_new
        
    def add_assistant_message(self, message: str):
        """Appends assistant message to history and simulates GNN logic."""
        self.history.append({"role": "assistant", "content": message})
        
        new_tokens = self.tokenizer.encode(f"{message}\n")
        num_new = len(new_tokens)
        
        # 1. Decay existing graph memory
        self._apply_node_pruning(num_new)
        
        # 2. Add new nodes
        self.memory_nodes.extend(new_tokens)
        self.total_tokens_processed += num_new

    def generate_response_stream(self) -> Generator[str, None, None]:
        """
        Calls Ollama to generate a response stream based on the degraded node buffer.
        Yields chunks of the response text.
        """
        # Decode the degraded memory nodes
        try:
            corrupted_context = self.tokenizer.decode(self.memory_nodes, errors="replace")
        except Exception:
            corrupted_context = self.tokenizer.decode(self.memory_nodes)

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

    def save_memory(self, filepath: str = "memory_gnn.pt"):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                "history": self.history,
                "memory_nodes": self.memory_nodes,
                "total_tokens_processed": self.total_tokens_processed
            }, f)

    def load_memory(self, filepath: str = "memory_gnn.pt") -> bool:
        import os
        import pickle
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.history = data["history"]
                self.memory_nodes = data.get("memory_nodes", [])
                self.total_tokens_processed = data.get("total_tokens_processed", 0)
            return True
        except Exception:
            return False
            
    def clear_memory(self):
        self.history = []
        self._set_system_prompt()
        self.memory_nodes = []
        self.total_tokens_processed = 0
