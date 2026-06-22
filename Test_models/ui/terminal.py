import os
# pyrefly: ignore [missing-import]
import tiktoken
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from typing import List, Tuple

class TUI:
    """
    Manages the Terminal User Interface using the `rich` library.
    It provides a Dashboard for the KV-Cache memory state and a Chat panel.
    """
    def __init__(self):
        self.console = Console()
        self.chat_history: List[Tuple[str, str, bool]] = [] # List of (role, message, tracked)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def clear_screen(self):
        """Clears the terminal screen for redrawing."""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def clear_chat(self):
        """Wipes the chat history."""
        self.chat_history = []
        
    def add_chat_message(self, role: str, message: str, tracked: bool = True):
        """Stores a message in the UI's local chat history."""
        self.chat_history.append((role, message, tracked))
        
    def render(self, active_tokens: int, max_capacity: int, dropped_tokens: int = 0, arch_name: str = "Transformer KV-Cache", model_name: str = "llama3"):
        """
        Clears the screen and renders the Dashboard and Chat panels.
        """
        self.clear_screen()
        
        # 1. Create Dashboard Panel
        percentage = (active_tokens / max_capacity) * 100
        dashboard_text = Text()
        dashboard_text.append(f"Model: Ollama ({model_name}) | Memory: Simulated {arch_name}\n", style="bold cyan")
        dashboard_text.append(f"Active Tokens in KV-Cache: ", style="yellow")
        
        # Color code the token count based on usage
        count_style = "green" if percentage < 75 else ("orange1" if percentage < 95 else "red")
        dashboard_text.append(f"{active_tokens} / {max_capacity} ", style=count_style)
        dashboard_text.append(f"({percentage:.1f}%)", style=count_style)
        
        dashboard_panel = Panel(
            dashboard_text,
            title="[bold blue]System Dashboard[/bold blue]",
            border_style="blue"
        )
        
        # 2. Create Chat Panel
        chat_display = Text()
        dropped_remaining = dropped_tokens
        
        for role, msg, tracked in self.chat_history:
            if role == "user":
                chat_display.append("\nUser: ", style="bold green")
            elif role == "assistant":
                chat_display.append("\nAI: ", style="bold magenta")
                
            if not tracked:
                if role == "system":
                    chat_display.append(Text.from_markup(msg + "\n"))
                else:
                    chat_display.append(msg + "\n", style="white")
                continue
                
            if dropped_remaining > 0:
                tokens = self.tokenizer.encode(msg)
                if len(tokens) <= dropped_remaining:
                    chat_display.append(msg + "\n", style="red")
                    dropped_remaining -= len(tokens)
                else:
                    dropped_part = self.tokenizer.decode(tokens[:dropped_remaining])
                    kept_part = self.tokenizer.decode(tokens[dropped_remaining:])
                    chat_display.append(dropped_part, style="red")
                    chat_display.append(kept_part + "\n", style="white")
                    dropped_remaining = 0
            else:
                chat_display.append(msg + "\n", style="white")
                
        if not self.chat_history:
            chat_display.append("No messages yet. Start typing to interact!", style="italic dim")
            
        chat_panel = Panel(
            chat_display,
            title="[bold magenta]Chat Interface[/bold magenta] (Type 'exit' or 'quit' to close)",
            border_style="magenta",
            expand=True
        )
        
        # Print panels
        self.console.print(dashboard_panel)
        self.console.print(chat_panel)
