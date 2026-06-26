#!/usr/bin/env python3
"""MAG Demo main entry point.

Default behavior is now a unified natural chat interface:

    python main.py

The historical model-selection menu is still available for old experiments:

    python main.py --legacy-menu
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def legacy_architecture_menu() -> int:
    """Run the original architecture-selection terminal UI."""

    from models.transformer_model import TransformerModel
    from models.lstm_model import LstmModel
    from models.gru_model import GruModel
    from models.gnn_model import GnnModel
    from models.mamba_model import MambaModel
    from models.titan_model import TitanModel
    from ui.terminal import TUI

    max_kv_capacity = 8192
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

    print("\n--- Legacy Architecture Selection ---")
    print("1. Transformer (KV-Cache)")
    print("2. LSTM (Hidden State)")
    print("3. GRU (Hidden State)")
    print("4. GNN (Graph Neural Network)")
    print("5. Mamba (External Memory)")
    print("6. Titan (External Neural Memory)")
    while True:
        choice = input("Select an architecture (1, 2, 3, 4, 5 or 6): ").strip()
        if choice in {"1", "2", "3", "4", "5", "6"}:
            break
        print("Invalid choice. Please enter 1, 2, 3, 4, 5 or 6.")

    if choice == "1":
        model_class = TransformerModel
        memory_file = "memory.pt"
        arch_name = "Transformer"
    elif choice == "2":
        model_class = LstmModel
        memory_file = "memory_lstm.pt"
        arch_name = "LSTM"
    elif choice == "3":
        model_class = GruModel
        memory_file = "memory_gru.pt"
        arch_name = "GRU"
    elif choice == "4":
        model_class = GnnModel
        memory_file = "memory_gnn.pt"
        arch_name = "GNN"
    elif choice == "5":
        model_class = MambaModel
        memory_file = "memory_mamba.pt"
        arch_name = "Mamba"
    else:
        model_class = TitanModel
        memory_file = "memory_titan.pt"
        arch_name = "Titan"

    model = model_class(model_name=model_name, max_capacity=max_kv_capacity)
    tui = TUI()

    if model.load_memory(memory_file):
        tui.add_chat_message("assistant", "--- Previous Session Restored ---", tracked=False)
        for msg in model.history:
            if msg["role"] != "system":
                tui.add_chat_message(msg["role"], msg["content"], tracked=True)
    else:
        tui.add_chat_message(
            "assistant",
            f"Hello! I am an intelligent agent running {model_name} via Ollama. "
            f"My memory is constrained by a simulated {max_kv_capacity}-token {arch_name} capacity.",
            tracked=False,
        )

    while True:
        tui.render(
            active_tokens=model.get_active_tokens_count(),
            max_capacity=model.max_capacity,
            dropped_tokens=model.get_dropped_tokens_count(),
            arch_name=arch_name,
            model_name=model_name,
        )
        try:
            user_input = input("\n[You] (Type 'exit' to close) > ")
        except (KeyboardInterrupt, EOFError):
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        if user_input.strip().lower() == "clear":
            model.clear_memory()
            tui.clear_chat()
            if os.path.exists(memory_file):
                os.remove(memory_file)
            tui.add_chat_message("assistant", "--- Memory Wiped ---", tracked=False)
            continue
        if user_input.strip().lower() == "help":
            help_text = (
                "\n[bold cyan]--- Available Commands ---[/bold cyan]\n"
                "[bold yellow]help[/bold yellow]   : Show this help message\n"
                "[bold yellow]memory[/bold yellow] : Display retained model structures\n"
                "[bold yellow]clear[/bold yellow]  : Wipe conversation history and memory\n"
                "[bold yellow]exit[/bold yellow]   : Close the application"
            )
            tui.add_chat_message("system", help_text, tracked=False)
            continue
        if user_input.strip().lower() == "memory":
            tui.add_chat_message("system", f"\n{model.get_memory_dump()}", tracked=False)
            continue
        if not user_input.strip():
            continue

        model.add_user_message(user_input)
        tui.add_chat_message("user", user_input)
        tui.render(
            active_tokens=model.get_active_tokens_count(),
            max_capacity=model.max_capacity,
            dropped_tokens=model.get_dropped_tokens_count(),
            arch_name=arch_name,
            model_name=model_name,
        )
        print("\n[AI] > Thinking...", end="\r")
        full_response = ""
        for chunk in model.generate_response_stream():
            full_response += chunk
            sys.stdout.write(chunk)
            sys.stdout.flush()
        model.add_assistant_message(full_response)
        tui.add_chat_message("assistant", full_response)
        model.save_memory(memory_file)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MAG Demo main entry point.")
    parser.add_argument("--legacy-menu", action="store_true", help="Run the old architecture-selection menu.")
    args, remaining = parser.parse_known_args(argv)
    if args.legacy_menu:
        return legacy_architecture_menu()

    from agents.unified_chat_interface import main as unified_main

    return unified_main(remaining)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nFatal error: {exc}")
        raise SystemExit(1)
