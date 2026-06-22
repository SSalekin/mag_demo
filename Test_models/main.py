import sys
import os
from models.transformer_model import TransformerModel
from models.lstm_model import LstmModel
from models.gru_model import GruModel
from models.gnn_model import GnnModel
from models.mamba_model import MambaModel
from models.titan_model import TitanModel
from ui.terminal import TUI

def main():
    # Configuration
    MAX_KV_CAPACITY = 8192
    # Keep the LLM backend configurable. Default is below 2B parameters; override with OLLAMA_MODEL.
    MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    
    # Model Selection
    print("\n--- Architecture Selection ---")
    print("1. Transformer (KV-Cache)")
    print("2. LSTM (Hidden State)")
    print("3. GRU (Hidden State)")
    print("4. GNN (Graph Neural Network)")
    print("5. Mamba (External Memory)")
    print("6. Titan (External Neural Memory)")
    while True:
        choice = input("Select an architecture (1, 2, 3, 4, 5 or 6): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6']:
            break
        print("Invalid choice. Please enter 1, 2, 3, 4, 5 or 6.")
        
    if choice == '1':
        model_class = TransformerModel
        memory_file = "memory.pt"
        arch_name = "Transformer"
    elif choice == '2':
        model_class = LstmModel
        memory_file = "memory_lstm.pt"
        arch_name = "LSTM"
    elif choice == '3':
        model_class = GruModel
        memory_file = "memory_gru.pt"
        arch_name = "GRU"
    elif choice == '4':
        model_class = GnnModel
        memory_file = "memory_gnn.pt"
        arch_name = "GNN"
    elif choice == '5':
        model_class = MambaModel
        memory_file = "memory_mamba.pt"
        arch_name = "Mamba"
    else:
        model_class = TitanModel
        memory_file = "memory_titan.pt"
        arch_name = "Titan"
        
    # Initialization
    model = model_class(model_name=MODEL_NAME, max_capacity=MAX_KV_CAPACITY)
    tui = TUI()
    
    # Load memory or show welcome message
    if model.load_memory(memory_file):
        tui.add_chat_message("assistant", "--- Previous Session Restored ---", tracked=False)
        for msg in model.history:
            if msg["role"] != "system":
                tui.add_chat_message(msg["role"], msg["content"], tracked=True)
    else:
        # Welcome message
        tui.add_chat_message("assistant", f"Hello! I am an intelligent agent running {MODEL_NAME} via Ollama. "
                                          f"My memory is constrained by a simulated {MAX_KV_CAPACITY}-token {arch_name} capacity.", tracked=False)
    
    while True:
        # 1. Render UI with current memory state
        tui.render(active_tokens=model.get_active_tokens_count(), max_capacity=model.max_capacity, dropped_tokens=model.get_dropped_tokens_count(), arch_name=arch_name, model_name=MODEL_NAME)
        
        # 2. Get User Input
        try:
            user_input = input("\n[You] (Type 'exit' to close) > ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if user_input.strip().lower() in ['exit', 'quit']:
            break
            
        if user_input.strip().lower() == 'clear':
            model.clear_memory()
            tui.clear_chat()
            if os.path.exists(memory_file):
                os.remove(memory_file)
            tui.add_chat_message("assistant", "--- Memory Wiped ---", tracked=False)
            continue
            
        if user_input.strip().lower() == 'help':
            help_text = (
                "\n[bold cyan]--- Available Commands ---[/bold cyan]\n"
                "[bold yellow]help[/bold yellow]   : Show this help message\n"
                "[bold yellow]memory[/bold yellow] : Display the exact internal structures the model currently retains\n"
                "[bold yellow]clear[/bold yellow]  : Wipe the conversation history and memory\n"
                "[bold yellow]exit[/bold yellow]   : Close the application (or 'quit')"
            )
            tui.add_chat_message("system", help_text, tracked=False)
            continue
            
        if user_input.strip().lower() == 'memory':
            mem_dump = model.get_memory_dump()
            tui.add_chat_message("system", f"\n{mem_dump}", tracked=False)
            continue
            
        if not user_input.strip():
            continue
            
        # 3. Process User Input
        model.add_user_message(user_input)
        tui.add_chat_message("user", user_input)
        
        # We render again before generating response to show the user's message immediately
        tui.render(active_tokens=model.get_active_tokens_count(), max_capacity=model.max_capacity, dropped_tokens=model.get_dropped_tokens_count(), arch_name=arch_name, model_name=MODEL_NAME)
        print("\n[AI] > Thinking...", end="\r")
        
        # 4. Generate AI Response
        full_response = ""
        stream = model.generate_response_stream()
        for chunk in stream:
            full_response += chunk
            sys.stdout.write(chunk)
            sys.stdout.flush()
            
        # 5. Process AI Response
        model.add_assistant_message(full_response)
        tui.add_chat_message("assistant", full_response)
        model.save_memory(memory_file)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
