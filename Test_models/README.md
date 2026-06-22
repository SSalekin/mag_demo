# Conversational Agent with Simulated Memory Architectures

This project is a local research and experimentation framework for an AI conversational agent. It simulates multiple memory architectures (Transformer KV-Cache, LSTM, GRU, GNN, Mamba, and Titan) to provide deep insight into token usage and memory constraints across different paradigms. The system leverages the local `ollama` ecosystem for privacy-focused, highly intelligent AI inference.

## Features

- **Local AI Inference**: Powered by local Large Language Models (LLMs) via Ollama (defaults to `llama3`). 100% free and private.
- **Unified Memory Models**: The system supports 6 interchangeable memory architectures seamlessly combined with Ollama inference:
  1. **Transformer** (KV-Cache Sliding Window)
  2. **LSTM** (Hidden State)
  3. **GRU** (Hidden State)
  4. **GNN** (Graph Neural Network)
  5. **Mamba** (External Memory)
  6. **Titan** (External Neural Memory)
- **Dynamic Terminal UI**: A modern Terminal User Interface (TUI) built with `rich`, featuring:
  - A real-time **System Dashboard** showing active token usage and dropped memories out of the max capacity.
  - A distinct, color-coded **Chat Interface**.
- **Comprehensive Benchmarking**: Detailed benchmarking tools, including stress tests with hard distractors and memory updates, to evaluate memory retention performance.

## Project Structure

```text
Test_models/
├── main.py          # The main entry point linking the TUI and the memory models.
├── requirements.txt # Project dependencies (ollama, rich, torch, transformers, etc.).
├── README.md        # This file.
├── benchmark/       # Benchmarking framework
│   ├── compare_memory_models.py         # Standard comparison test for all architectures
│   └── compare_memory_models_stress.py  # Hard stress benchmark (distractors, forgetting)
├── models/          # Implementations of the different memory architectures
│   ├── __init__.py
│   ├── transformer_model.py
│   ├── lstm_model.py
│   ├── gru_model.py
│   ├── gnn_model.py
│   ├── mamba_model.py
│   └── titan_model.py
└── ui/
    ├── __init__.py
    └── terminal.py  # The interface module utilizing `rich` for terminal display.
```

## Prerequisites

1. **Ollama**: You must have Ollama installed and running on your system.
   - Download it from [ollama.com](https://ollama.com/).
2. **Local Model**: Pull the default model (or change it in `main.py`).
   ```bash
   ollama pull llama3
   ```
3. **Python 3.8+**: Ensure you have a recent version of Python installed.

## Installation & Setup

1. Open your terminal and navigate to the `Test_models` directory.
2. It is highly recommended to create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Start the agent by running the main entry script:

```bash
python main.py
```

Upon launch, you will be prompted to select the memory architecture you wish to use (1 to 6).

Interact with the AI through the chat panel and monitor the simulated token consumption on the dashboard panel above it. To exit the session, simply type `exit` or `quit`.

## Benchmarking

To test and compare the memory retention across all architectures, you can use the benchmark scripts:

```bash
# Run the standard benchmark
python benchmark/compare_memory_models.py

# Run the advanced stress benchmark
python benchmark/compare_memory_models_stress.py
```
