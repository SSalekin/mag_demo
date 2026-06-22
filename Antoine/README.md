# Conversational Agent with Simulated Transformer KV-Cache

This project is a local research and experimentation framework for an AI conversational agent. It simulates an external Transformer Key-Value (KV) Cache memory using a sliding window approach, ensuring deep insight into token usage and memory constraints. The system leverages the local `ollama` ecosystem for privacy-focused, highly intelligent AI inference.

## Features

- **Local AI Inference**: Powered by local Large Language Models (LLMs) via Ollama (defaults to `llama3`). 100% free and private.
- **Unified Transformer Model**: The model seamlessly combines the intelligent inference of Ollama with the exact memory constraints (Token embedding simulation, KV-Cache sliding window) locally.
- **Dynamic Terminal UI**: A modern Terminal User Interface (TUI) built with `rich`, featuring:
  - A real-time **System Dashboard** showing active token usage out of the Transformer's max capacity.
  - A distinct, color-coded **Chat Interface**.

## Project Structure

```text
Antoine/
├── main.py          # The main entry point linking the TUI and the Unified Transformer Model.
├── requirements.txt # Project dependencies (ollama, rich, tiktoken, numpy, matplotlib, etc.).
├── README.md        # This file.
├── benchmark/       # Benchmarking framework for metrics and visualization
│   ├── dataset.json # Evaluation prompts
│   ├── evaluator.py # Runs tests against models and logs to CSV
│   └── plotter.py   # Generates Scatter Plots and Radar Charts
├── models/
│   ├── __init__.py
│   └── transformer_model.py # Unified class handling both Ollama connection and KV-Cache simulation.
└── ui/
    ├── __init__.py
    └── terminal.py    # The interface module utilizing `rich` for terminal display.
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

1. Open your terminal and navigate to the `Antoine` directory.
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

Interact with the AI through the chat panel and monitor the simulated token consumption on the dashboard panel above it. To exit the session, simply type `exit` or `quit`.
