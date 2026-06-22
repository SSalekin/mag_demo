# Conversational Agent with Simulated Transformer KV-Cache (mag_demo)

This repository contains the code for a local research and experimentation framework for an AI conversational agent.

## Repository Structure

- **`Antoine/`**: Contains the active, modern version of the project. It features a Terminal UI (TUI) built with `rich`, local inference via `ollama`, and a benchmarking suite. 
  - *Please navigate to the `Antoine/` directory for the most up-to-date code and detailed documentation.*
- **`Legacy/`**: Contains older iterations and standalone implementations (e.g., `titan_implementation.py`, `transformer_implementation.py`, etc.).

## Quick Start (Active Project)

To run the current version of the conversational agent:

1. **Prerequisites**: Ensure you have [Ollama](https://ollama.com/) installed and running.
2. **Set up Virtual Environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Agent**:
   ```bash
   cd Antoine
   python main.py
   ```

For more detailed instructions, metrics, and architecture details, please refer to the [`Antoine/README.md`](Antoine/README.md).