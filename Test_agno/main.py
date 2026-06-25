import os
from dotenv import load_dotenv

# Load environment variables (API keys, e.g., GOOGLE_API_KEY for Gemini)
load_dotenv()

from agents import manager_agent

def main():
    print("==========================================================")
    print("   AGNO CODER AGENTS TEAM WITH TITAN MEMORY             ")
    print("==========================================================")
    print("Manager: Ready! I orchestrate Coder, DevOps, and Tester.")
    print("Launching interactive view. Type 'exit' or 'quit' to exit.")
    
    try:
        manager_agent.cli_app(markdown=True)
    except KeyboardInterrupt:
        print("\nSystem shutting down.")

if __name__ == "__main__":
    main()
