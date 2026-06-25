import os
from dotenv import load_dotenv

# Load environment variables (API keys, e.g., GOOGLE_API_KEY for Gemini)
load_dotenv()

from agents import manager_agent

from tools.file_tools import WORKSPACE_DIR, clear_workspace, clear_staging

def main():
    print("==========================================================")
    print("   AGNO CODER AGENTS TEAM WITH TITAN MEMORY             ")
    print("==========================================================")
    
    # Always wipe staging clean on launch
    clear_staging()
    
    # Verify if workspace is empty
    if WORKSPACE_DIR.exists() and any(WORKSPACE_DIR.iterdir()):
        user_input = input("⚠️ The 'workspace' folder is not empty. Do you want to clear it before starting? (y/n): ").strip().lower()
        if user_input in ['y', 'yes']:
            clear_workspace()
            print("Workspace cleared successfully.\n")
        else:
            print("Keeping old files in the workspace.\n")

    print("Manager: Ready! I orchestrate Coder, DevOps, and Tester.")
    print("Launching interactive view. Type 'exit' or 'quit' to exit.")
    
    try:
        manager_agent.cli_app(markdown=True, emoji="")
    except KeyboardInterrupt:
        print("\nSystem shutting down.")

if __name__ == "__main__":
    main()
