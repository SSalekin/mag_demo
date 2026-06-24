import os
from dotenv import load_dotenv

# Charger les variables d'environnement (API keys, ex: GOOGLE_API_KEY pour Gemini)
load_dotenv()

from agents import manager_agent

def main():
    print("==========================================================")
    print(" 🚀 ÉQUIPE D'AGENTS CODEURS AGNO AVEC MÉMOIRE TITAN 🚀    ")
    print("==========================================================")
    print("Manager: Prêt ! J'orchestre Coder, DevOps et Tester.")
    print("Lancement de la vue interactive. Tapez 'exit' ou 'quit' pour quitter.")
    
    try:
        manager_agent.cli_app(markdown=True)
    except KeyboardInterrupt:
        print("\nArrêt du système.")

if __name__ == "__main__":
    main()
