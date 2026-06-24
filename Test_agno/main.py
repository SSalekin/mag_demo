import os
import sys

# Ajouter le répertoire parent au path si besoin
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# pyrefly: ignore [missing-import]
from agno.models.ollama import Ollama
from agents.manager import get_manager_agent

def main():
    print("==========================================================")
    print(" Initialisation de l'Usine Logicielle - Phase 4.5         ")
    print(" Architecture Modulaire & Titan Memory                    ")
    print("==========================================================")
    
    # Modèle partagé par toute l'équipe (Qwen 2.5 3B, excellent pour le Tool Calling)
    llm_model = Ollama(id="qwen2.5:3b")
    
    # Initialisation de l'équipe complète via le Manager
    manager_agent = get_manager_agent(llm_model)
    
    print("\n[OK] Équipe d'agents initialisée avec Ollama.")
    print("[OK] Outil de mémoire Titan encapsulé et Logs configurés.")
    print("[OK] Sandbox Docker prête dans son propre dossier.\n")
    print("Lancement de la vue interactive. Tapez 'exit' ou 'quit' pour quitter.")
    
    try:
        manager_agent.cli_app(markdown=True)
    except KeyboardInterrupt:
        print("\nArrêt du système.")

if __name__ == "__main__":
    main()
