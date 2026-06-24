import os
import sys
import time
import csv
import shutil
from pathlib import Path

# Ajouter le répertoire parent au sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.file_tools import clear_workspace
from agents import coder_agent, devops_agent, tester_agent, manager_agent

STAGING_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "staging"
WORKSPACE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "workspace"

LOREM_IPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. "

def generate_prompt(base_prompt: str, size: str) -> str:
    if size == "Small":
        return base_prompt
    elif size == "Medium":
        padding = LOREM_IPSUM * 5 # ~350 mots
        return f"{base_prompt}\n\nContexte supplémentaire sans importance, merci de l'ignorer totalement :\n{padding}"
    elif size == "Large":
        padding = LOREM_IPSUM * 25 # ~1750 mots
        return f"{base_prompt}\n\nContexte supplémentaire très long et sans importance, merci de l'ignorer totalement :\n{padding}"
    return base_prompt

def evaluate_accuracy(agent_name: str, response: str) -> bool:
    """Évalue si l'agent a réussi sa tâche (Accuracy = 0 ou 100%)."""
    if agent_name == "Coder":
        if STAGING_DIR.exists() and any(f.endswith(".py") for f in os.listdir(STAGING_DIR)):
            return True
        return False
    elif agent_name == "DevOps":
        if STAGING_DIR.exists() and "Dockerfile" in os.listdir(STAGING_DIR):
            return True
        return False
    elif agent_name == "Tester":
        resp_lower = str(response).lower()
        if "valid" in resp_lower or "rejet" in resp_lower or "review" in resp_lower or "succès" in resp_lower:
            return True
        return False
    elif agent_name == "Manager":
        resp_lower = str(response).lower()
        if "délég" in resp_lower or "coder" in resp_lower or "base de données" in resp_lower:
            return True
        return False
    return False

def run_agent_benchmark():
    print("🚀 Lancement du Benchmark de Performance des LLM par Agent 🚀\n")
    print("Ce test utilise Ollama. Il prendra du temps en fonction de votre GPU/CPU.\n")
    
    tasks = [
        {"agent": coder_agent, "name": "Coder", "base_prompt": "Écris une fonction Python simple qui additionne deux nombres et sauvegarde-la dans 'math_utils.py' avec ton outil."},
        {"agent": devops_agent, "name": "DevOps", "base_prompt": "Crée un fichier Dockerfile très simple pour une application Python et sauvegarde-le dans le staging."},
        {"agent": tester_agent, "name": "Tester", "base_prompt": "Fais un rapport de test fictif indiquant que tout est VALIDÉ sans utiliser d'outils."},
        {"agent": manager_agent, "name": "Manager", "base_prompt": "Analyse la demande suivante et dis-moi à quel agent tu vas la déléguer : 'Je veux un script de base de données'."}
    ]
    
    sizes = ["Small", "Medium", "Large"]
    results = []
    
    print(f"{'Agent':<10} | {'Taille':<10} | {'Mots (env.)':<12} | {'Temps (s)':<10} | {'Accuracy'}")
    print("-" * 65)
    
    for task in tasks:
        agent = task["agent"]
        agent_name = task["name"]
        
        for size in sizes:
            prompt = generate_prompt(task["base_prompt"], size)
            approx_words = len(prompt.split())
            
            # Nettoyage avant chaque test
            if STAGING_DIR.exists():
                shutil.rmtree(STAGING_DIR)
            STAGING_DIR.mkdir()
                
            start_time = time.time()
            try:
                # Masquer stdout et stderr pendant l'exécution pour cacher les WARNINGS Agno et les barres de progression
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                try:
                    with open(os.devnull, 'w') as devnull:
                        sys.stdout = devnull
                        sys.stderr = devnull
                        response = agent.run(prompt)
                finally:
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                    
                resp_content = response.content if hasattr(response, 'content') else str(response)
                
                duration = time.time() - start_time
                success = evaluate_accuracy(agent_name, resp_content)
                
            except Exception as e:
                duration = time.time() - start_time
                success = False
                resp_content = f"ERROR: {e}"
                
            if hasattr(agent, 'memory') and hasattr(agent.memory, 'messages'):
                try: agent.memory.messages.clear()
                except: pass
            if hasattr(agent, 'session_state') and agent.session_state is not None:
                try: agent.session_state.clear()
                except: pass
                
            acc_str = "✅ 100%" if success else "❌ 0%"
            print(f"{agent_name:<10} | {size:<10} | {approx_words:<12} | {duration:<10.2f} | {acc_str}")
            
            results.append({
                "Agent": agent_name,
                "Taille_Prompt": size,
                "Nombre_Mots": approx_words,
                "Temps_secondes": round(duration, 2),
                "Accuracy": 100 if success else 0
            })
            
    print("-" * 65)
    
    # Calcul des statistiques globales
    total_tests = len(results)
    total_successes = sum(1 for r in results if r["Accuracy"] == 100)
    avg_duration = sum(r["Temps_secondes"] for r in results) / total_tests if total_tests > 0 else 0
    global_accuracy = (total_successes / total_tests) * 100 if total_tests > 0 else 0
    
    print("\n=== Résumé Global ===")
    print(f"Tests effectués : {total_tests}")
    print(f"Temps de réponse moyen : {avg_duration:.2f} s")
    print(f"Accuracy Globale : {global_accuracy:.1f}% ({total_successes}/{total_tests})")
            
    out_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    csv_path = out_dir / "benchmark_llm_results.csv"
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Agent", "Taille_Prompt", "Nombre_Mots", "Temps_secondes", "Accuracy"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n📊 Rapport complet exporté dans : {csv_path}")

if __name__ == "__main__":
    run_agent_benchmark()
