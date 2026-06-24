import os
import logging
from pathlib import Path
from agno.agent import Agent
from agno.models.ollama import Ollama

# Configuration des logs
LOG_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "agent_team.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TeamLogger")

# Import custom tools
from tools.memory_tools import store_fact, retrieve_fact
from tools.docker_tools import create_dockerfile, create_docker_compose, run_tests_in_docker
from tools.file_tools import write_code_to_staging, list_staging_files, publish_to_workspace, clear_workspace

# Coder Agent
# Rôle : Écrire le code source et les tests unitaires associés dans le staging.
coder_agent = Agent(
    name="Coder",
    role="Ingénieur Logiciel Expert",
    model=Ollama(id="qwen2.5:3b"),
    tools=[write_code_to_staging, list_staging_files],
    instructions=[
        "Tu es un développeur expert.",
        "Tu reçois des spécifications techniques et tu dois générer du code propre et bien commenté.",
        "Tu dois TOUJOURS utiliser l'outil 'write_code_to_staging' pour sauvegarder tes fichiers (ex: main.py, test_main.py, requirements.txt).",
        "Assure-toi de toujours créer au moins un fichier de test pour valider ton code.",
        "IMPORTANT: Ton fichier principal doit toujours contenir un bloc `if __name__ == '__main__':` avec un exemple d'exécution qui affiche un résultat visible avec `print()`, pour que l'utilisateur puisse le tester facilement.",
        "IMPORTANT: Tu dois TOUJOURS générer un fichier `README.md` explicatif dans le staging, qui détaille à quoi sert le code généré, comment installer les dépendances (si nécessaire) et la commande exacte pour lancer le script."
    ],
    markdown=True,
)

# DevOps Agent
# Rôle : Configurer l'environnement Docker pour tester le code généré.
devops_agent = Agent(
    name="DevOps",
    role="Ingénieur DevOps Expert",
    model=Ollama(id="qwen2.5:3b"),
    tools=[create_dockerfile, create_docker_compose, list_staging_files],
    instructions=[
        "Tu es un ingénieur DevOps. Ton but est de conteneuriser l'application présente dans le 'staging'.",
        "Analyse les fichiers existants via 'list_staging_files' (ex: y a-t-il un requirements.txt ?).",
        "Utilise l'outil 'create_dockerfile' pour créer un Dockerfile adapté.",
        "Utilise l'outil 'create_docker_compose' pour créer un docker-compose.yml qui lance les tests.",
        "Le conteneur de test doit lancer les tests (ex: pytest) et s'arrêter. N'utilise pas d'applications tournant à l'infini (pas d'API serveur qui ne s'arrête pas) car cela bloquerait les tests."
    ],
    markdown=True,
)

# Tester Agent
# Rôle : Exécuter l'environnement Docker, faire les benchmarks et donner une review.
tester_agent = Agent(
    name="Tester",
    role="Ingénieur QA et Évaluateur",
    model=Ollama(id="qwen2.5:3b"),
    tools=[run_tests_in_docker],
    instructions=[
        "Tu es un ingénieur Qualité. Ton but est d'exécuter les tests via Docker et de rédiger une review complète.",
        "Utilise l'outil 'run_tests_in_docker' pour lancer la suite de tests et récupérer les logs.",
        "Analyse les logs : le code compile-t-il ? Les tests passent-ils ? Les performances sont-elles acceptables ?",
        "Fais un rapport détaillé au Manager avec la mention [REVIEW COMPLETE]. Indique clairement si le code est VALIDÉ ou REJETÉ.",
        "IMPORTANT: Sois extrêmement CONCIS dans ta réponse (moins de 200 mots). Ne recopie JAMAIS le code source ou les longs logs d'erreur, donne juste ton verdict et le résumé de l'erreur s'il y en a une."
    ],
    markdown=True,
)

# Manager Agent
# Rôle : Orchestrer le tout, communiquer avec le user, et gérer la mémoire.
def ask_coder(prompt: str) -> str:
    """Délègue une tâche à l'agent Coder pour écrire du code.
    Args:
        prompt (str): Les spécifications du code à écrire.
    """
    logger.info(f"MANAGER délègue au CODER: {prompt}")
    response = coder_agent.run(prompt)
    logger.info("CODER a terminé sa tâche.")
    return "Tâche terminée par le Coder. Les fichiers ont été écrits dans le dossier staging."

def ask_devops(prompt: str) -> str:
    """Délègue une tâche à l'agent DevOps pour créer l'environnement Docker.
    Args:
        prompt (str): Les instructions pour le DevOps (ex: 'Crée un Dockerfile pour le code dans staging').
    """
    logger.info(f"MANAGER délègue au DEVOPS: {prompt}")
    response = devops_agent.run(prompt)
    logger.info("DEVOPS a terminé sa tâche.")
    return "Tâche terminée par le DevOps. Les fichiers Docker ont été créés dans staging."

def ask_tester(prompt: str) -> str:
    """Délègue une tâche à l'agent Tester pour exécuter les tests et faire une review.
    Args:
        prompt (str): Les instructions pour le Tester (ex: 'Lance les tests et dis-moi si tout est bon').
    """
    logger.info(f"MANAGER délègue au TESTER: {prompt}")
    response = tester_agent.run(prompt)
    logger.info("TESTER a terminé sa tâche.")
    return response.content

def estimate_and_progress(eta_minutes: int, current_step: str, percentage: int) -> str:
    """Utilise cet outil pour mettre à jour la barre de progression de l'interface utilisateur.
    Args:
        eta_minutes (int): Temps estimé en minutes.
        current_step (str): L'étape en cours (ex: 'PLANIFICATION', 'DEV', 'DOCKER', 'TEST', 'EVALUATION').
        percentage (int): Pourcentage d'avancement (0 à 100).
    """
    bar_length = 25
    filled = int(bar_length * percentage // 100)
    bar = '=' * filled + '-' * (bar_length - filled)
    
    msg = f"[{bar}] {percentage:>3}% | ETA: {eta_minutes} min | ETAPE: {current_step}"
    logger.info(msg)
    print(f"\n{msg}\n")
    return "Progression mise à jour et affichée à l'utilisateur."

manager_agent = Agent(
    name="Manager",
    role="Chef de Projet IA et Orchestrateur",
    model=Ollama(id="qwen2.5:3b"),
    tools=[store_fact, retrieve_fact, publish_to_workspace, clear_workspace, ask_coder, ask_devops, ask_tester, estimate_and_progress],
    instructions=[
        "Tu es le chef de l'équipe de développement autonome.",
        "Voici ton processus de travail obligatoire :",
        "1. AVANT toute chose, pose la question suivante à l'utilisateur : 'Voulez-vous que je supprime les anciens fichiers du workspace avant de commencer ? (oui/non)'.",
        "2. ATTENDS la réponse de l'utilisateur. Si 'oui', utilise l'outil 'clear_workspace'.",
        "3. Appelle 'estimate_and_progress' (percentage=0, current_step='PLANIFICATION') pour donner un ETA.",
        "4. (Optionnel) Utilise 'retrieve_fact' pour chercher le contexte dans Titan.",
        "5. Mets à jour la progression (percentage=25, current_step='DEV'). Délègue au Coder via 'ask_coder'.",
        "6. Mets à jour la progression (percentage=50, current_step='DOCKER'). Délègue au DevOps via 'ask_devops'.",
        "7. Mets à jour la progression (percentage=75, current_step='TEST'). Délègue au Tester via 'ask_tester'.",
        "8. Si la review est REJETÉE, mets à jour la progression (percentage=80, current_step='CORRECTION') et boucle vers le Coder.",
        "9. Si la review est VALIDÉE, mets à jour la progression (percentage=90, current_step='EVALUATION').",
        "10. Utilise l'outil 'publish_to_workspace' en passant en argument EXACTEMENT la liste des fichiers qui sont vraiment utiles pour l'utilisateur final (ex: le fichier python principal ET le README.md, mais JAMAIS Dockerfile ou docker-compose).",
        "11. Utilise OBLIGATOIREMENT 'store_fact' pour enregistrer un résumé du travail accompli et du code généré dans Titan.",
        "12. Mets à jour la progression à 100% (current_step='TERMINE'). Présente le résultat à l'utilisateur sans emojis et explique clairement comment exécuter le script localement."
    ],
    markdown=True,
    add_history_to_context=True,
    num_history_messages=5,
)
