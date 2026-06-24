import os
import subprocess
from pathlib import Path

STAGING_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "staging"
WORKSPACE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "workspace"

def create_dockerfile(content: str) -> str:
    """Crée un Dockerfile dans le dossier de staging pour l'environnement de test.
    
    Args:
        content (str): Le contenu complet du Dockerfile.
    """
    try:
        with open(STAGING_DIR / "Dockerfile", "w", encoding="utf-8") as f:
            f.write(content)
        return "Dockerfile créé avec succès dans staging."
    except Exception as e:
        return f"Erreur lors de la création du Dockerfile : {e}"

def create_docker_compose(content: str) -> str:
    """Crée un docker-compose.yml dans le dossier de staging.
    
    Args:
        content (str): Le contenu complet du fichier docker-compose.yml.
    """
    try:
        with open(STAGING_DIR / "docker-compose.yml", "w", encoding="utf-8") as f:
            f.write(content)
        return "docker-compose.yml créé avec succès dans staging."
    except Exception as e:
        return f"Erreur lors de la création de docker-compose : {e}"

def run_tests_in_docker() -> str:
    """Construit et exécute les conteneurs Docker dans le staging pour lancer les tests.
    Retourne les logs de l'exécution pour que le testeur puisse évaluer le code.
    
    Returns:
        str: Les logs complets d'exécution (stdout et stderr).
    """
    try:
        # Exécute docker compose up --build --abort-on-container-exit
        result = subprocess.run(
            ["docker", "compose", "up", "--build", "--abort-on-container-exit"],
            cwd=str(STAGING_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300 # 5 minutes max
        )
        
        # Troncature pour éviter de saturer le contexte du LLM (max 4000 caractères à la fin)
        stdout_str = result.stdout[-4000:] if result.stdout and len(result.stdout) > 4000 else result.stdout
        stderr_str = result.stderr[-4000:] if result.stderr and len(result.stderr) > 4000 else result.stderr

        output = f"Code de retour : {result.returncode}\n\n"
        output += f"--- STDOUT (tronqué) ---\n{stdout_str}\n"
        if stderr_str:
            output += f"--- STDERR (tronqué) ---\n{stderr_str}\n"
            
        # Nettoyage
        subprocess.run(["docker", "compose", "down"], cwd=str(STAGING_DIR), capture_output=True)
        
        return output
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "compose", "down"], cwd=str(STAGING_DIR), capture_output=True)
        return "Erreur: L'exécution a dépassé le temps limite de 5 minutes."
    except Exception as e:
        return f"Erreur d'exécution Docker : {e}"
