import os
import shutil
from pathlib import Path

STAGING_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "staging"
WORKSPACE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "workspace"

def write_code_to_staging(filename: str, content: str) -> str:
    """Écrit un fichier de code dans le dossier de staging (mémoire de travail temporaire).
    Le Coder Agent utilise cet outil pour créer les fichiers sources et de tests.
    
    Args:
        filename (str): Le nom du fichier (ex: 'main.py' ou 'test_main.py').
        content (str): Le code source.
    """
    try:
        file_path = STAGING_DIR / filename
        # Ensure subdirectories exist if filename contains a path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Code écrit avec succès dans staging/{filename}"
    except Exception as e:
        return f"Erreur lors de l'écriture dans staging : {e}"

def list_staging_files() -> str:
    """Liste tous les fichiers présents dans le dossier de staging."""
    try:
        files = []
        for root, dirs, filenames in os.walk(STAGING_DIR):
            for f in filenames:
                files.append(os.path.relpath(os.path.join(root, f), STAGING_DIR))
        if not files:
            return "Le dossier de staging est vide."
        return "Fichiers dans staging:\n" + "\n".join(files)
    except Exception as e:
        return f"Erreur lors du listage de staging : {e}"

def clear_workspace() -> str:
    """Supprime tous les fichiers présents dans le dossier workspace.
    À utiliser si l'utilisateur a donné son accord pour nettoyer le dossier.
    """
    try:
        for item in os.listdir(WORKSPACE_DIR):
            item_path = WORKSPACE_DIR / item
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        return "Le workspace a été entièrement nettoyé."
    except Exception as e:
        return f"Erreur lors du nettoyage du workspace : {e}"

def publish_to_workspace(useful_files: list = None) -> str:
    """Déplace les fichiers du dossier de staging vers le workspace.
    Si useful_files est fourni, seuls ces fichiers (ex: ['main.py', 'utils.py']) seront déplacés.
    Le dossier de staging est ensuite nettoyé. À utiliser par le Manager après validation finale.
    
    Args:
        useful_files (list, optional): Liste des noms de fichiers ou dossiers à conserver pour l'utilisateur.
    """
    try:
        moved_files = []
        for item in os.listdir(STAGING_DIR):
            s = STAGING_DIR / item
            d = WORKSPACE_DIR / item
            
            # On ignore toujours les fichiers techniques Docker
            if item in ["Dockerfile", "docker-compose.yml"]:
                continue
                
            # Si on a spécifié useful_files, on ignore ce qui n'est pas dedans
            if useful_files is not None and item not in useful_files:
                continue
                
            if os.path.exists(d):
                if os.path.isdir(d):
                    shutil.rmtree(d)
                else:
                    os.remove(d)
            shutil.move(str(s), str(WORKSPACE_DIR))
            moved_files.append(item)
            
        # Nettoyer complètement le staging
        for item in os.listdir(STAGING_DIR):
            item_path = STAGING_DIR / item
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
                
        return f"Succès: {len(moved_files)} fichiers vraiment utiles ({', '.join(moved_files)}) ont été publiés dans le workspace. Staging nettoyé."
    except Exception as e:
        return f"Erreur lors de la publication vers le workspace : {e}"
