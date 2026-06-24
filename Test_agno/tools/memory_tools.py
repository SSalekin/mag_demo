import sys
import os
from pathlib import Path

# Ajouter le dossier parent au path pour importer les modèles
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.titan_model import TitanExternalMemory
except ImportError:
    # Fallback pour un mock si le fichier n'est pas dispo
    class TitanExternalMemory:
        def __init__(self, **kwargs): pass
        def store(self, text: str): return ("Mock stored", None, [])
        def retrieve(self, query: str, k=5, min_score=0.1): return []

def store_fact(fact: str) -> str:
    """Stocke une information importante dans la mémoire à long terme Titan.
    
    Args:
        fact (str): L'information ou le fait à mémoriser (ex: 'Le port de l'API est 8080').
    """
    try:
        memory_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "titan_db"
        memory_dir.mkdir(parents=True, exist_ok=True)
        memory_path = memory_dir / "memory.pt"
        titan = TitanExternalMemory(memory_path=memory_path)
        titan.load()
        result_msg, _, _ = titan.store(fact)
        titan.save()
        return f"Succès: {result_msg}"
    except Exception as e:
        return f"Erreur lors de la sauvegarde dans Titan: {e}"

def retrieve_fact(query: str) -> str:
    """Recherche une information dans la mémoire à long terme Titan.
    
    Args:
        query (str): La question ou le mot clé à rechercher.
    """
    try:
        memory_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "titan_db"
        memory_path = memory_dir / "memory.pt"
        titan = TitanExternalMemory(memory_path=memory_path)
        titan.load()
        results = titan.retrieve(query, k=5, min_score=0.1)
        if not results:
            return "Aucune information trouvée en mémoire."
        
        # Format les résultats
        output = "Résultats trouvés en mémoire Titan :\n"
        for score, breakdown, item in results:
            output += f"- {item.text} (pertinence: {score:.2f})\n"
        return output
    except Exception as e:
        return f"Erreur lors de la lecture de Titan: {e}"
