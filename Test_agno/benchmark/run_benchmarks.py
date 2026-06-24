import os
import sys
import time
import unittest
import shutil
from pathlib import Path

# Ajouter le répertoire parent au sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.file_tools import write_code_to_staging, list_staging_files, publish_to_workspace, clear_workspace
from tools.docker_tools import create_dockerfile, create_docker_compose, run_tests_in_docker
from tools.memory_tools import store_fact, retrieve_fact

WORKSPACE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "workspace"
STAGING_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "staging"

class TestArchitectureRobustness(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """S'exécute une fois avant tous les tests."""
        print("\n🚀 Démarrage de la Suite de Tests d'Architecture (Sans LLM) 🚀")
        clear_workspace()
        if STAGING_DIR.exists():
            shutil.rmtree(STAGING_DIR)
        STAGING_DIR.mkdir(exist_ok=True)

    def test_1_file_management(self):
        """Teste la capacité à écrire, lister et isoler des fichiers dans staging."""
        print("\n[Test] File Management...")
        
        # Write files
        res1 = write_code_to_staging("app.py", "print('hello')")
        self.assertIn("Succès", res1 or "Code écrit avec succès")
        
        res2 = write_code_to_staging("nested/utils.py", "def add(a, b): return a + b")
        self.assertIn("Succès", res2 or "Code écrit avec succès")
        
        # List files
        files_list = list_staging_files()
        self.assertIn("app.py", files_list)
        self.assertIn("utils.py", files_list)

    def test_2_docker_pipeline_real_execution(self):
        """Teste la capacité réelle de Docker à conteneuriser et exécuter les tests."""
        print("\n[Test] Docker Pipeline Execution...")
        
        # On crée un vrai test unitaire Python pour que Docker le lance
        app_code = "def get_password(): return 'supersecret123'"
        test_code = (
            "import pytest\n"
            "from main import get_password\n"
            "def test_pwd():\n"
            "    assert get_password() == 'supersecret123'\n"
        )
        write_code_to_staging("main.py", app_code)
        write_code_to_staging("test_main.py", test_code)
        write_code_to_staging("requirements.txt", "pytest==7.4.2\n")
        
        dockerfile = (
            "FROM python:3.10-slim\n"
            "WORKDIR /app\n"
            "COPY . .\n"
            "RUN pip install -r requirements.txt\n"
            "CMD ['pytest', 'test_main.py']\n"
        )
        create_dockerfile(dockerfile)
        
        compose = (
            "version: '3'\n"
            "services:\n"
            "  test_app:\n"
            "    build: .\n"
        )
        create_docker_compose(compose)
        
        # On lance vraiment Docker ! (Ce test prendra quelques secondes)
        start_time = time.time()
        docker_logs = run_tests_in_docker()
        duration = time.time() - start_time
        
        print(f"  -> Docker a compilé et exécuté les tests en {duration:.1f}s")
        # On vérifie que le test pytest est passé ("1 passed" ou retour 0)
        self.assertIn("Code de retour : 0", docker_logs)

    def test_3_workspace_publishing_and_security(self):
        """Teste que les fichiers publiés respectent le filtre de sécurité."""
        print("\n[Test] Workspace Publishing & Security...")
        
        # S'assure que staging a des fichiers (créés par le test Docker)
        self.assertTrue(STAGING_DIR.exists())
        
        # On publie uniquement main.py et test_main.py
        result = publish_to_workspace(useful_files=["main.py", "test_main.py"])
        self.assertIn("Succès", result)
        
        published = os.listdir(WORKSPACE_DIR)
        
        # Vérifications de sécurité
        self.assertIn("main.py", published)
        self.assertNotIn("Dockerfile", published, "Faille: Le Dockerfile a été publié !")
        self.assertNotIn("docker-compose.yml", published, "Faille: docker-compose a été publié !")
        
        # Le dossier staging doit être vide après publication
        self.assertEqual(len(os.listdir(STAGING_DIR)), 0, "Le staging n'a pas été nettoyé.")

    def test_4_titan_memory_engine(self):
        """Teste le fonctionnement de la base de données Titan."""
        print("\n[Test] Titan Neural Memory...")
        
        fact_text = "Le système de test fonctionne à 100% de ses capacités."
        store_result = store_fact(fact_text)
        self.assertIn("succès", store_result.lower())
        
        retrieve_result = retrieve_fact("test")
        self.assertIn(fact_text, retrieve_result)

if __name__ == '__main__':
    # Lance la suite de tests de manière verbeuse
    unittest.main(verbosity=2)
