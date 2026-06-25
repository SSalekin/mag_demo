import os
import sys
import time
import unittest
import shutil
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.file_tools import write_code_to_staging, list_staging_files, publish_to_workspace, clear_workspace
from tools.docker_tools import create_dockerfile, create_docker_compose, run_tests_in_docker
from tools.memory_tools import store_fact, retrieve_fact

WORKSPACE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "workspace"
STAGING_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "staging"

class TestArchitectureRobustness(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Runs once before all tests."""
        print("\n🚀 Starting Architecture Test Suite (No LLM) 🚀")
        clear_workspace()
        if STAGING_DIR.exists():
            shutil.rmtree(STAGING_DIR)
        STAGING_DIR.mkdir(exist_ok=True)

    def test_1_file_management(self):
        """Tests the ability to write, list, and isolate files in staging."""
        print("\n[Test] File Management...")
        
        # Write files
        res1 = write_code_to_staging("app.py", "print('hello')")
        self.assertIn("Success", res1 or "Code successfully written")
        
        res2 = write_code_to_staging("nested/utils.py", "def add(a, b): return a + b")
        self.assertIn("Success", res2 or "Code successfully written")
        
        # List files
        files_list = list_staging_files()
        self.assertIn("app.py", files_list)
        self.assertIn("utils.py", files_list)

    def test_2_docker_pipeline_real_execution(self):
        """Tests the actual ability of Docker to containerize and execute tests."""
        print("\n[Test] Docker Pipeline Execution...")
        
        # Create a real Python unit test for Docker to run
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
        
        # Actually run Docker! (This test will take a few seconds)
        start_time = time.time()
        docker_logs = run_tests_in_docker()
        duration = time.time() - start_time
        
        print(f"  -> Docker compiled and ran the tests in {duration:.1f}s")
        # Check that the pytest test passed ("1 passed" or return code 0)
        self.assertIn("Return code: 0", docker_logs)

    def test_3_workspace_publishing_and_security(self):
        """Tests that published files respect the security filter."""
        print("\n[Test] Workspace Publishing & Security...")
        
        # Ensure staging has files (created by the Docker test)
        self.assertTrue(STAGING_DIR.exists())
        
        # Publish only main.py and test_main.py
        result = publish_to_workspace(useful_files=["main.py", "test_main.py"])
        self.assertIn("Success", result)
        
        published = os.listdir(WORKSPACE_DIR)
        
        # Security checks
        self.assertIn("main.py", published)
        self.assertNotIn("Dockerfile", published, "Vulnerability: Dockerfile was published!")
        self.assertNotIn("docker-compose.yml", published, "Vulnerability: docker-compose was published!")
        
        # The staging folder must be empty after publishing
        self.assertEqual(len(os.listdir(STAGING_DIR)), 0, "Staging was not cleaned.")

    def test_4_titan_memory_engine(self):
        """Tests the functionality of the Titan database."""
        print("\n[Test] Titan Neural Memory...")
        
        fact_text = "The testing system is operating at 100% capacity."
        store_result = store_fact(fact_text)
        self.assertIn("success", store_result.lower())
        
        retrieve_result = retrieve_fact("test")
        self.assertIn(fact_text, retrieve_result)

if __name__ == '__main__':
    # Run the test suite verbosely
    unittest.main(verbosity=2)
