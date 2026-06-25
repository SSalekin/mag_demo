import os
import subprocess
from pathlib import Path

STAGING_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "staging"
WORKSPACE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "workspace"

def create_dockerfile(content: str) -> str:
    """Creates a Dockerfile in the staging folder for the test environment.
    
    Args:
        content (str): The full content of the Dockerfile.
    """
    try:
        with open(STAGING_DIR / "Dockerfile", "w", encoding="utf-8") as f:
            f.write(content)
        return "Dockerfile successfully created in staging."
    except Exception as e:
        return f"Error creating Dockerfile: {e}"

def create_docker_compose(content: str) -> str:
    """Creates a docker-compose.yml in the staging folder.
    
    Args:
        content (str): The full content of the docker-compose.yml file.
    """
    try:
        with open(STAGING_DIR / "docker-compose.yml", "w", encoding="utf-8") as f:
            f.write(content)
        return "docker-compose.yml successfully created in staging."
    except Exception as e:
        return f"Error creating docker-compose: {e}"

def run_tests_in_docker() -> str:
    """Builds and runs Docker containers in staging to launch tests.
    Returns the execution logs so the tester can evaluate the code.
    
    Returns:
        str: The full execution logs (stdout and stderr).
    """
    try:
        # Run docker compose up --build --abort-on-container-exit
        result = subprocess.run(
            ["docker", "compose", "up", "--build", "--abort-on-container-exit"],
            cwd=str(STAGING_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300 # 5 minutes max
        )
        
        # Truncate to avoid overloading the LLM context (max 4000 characters at the end)
        stdout_str = result.stdout[-4000:] if result.stdout and len(result.stdout) > 4000 else result.stdout
        stderr_str = result.stderr[-4000:] if result.stderr and len(result.stderr) > 4000 else result.stderr

        output = f"Return code: {result.returncode}\n\n"
        output += f"--- STDOUT (truncated) ---\n{stdout_str}\n"
        if stderr_str:
            output += f"--- STDERR (truncated) ---\n{stderr_str}\n"
            
        # Cleanup
        subprocess.run(["docker", "compose", "down"], cwd=str(STAGING_DIR), capture_output=True)
        
        return output
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "compose", "down"], cwd=str(STAGING_DIR), capture_output=True)
        return "Error: Execution exceeded the 5-minute time limit."
    except Exception as e:
        return f"Docker execution error: {e}"
