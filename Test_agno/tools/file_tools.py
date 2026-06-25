import os
import shutil
from pathlib import Path

STAGING_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "staging"
WORKSPACE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "workspace"

def write_code_to_staging(filename: str, content: str) -> str:
    """Writes a code file to the staging folder (temporary working memory).
    The Coder Agent uses this tool to create source and test files.
    
    Args:
        filename (str): The name of the file (e.g., 'main.py' or 'test_main.py').
        content (str): The source code.
    """
    try:
        file_path = STAGING_DIR / filename
        # Ensure subdirectories exist if filename contains a path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Code successfully written to staging/{filename}"
    except Exception as e:
        return f"Error writing to staging: {e}"

def list_staging_files() -> str:
    """Lists all files present in the staging folder."""
    try:
        files = []
        for root, dirs, filenames in os.walk(STAGING_DIR):
            for f in filenames:
                files.append(os.path.relpath(os.path.join(root, f), STAGING_DIR))
        if not files:
            return "The staging folder is empty."
        return "Files in staging:\n" + "\n".join(files)
    except Exception as e:
        return f"Error listing staging folder: {e}"

def clear_workspace() -> str:
    """Deletes all files present in the workspace folder.
    To be used if the user has given their permission to clean the folder.
    """
    try:
        for item in os.listdir(WORKSPACE_DIR):
            item_path = WORKSPACE_DIR / item
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        return "The workspace has been completely cleaned."
    except Exception as e:
        return f"Error cleaning workspace: {e}"

def publish_to_workspace(useful_files: list = None) -> str:
    """Moves files from the staging folder to the workspace.
    If useful_files is provided, only these files (e.g., ['main.py', 'utils.py']) will be moved.
    The staging folder is then cleaned. To be used by the Manager after final validation.
    
    Args:
        useful_files (list, optional): List of file or folder names to keep for the user.
    """
    try:
        moved_files = []
        for item in os.listdir(STAGING_DIR):
            s = STAGING_DIR / item
            d = WORKSPACE_DIR / item
            
            # Always ignore Docker technical files
            if item in ["Dockerfile", "docker-compose.yml"]:
                continue
                
            # If useful_files is specified, ignore what is not in it
            if useful_files is not None and item not in useful_files:
                continue
                
            if os.path.exists(d):
                if os.path.isdir(d):
                    shutil.rmtree(d)
                else:
                    os.remove(d)
            shutil.move(str(s), str(WORKSPACE_DIR))
            moved_files.append(item)
            
        # Completely clean staging
        for item in os.listdir(STAGING_DIR):
            item_path = STAGING_DIR / item
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
                
        return f"Success: {len(moved_files)} truly useful files ({', '.join(moved_files)}) were published to the workspace. Staging cleaned."
    except Exception as e:
        return f"Error publishing to workspace: {e}"
