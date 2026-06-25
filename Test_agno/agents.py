import os
import logging
from pathlib import Path
from agno.agent import Agent
from agno.models.ollama import Ollama

# Log configuration
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
from tools.docker_tools import create_dockerfile, run_tests_in_docker
from tools.file_tools import write_code_to_staging, list_staging_files, publish_to_workspace, clear_workspace

# Coder Agent
# Role: Write source code and associated unit tests in staging.
coder_agent = Agent(
    name="Coder",
    role="Expert Software Engineer",
    model=Ollama(id="qwen2.5:3b"),
    tools=[write_code_to_staging, list_staging_files],
    instructions=[
        "You are an expert Python developer.",
        "Write the requested Python script and its test file.",
        "You MUST save your code using the 'write_code_to_staging' tool.",
        "When calling 'write_code_to_staging':",
        "- 'filename' must be the name of the file (e.g., 'main.py' or 'test_main.py').",
        "- 'content' must contain ONLY the raw Python code. Do not include markdown (```) or explanations.",
        "CRITICAL RULE 1: Your script MUST end with an `if __name__ == '__main__':` block that calls print().",
        "CRITICAL RULE 2: You MUST import all necessary modules (e.g., `import random`, `import string`) at the top of your code."
    ],
    markdown=True,
    add_history_to_context=False
)

# DevOps Agent
# Role: Configure the Docker environment to test the generated code.
from tools.docker_tools import create_dockerfile
devops_agent = Agent(
    name="DevOps",
    role="Expert DevOps Engineer",
    model=Ollama(id="qwen2.5:3b"),
    tools=[create_dockerfile, list_staging_files],
    instructions=[
        "You are a DevOps engineer. Your goal is to containerize the application located in the 'staging' directory.",
        "Analyze existing files using 'list_staging_files' (e.g., is there a main.py?).",
        "Use the 'create_dockerfile' tool to create a simple Dockerfile that runs the Python script.",
        "The Dockerfile must use python:3.9-slim, copy the files, and run the main python script.",
        "DO NOT write docker-compose configurations in the Dockerfile. Just standard Dockerfile syntax."
    ],
    markdown=True,
    add_history_to_context=False
)

# Tester Agent
# Role: Validate the generated code and the Docker environment.
tester_agent = Agent(
    name="Tester",
    role="Quality Assurance Engineer",
    model=Ollama(id="qwen2.5:3b"),
    tools=[run_tests_in_docker],
    instructions=[
        "You are a QA engineer. Your role is to validate the staging environment.",
        "Use the 'run_tests_in_docker' tool to execute the Docker containers and get the execution logs.",
        "CRITICAL RULE: Check the output of the tests.",
        "If you see any Python syntax errors, exceptions, or 'not defined' errors, you MUST return REJECTED.",
        "If the tests pass, but there is no `print` output visible in the console showing the actual result, you MUST return REJECTED.",
        "If everything works and there are no errors, you MUST return APPROVED.",
        "Provide a summary of the logs, your analysis, and finish your message with exactly the word APPROVED or REJECTED."
    ],
    markdown=True,
    add_history_to_context=False
)

# Manager Agent
# Role: Orchestrate everything, communicate with the user, and manage memory.
def estimate_and_progress(eta_minutes: int = 0, current_step: str = "PROCESSING", percentage: int = 0) -> str:
    """Use this tool to update the progress bar in the user interface.
    Args:
        eta_minutes (int): Estimated time in minutes.
        current_step (str): The current step (e.g., 'PLANNING', 'DEV', 'DOCKER', 'TEST', 'EVALUATION').
        percentage (int): Progress percentage (0 to 100).
    """
    bar_length = 25
    filled = int(bar_length * percentage // 100)
    bar = '=' * filled + '-' * (bar_length - filled)
    
    msg = f"[{bar}] {percentage:>3}% | ETA: {eta_minutes} min | STEP: {current_step}"
    logger.info(msg)
    print(f"\n{msg}\n")
    return "Progress updated and displayed to the user."

def ask_coder(prompt: str) -> str:
    """Delegates a coding task to the Coder Agent."""
    estimate_and_progress(0, 'DEV', 25)
    logger.info(f"MANAGER delegates to CODER: {prompt}")
    response = coder_agent.run(prompt)
    logger.info("CODER finished its task.")
    return response.content

def ask_devops(prompt: str) -> str:
    """Delegates a deployment/Docker task to the DevOps Agent."""
    estimate_and_progress(0, 'DOCKER', 50)
    logger.info(f"MANAGER delegates to DEVOPS: {prompt}")
    response = devops_agent.run(prompt)
    logger.info("DEVOPS finished its task.")
    return response.content

def ask_tester(prompt: str) -> str:
    """Delegates a testing task to the Tester Agent."""
    estimate_and_progress(0, 'TEST', 75)
    logger.info(f"MANAGER delegates to TESTER: {prompt}")
    response = tester_agent.run(prompt)
    logger.info("TESTER finished its task.")
    return response.content

def execute_project(prompt: str) -> str:
    """Executes the entire project pipeline sequentially: Coder -> DevOps -> Tester -> Publish.
    This guarantees that the workspace is delivered to the user.
    
    Args:
        prompt (str): The user's original request.
    """
    logger.info("PIPELINE STARTED.")
    
    # 1. Coder
    ask_coder(prompt)
    
    # 2. DevOps
    ask_devops("Create a Dockerfile to run the Python script in staging.")
    
    # 3. Tester
    tester_result = ask_tester("Run the code in Docker and verify it works.")
    
    # 4. Feedback Loop (Retry once if failed)
    if "REJECTED" in tester_result:
        logger.warning("Tester rejected the code. Giving the Coder one chance to fix it...")
        ask_coder(f"The tests failed. Here is the feedback:\n{tester_result}\nPlease fix the code and write it again to staging.")
        tester_result = ask_tester("Run the fixed code in Docker and verify it works.")
    
    # 5. Publish
    if "REJECTED" in tester_result:
        logger.warning("Tester rejected the code again, publishing anyway.")
    else:
        logger.info("Code was approved by Tester.")
        
    publish_result = publish_to_workspace()
    
    from tools.file_tools import clear_workspace
    # Notice: we don't clear workspace here, it's done by publish_to_workspace filtering.
    
    logger.info("PIPELINE FINISHED.")
    return f"Project execution complete. {publish_result}"

manager_agent = Agent(
    name="Manager",
    role="AI Project Manager and Orchestrator",
    model=Ollama(id="qwen2.5:3b"),
    tools=[execute_project],
    instructions=[
        "You are the AI Project Manager.",
        "Your ONLY job is to take the user's request and pass it to the 'execute_project' tool.",
        "CRITICAL RULE: DO NOT use natural language. ONLY output the tool call.",
        "Once 'execute_project' returns, you are DONE. Stop and say nothing."
    ],
    markdown=True,
    add_history_to_context=True,
    num_history_messages=5,
)
