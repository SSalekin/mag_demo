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
from tools.docker_tools import create_dockerfile, create_docker_compose, run_tests_in_docker
from tools.file_tools import write_code_to_staging, list_staging_files, publish_to_workspace, clear_workspace

# Coder Agent
# Role: Write source code and associated unit tests in staging.
coder_agent = Agent(
    name="Coder",
    role="Expert Software Engineer",
    model=Ollama(id="qwen2.5:3b"),
    tools=[write_code_to_staging, list_staging_files],
    instructions=[
        "You are an expert developer.",
        "You receive technical specifications and you must generate clean and well-commented code.",
        "You must ALWAYS use the 'write_code_to_staging' tool to save your files (e.g., main.py, test_main.py, requirements.txt).",
        "Make sure to always create at least one test file to validate your code.",
        "IMPORTANT: Your main file must always contain an `if __name__ == '__main__':` block with an execution example that prints a visible result using `print()`, so the user can easily test it.",
        "IMPORTANT: You must ALWAYS generate an explanatory `README.md` file in the staging directory, detailing what the generated code does, how to install dependencies (if needed), and the exact command to run the script."
    ],
    markdown=True,
)

# DevOps Agent
# Role: Configure the Docker environment to test the generated code.
devops_agent = Agent(
    name="DevOps",
    role="Expert DevOps Engineer",
    model=Ollama(id="qwen2.5:3b"),
    tools=[create_dockerfile, create_docker_compose, list_staging_files],
    instructions=[
        "You are a DevOps engineer. Your goal is to containerize the application located in the 'staging' directory.",
        "Analyze existing files using 'list_staging_files' (e.g., is there a requirements.txt?).",
        "Use the 'create_dockerfile' tool to create an appropriate Dockerfile.",
        "Use the 'create_docker_compose' tool to create a docker-compose.yml that runs the tests.",
        "The test container must run the tests (e.g., pytest) and then exit. Do not use applications that run indefinitely (no continuous server API) because that would block the tests."
    ],
    markdown=True,
)

# Tester Agent
# Role: Run the Docker environment, perform benchmarks, and provide a review.
tester_agent = Agent(
    name="Tester",
    role="QA Engineer and Evaluator",
    model=Ollama(id="qwen2.5:3b"),
    tools=[run_tests_in_docker],
    instructions=[
        "You are a Quality Assurance engineer. Your goal is to execute tests via Docker and write a comprehensive review.",
        "Use the 'run_tests_in_docker' tool to run the test suite and retrieve the logs.",
        "Analyze the logs: does the code compile? Do the tests pass? Is the performance acceptable?",
        "Write a detailed report to the Manager with the mention [REVIEW COMPLETE]. Clearly indicate whether the code is APPROVED or REJECTED.",
        "IMPORTANT: Be extremely CONCISE in your response (under 200 words). NEVER copy the source code or long error logs, just give your verdict and a summary of the error if there is one."
    ],
    markdown=True,
)

# Manager Agent
# Role: Orchestrate everything, communicate with the user, and manage memory.
def ask_coder(prompt: str) -> str:
    """Delegates a task to the Coder agent to write code.
    Args:
        prompt (str): The specifications of the code to write.
    """
    logger.info(f"MANAGER delegates to CODER: {prompt}")
    response = coder_agent.run(prompt)
    logger.info("CODER finished its task.")
    return "Task completed by the Coder. The files have been written to the staging folder."

def ask_devops(prompt: str) -> str:
    """Delegates a task to the DevOps agent to create the Docker environment.
    Args:
        prompt (str): Instructions for DevOps (e.g., 'Create a Dockerfile for the code in staging').
    """
    logger.info(f"MANAGER delegates to DEVOPS: {prompt}")
    response = devops_agent.run(prompt)
    logger.info("DEVOPS finished its task.")
    return "Task completed by DevOps. The Docker files have been created in staging."

def ask_tester(prompt: str) -> str:
    """Delegates a task to the Tester agent to execute tests and perform a review.
    Args:
        prompt (str): Instructions for Tester (e.g., 'Run the tests and tell me if everything is fine').
    """
    logger.info(f"MANAGER delegates to TESTER: {prompt}")
    response = tester_agent.run(prompt)
    logger.info("TESTER finished its task.")
    return response.content

def estimate_and_progress(eta_minutes: int, current_step: str, percentage: int) -> str:
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

manager_agent = Agent(
    name="Manager",
    role="AI Project Manager and Orchestrator",
    model=Ollama(id="qwen2.5:3b"),
    tools=[store_fact, retrieve_fact, publish_to_workspace, clear_workspace, ask_coder, ask_devops, ask_tester, estimate_and_progress],
    instructions=[
        "You are the leader of the autonomous development team.",
        "Here is your mandatory workflow:",
        "1. BEFORE doing anything else, ask the user the following question: 'Do you want me to clear the old files from the workspace before starting? (yes/no)'.",
        "2. WAIT for the user's answer. If 'yes', use the 'clear_workspace' tool.",
        "3. Call 'estimate_and_progress' (percentage=0, current_step='PLANNING') to provide an ETA.",
        "4. (Optional) Use 'retrieve_fact' to search for context in Titan.",
        "5. Update the progress (percentage=25, current_step='DEV'). Delegate to Coder via 'ask_coder'.",
        "6. Update the progress (percentage=50, current_step='DOCKER'). Delegate to DevOps via 'ask_devops'.",
        "7. Update the progress (percentage=75, current_step='TEST'). Delegate to Tester via 'ask_tester'.",
        "8. If the review is REJECTED, update the progress (percentage=80, current_step='FIXING') and loop back to the Coder.",
        "9. If the review is APPROVED, update the progress (percentage=90, current_step='EVALUATION').",
        "10. Use the 'publish_to_workspace' tool, passing EXACTLY the list of files that are truly useful to the end user as arguments (e.g., the main python file AND the README.md, but NEVER Dockerfile or docker-compose).",
        "11. You MUST use 'store_fact' to record a summary of the accomplished work and the generated code in Titan.",
        "12. Update the progress to 100% (current_step='DONE'). Present the result to the user without emojis and clearly explain how to execute the script locally."
    ],
    markdown=True,
    add_history_to_context=True,
    num_history_messages=5,
)
