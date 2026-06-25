import os
import sys
import time
import csv
import shutil
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.file_tools import clear_workspace
from agents import coder_agent, devops_agent, tester_agent, manager_agent

STAGING_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "staging"
WORKSPACE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "workspace"

LOREM_IPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. "

def generate_prompt(base_prompt: str, size: str) -> str:
    if size == "Small":
        return base_prompt
    elif size == "Medium":
        padding = LOREM_IPSUM * 5 # ~350 words
        return f"{base_prompt}\n\nAdditional unimportant context, please ignore completely:\n{padding}"
    elif size == "Large":
        padding = LOREM_IPSUM * 25 # ~1750 words
        return f"{base_prompt}\n\nVery long and unimportant additional context, please ignore completely:\n{padding}"
    return base_prompt

def evaluate_accuracy(agent_name: str, response: str) -> bool:
    """Evaluates if the agent successfully completed its task (Accuracy = 0 or 100%)."""
    if agent_name == "Coder":
        if STAGING_DIR.exists() and any(f.endswith(".py") for f in os.listdir(STAGING_DIR)):
            return True
        return False
    elif agent_name == "DevOps":
        if STAGING_DIR.exists() and "Dockerfile" in os.listdir(STAGING_DIR):
            return True
        return False
    elif agent_name == "Tester":
        resp_lower = str(response).lower()
        if "valid" in resp_lower or "reject" in resp_lower or "review" in resp_lower or "success" in resp_lower:
            return True
        return False
    elif agent_name == "Manager":
        resp_lower = str(response).lower()
        if "deleg" in resp_lower or "coder" in resp_lower or "database" in resp_lower:
            return True
        return False
    return False

def run_agent_benchmark():
    print("🚀 Starting Agent LLM Performance Benchmark 🚀\n")
    print("This test uses Ollama. It will take time depending on your GPU/CPU.\n")
    
    tasks = [
        {"agent": coder_agent, "name": "Coder", "base_prompt": "Write a simple Python function that adds two numbers and save it to 'math_utils.py' using your tool."},
        {"agent": devops_agent, "name": "DevOps", "base_prompt": "Create a very simple Dockerfile for a Python application and save it in staging."},
        {"agent": tester_agent, "name": "Tester", "base_prompt": "Create a fictitious test report stating that everything is APPROVED without using any tools."},
        {"agent": manager_agent, "name": "Manager", "base_prompt": "Analyze the following request and tell me which agent you will delegate it to: 'I want a database script'."}
    ]
    
    sizes = ["Small", "Medium", "Large"]
    results = []
    
    print(f"{'Agent':<10} | {'Size':<10} | {'Words (approx)':<15} | {'Time (s)':<10} | {'Accuracy'}")
    print("-" * 70)
    
    for task in tasks:
        agent = task["agent"]
        agent_name = task["name"]
        
        for size in sizes:
            prompt = generate_prompt(task["base_prompt"], size)
            approx_words = len(prompt.split())
            
            # Cleanup before each test
            if STAGING_DIR.exists():
                shutil.rmtree(STAGING_DIR)
            STAGING_DIR.mkdir()
                
            start_time = time.time()
            try:
                # Hide stdout and stderr during execution to hide Agno WARNINGS and progress bars
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                try:
                    with open(os.devnull, 'w') as devnull:
                        sys.stdout = devnull
                        sys.stderr = devnull
                        response = agent.run(prompt)
                finally:
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                    
                resp_content = response.content if hasattr(response, 'content') else str(response)
                
                duration = time.time() - start_time
                success = evaluate_accuracy(agent_name, resp_content)
                
            except Exception as e:
                duration = time.time() - start_time
                success = False
                resp_content = f"ERROR: {e}"
                
            if hasattr(agent, 'memory') and hasattr(agent.memory, 'messages'):
                try: agent.memory.messages.clear()
                except: pass
            if hasattr(agent, 'session_state') and agent.session_state is not None:
                try: agent.session_state.clear()
                except: pass
                
            acc_str = "✅ 100%" if success else "❌ 0%"
            print(f"{agent_name:<10} | {size:<10} | {approx_words:<15} | {duration:<10.2f} | {acc_str}")
            
            results.append({
                "Agent": agent_name,
                "Prompt_Size": size,
                "Word_Count": approx_words,
                "Time_seconds": round(duration, 2),
                "Accuracy": 100 if success else 0
            })
            
    print("-" * 70)
    
    # Calculate global statistics
    total_tests = len(results)
    total_successes = sum(1 for r in results if r["Accuracy"] == 100)
    avg_duration = sum(r["Time_seconds"] for r in results) / total_tests if total_tests > 0 else 0
    global_accuracy = (total_successes / total_tests) * 100 if total_tests > 0 else 0
    
    print("\n=== Global Summary ===")
    print(f"Tests executed: {total_tests}")
    print(f"Average response time: {avg_duration:.2f} s")
    print(f"Global Accuracy: {global_accuracy:.1f}% ({total_successes}/{total_tests})")
            
    out_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    csv_path = out_dir / "benchmark_llm_results.csv"
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Agent", "Prompt_Size", "Word_Count", "Time_seconds", "Accuracy"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n📊 Full report exported to: {csv_path}")

if __name__ == "__main__":
    run_agent_benchmark()
