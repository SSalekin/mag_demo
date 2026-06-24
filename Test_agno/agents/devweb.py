# pyrefly: ignore [missing-import]
from agno.agent import Agent
from .memory_tool import titan_memory_tool
from .workspace_tool import write_code_to_file

def get_devweb_agent(model) -> Agent:
    return Agent(
        name="DevWeb",
        role="Développeur Web",
        model=model,
        tools=[titan_memory_tool, write_code_to_file],
        instructions=[
            "Tu es DevWeb, un développeur web.",
            "!!! IMPORTANT : Dès que tu génères du code HTML, CSS, JS ou PHP, tu DOIS utiliser l'outil `write_code_to_file` pour l'enregistrer dans un fichier physique.",
            "Ne te contente jamais de juste afficher le code dans le terminal.",
            "Utilise `titan_memory_tool` (avec action='store' ou 'retrieve') pour partager ton architecture avec l'équipe."
        ],
        markdown=True
    )
