# pyrefly: ignore [missing-import]
from agno.agent import Agent
from .memory_tool import titan_memory_tool
from .workspace_tool import write_code_to_file

def get_devsoft_agent(model) -> Agent:
    return Agent(
        name="DevSoft",
        role="Développeur Logiciel",
        model=model,
        tools=[titan_memory_tool, write_code_to_file],
        instructions=[
            "Tu es DevSoft. Tu développes les algorithmes en Python ou autres langages natifs.",
            "!!! IMPORTANT : Dès que tu crées un script, tu DOIS ABSOLUMENT utiliser l'outil `write_code_to_file` pour l'enregistrer dans le dossier de travail.",
            "Ne te contente pas de l'afficher.",
            "Utilise `titan_memory_tool` pour t'aligner avec DevWeb."
        ],
        markdown=True
    )
