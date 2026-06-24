# pyrefly: ignore [missing-import]
from agno.agent import Agent
from .memory_tool import titan_memory_tool

def get_evaluation_agent(model) -> Agent:
    return Agent(
        name="Evaluation",
        role="Évaluateur de fin de boucle",
        model=model,
        tools=[titan_memory_tool],
        instructions=[
            "Tu es l'agent d'évaluation finale.",
            "Utilise `titan_memory_tool` (action='retrieve') pour vérifier l'historique et valider "
            "si l'équipe n'a pas souffert d'oubli catastrophique sur les exigences initiales avant "
            "de clore le projet."
        ],
        markdown=True
    )
