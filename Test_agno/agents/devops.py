# pyrefly: ignore [missing-import]
from agno.agent import Agent

def get_devops_agent(model) -> Agent:
    return Agent(
        name="DevOps",
        role="Spécialiste CI/CD et Infrastructure",
        model=model,
        instructions=[
            "Tu es DevOps. Ton rôle est de préparer le déploiement ou structurer les fichiers "
            "pour que les livrables de DevWeb et DevSoft puissent être packagés."
        ],
        markdown=True
    )
