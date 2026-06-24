# pyrefly: ignore [missing-import]
from agno.agent import Agent
from agno.tools import tool

from .devweb import get_devweb_agent
from .devsoft import get_devsoft_agent
from .devops import get_devops_agent
from .test_qa import get_test_qa_agent
from .evaluation import get_evaluation_agent

def get_manager_agent(model) -> Agent:
    devweb = get_devweb_agent(model)
    devsoft = get_devsoft_agent(model)
    devops = get_devops_agent(model)
    test_qa = get_test_qa_agent(model)
    evaluation = get_evaluation_agent(model)

    @tool
    def delegate_to_devweb(task_description: str) -> str:
        """Envoie une tâche de développement web (HTML/CSS/JS) à l'agent DevWeb."""
        response = devweb.run(task_description)
        return response.content if hasattr(response, 'content') else str(response)

    @tool
    def delegate_to_devsoft(task_description: str) -> str:
        """Envoie une tâche de développement logiciel ou backend à l'agent DevSoft."""
        response = devsoft.run(task_description)
        return response.content if hasattr(response, 'content') else str(response)

    @tool
    def delegate_to_test_qa(code_or_instruction: str) -> str:
        """Envoie une requête de test à l'agent Test pour l'exécuter dans la Sandbox."""
        response = test_qa.run(code_or_instruction)
        return response.content if hasattr(response, 'content') else str(response)

    @tool
    def delegate_to_evaluation(results: str) -> str:
        """Envoie les résultats finaux à l'agent Evaluation pour validation."""
        response = evaluation.run(results)
        return response.content if hasattr(response, 'content') else str(response)

    return Agent(
        name="Manager",
        role="Coordonnateur",
        model=model,
        tools=[delegate_to_devweb, delegate_to_devsoft, delegate_to_test_qa, delegate_to_evaluation],
        instructions=[
            "Tu es le chef de projet de l'Usine Logicielle Multi-Agents.",
            "Ton rôle est de coordonner l'équipe (DevWeb, DevSoft, DevOps, Test, Evaluation).",
            "!!! RÈGLE CRITIQUE DE DÉLÉGATION !!! :",
            "- Tu ne dois JAMAIS écrire de code ou te contenter de lister un plan d'action de manière théorique.",
            "- Tu DOIS OBLIGATOIREMENT APPELER LES OUTILS (delegate_to_devweb, delegate_to_devsoft, etc.) pour déléguer le travail aux agents.",
            "- Délègue une seule tâche à la fois. Attends le retour de l'outil avant de passer à l'étape suivante.",
            "- Assure-toi que les développeurs utilisent `write_code_to_file` pour sauvegarder leur code.",
            "- Une fois le code généré, demande à l'outil Test de le vérifier, puis à Evaluation de le valider."
        ],
        markdown=True
    )
