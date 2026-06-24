# MAG Demo (Multi-Agent System)

Ce dépôt contient le code source de l'architecture du système multi-agents (MAG) - **Usine Logicielle**.

## Architecture & Composants (Phase 5)
L'usine logicielle est désormais totalement modulaire et fonctionnelle. Elle est pilotée par **Ollama** via le framework **Agno** et se compose des éléments suivants :

### L'Équipe Multi-Agents
- **Manager** : Superviseur du processus global. Il coordonne l'équipe et divise le travail (via l'appel d'outils/tools).
- **DevWeb** : Spécialiste du développement front-end et web.
- **DevSoft** : Spécialiste du développement logiciel et algorithmique.
- **DevOps** : En charge de l'infrastructure et du déploiement.
- **Test** : Responsable des vérifications et de l'exécution en Sandbox Docker.
- **Évaluation** : Contrôleur qualité final avant validation.

### Outils et Modules Personnalisés
- **Titan Memory (`memory_tool`)** : Un modèle de mémoire neuronale local (basé sur PyTorch) utilisé par les Devs et l'Évaluateur pour mémoriser dynamiquement et scientifiquement le contexte (avec logs d'activation synaptique et mesure de surprise).
- **Docker Sandbox (`sandbox_tool`)** : L'agent Test dispose d'un module d'exécution sécurisée isolant totalement le code généré via l'API Docker.
- **Espace de Travail (`workspace_tool`)** : Les agents sauvegardent physiquement leur code dans le répertoire `workspace/` afin que l'utilisateur puisse le consulter et l'utiliser avant mise en production.

---

## Installation & Prérequis

### 1. Préparation de l'Environnement Python
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configuration d'Ollama
Le système utilise **Qwen 2.5 (3B)**, un modèle extrêmement performant pour l'orchestration des agents et le Tool Calling tout en restant très léger (idéal pour petites machines).
Assurez-vous d'avoir installé [Ollama](https://ollama.com/), puis téléchargez le modèle :
```bash
ollama pull qwen2.5:3b
```

### 3. Configuration de Docker
L'agent Test nécessite que **Docker Desktop** (ou le moteur Docker) soit installé et en cours d'exécution sur votre machine.

---

## Démarrage de l'Usine Logicielle

Pour lancer l'application et interagir avec le Manager via l'interface interactive :

```bash
python main.py
```

### Dossiers générés automatiquement :
- `workspace/` : Contient tout le code source généré par l'équipe.
- `logs/` : Contient l'historique complet (ex: `memory_tracker.log`) avec les métriques scientifiques de la mémoire Titan (taux de saturation, activations LTM).
- Un fichier local `neural_memory.pt` sera créé à la racine lors du premier démarrage pour stocker les poids dynamiques.