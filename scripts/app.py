"""Lance l'interface web de démonstration.

Lancer depuis la racine du projet :  uv run python scripts/app.py
La page s'ouvre toute seule dans le navigateur (http://127.0.0.1:7860, ou le port libre suivant).
Arrêter : Ctrl + C dans le terminal.
"""

from avions.app import THEME, creer_demo

if __name__ == "__main__":
    creer_demo().launch(inbrowser=True, theme=THEME)
