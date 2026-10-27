"""Lance l'interface web de démonstration.

Lancer depuis la racine du projet :  uv run python scripts/app.py
Puis ouvrir http://127.0.0.1:7860 dans le navigateur (la page s'ouvre toute seule).
Arrêter : Ctrl + C dans le terminal.
"""

import argparse

from avions.app import CSS, JS, THEME, creer_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--hote", default="127.0.0.1", help="0.0.0.0 pour l'ouvrir depuis l'extérieur (Docker)"
    )
    parser.add_argument("--sans-navigateur", action="store_true", dest="no_browser")
    args = parser.parse_args()

    creer_demo().launch(
        server_name=args.hote,
        server_port=args.port,
        inbrowser=not args.no_browser,
        theme=THEME,  # couleurs et police de l'interface
        css=CSS,  # disposition et style
        js=JS,  # thème clair forcé
        footer_links=[],  # pas de pied de page Gradio
    )


if __name__ == "__main__":
    main()
