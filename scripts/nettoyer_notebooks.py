"""Retire des notebooks les informations personnelles, sans toucher aux résultats.

Lancer depuis la racine du projet (avant chaque commit, après un passage par Colab) :
    uv run python scripts/nettoyer_notebooks.py

Ce qui est retiré :
- les traces d'exécution que Colab enregistre dans chaque cellule (« executionInfo » : nom et
  identifiant du compte Google, heure d'exécution) et ses autres métadonnées de cellule ;
- l'historique des fichiers Drive d'origine (« provenance ») ;
- le chemin du projet sur ce PC dans les sorties affichées (remplacé par « projet_ia »).
"""

import nbformat

from avions.constants import PROJECT_ROOT

CELL_KEYS_TO_REMOVE = ("executionInfo", "colab", "outputId")


def clean_text(text: str) -> str:
    for path in {
        str(PROJECT_ROOT),
        PROJECT_ROOT.as_posix(),
        str(PROJECT_ROOT).replace("\\", "\\\\"),
    }:
        text = text.replace(path, "projet_ia")
    return text


def clean_notebook(path) -> bool:
    """Nettoie un notebook ; renvoie True s'il a été modifié."""
    notebook = nbformat.read(path, as_version=4)
    before = nbformat.writes(notebook)

    if "colab" in notebook.metadata:
        notebook.metadata["colab"]["provenance"] = []
    for cell in notebook.cells:
        for key in CELL_KEYS_TO_REMOVE:
            cell.metadata.pop(key, None)
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                output["text"] = clean_text(output["text"])
            elif "text/plain" in output.get("data", {}):
                output["data"]["text/plain"] = clean_text(output["data"]["text/plain"])

    if nbformat.writes(notebook) == before:
        return False
    nbformat.write(notebook, path)
    return True


def main():
    for path in sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")):
        print(f"{path.name:32s} {'nettoyé' if clean_notebook(path) else 'déjà propre'}")


if __name__ == "__main__":
    main()
