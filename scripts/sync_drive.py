"""Synchronise le projet entre le PC et Google Drive (dossier « Mon Drive/projet_avions »).

Lancer depuis la racine du projet :  uv run python scripts/sync_drive.py

Le Drive contient exactement la même arborescence que le projet sur le PC (src/, scripts/,
configs/, data/, notebooks/, docs/, reports/, models/, runs/...). On peut donc suivre chaque
étape du projet depuis le Drive, et Colab travaille sur les mêmes fichiers.

Règle : pour chaque fichier, LA VERSION LA PLUS RÉCENTE GAGNE et elle est copiée de l'autre côté.
- Code, documents, données modifiés sur le PC  ->  envoyés sur le Drive ;
- modèles et résultats produits par Colab sur le Drive  ->  ramenés sur le PC.
Les suppressions ne sont pas recopiées (on ne supprime jamais rien de l'autre côté).

Ne sont pas synchronisés : l'environnement Python (.venv), l'historique Git (.git), les caches,
le dataset YOLO et les vignettes (data/yolo, data/crops : régénérés en moins d'une minute) et
les points de sauvegarde intermédiaires des entraînements (dossiers weights/ ; les poids
finaux sont dans models/).
"""

import os
import shutil
from collections import Counter
from pathlib import Path

from avions.constants import PROJECT_ROOT

# Le Drive apparaît sur le PC comme un lecteur grâce à « Google Drive pour ordinateur » ;
# le dossier « projet_avions » a la même arborescence que le projet.
DRIVE_FOLDER_NAME = "projet_avions"
# Emplacements habituels du Drive sur le PC (le nom dépend de la langue de Windows).
# Sur un autre PC (ex. celui du binôme), on peut indiquer le sien avec la variable
# d'environnement AVIONS_DRIVE_DIR.
DRIVE_ROOT_CANDIDATES = [Path("G:/Mon Drive"), Path("G:/My Drive")]


def find_drive_project_dir() -> Path | None:
    """Renvoie le dossier « projet_avions » du Drive sur ce PC, ou None si le Drive est absent."""
    if "AVIONS_DRIVE_DIR" in os.environ:
        return Path(os.environ["AVIONS_DRIVE_DIR"])
    for root in DRIVE_ROOT_CANDIDATES:
        if root.exists():
            return root / DRIVE_FOLDER_NAME
    return None


EXCLUDED_NAMES = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "weights",
    "yolo11n.pt",
    "yolo26n.pt",
}
EXCLUDED_PATHS = {"data/yolo", "data/crops"}
MTIME_TOLERANCE = 2  # secondes : le Drive arrondit parfois les dates de modification


def list_files(root: Path) -> dict[str, Path]:
    """Les fichiers à synchroniser sous root, indexés par chemin relatif (ex. 'src/a.py')."""
    files = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        excluded = any(part in EXCLUDED_NAMES for part in path.relative_to(root).parts) or any(
            relative == p or relative.startswith(p + "/") for p in EXCLUDED_PATHS
        )
        if path.is_file() and not excluded and not relative.endswith(".cache"):
            files[relative] = path
    return files


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)  # copy2 garde la date de modification -> pas d'aller-retour


def main():
    drive_dir = find_drive_project_dir()
    if drive_dir is None:
        raise SystemExit("Google Drive pour ordinateur introuvable (lecteur G:\\Mon Drive absent).")
    drive_dir.mkdir(parents=True, exist_ok=True)

    local_files, drive_files = list_files(PROJECT_ROOT), list_files(drive_dir)
    # Nombre de fichiers copiés dans chaque sens, par dossier de premier niveau
    sent, received = Counter(), Counter()
    for relative in sorted(local_files.keys() | drive_files.keys()):
        local, drive = PROJECT_ROOT / relative, drive_dir / relative
        top_folder = relative.split("/")[0]
        if relative not in drive_files:
            copy(local, drive)
            sent[top_folder] += 1
        elif relative not in local_files:
            copy(drive, local)
            received[top_folder] += 1
        elif local.stat().st_mtime > drive.stat().st_mtime + MTIME_TOLERANCE:
            copy(local, drive)
            sent[top_folder] += 1
        elif drive.stat().st_mtime > local.stat().st_mtime + MTIME_TOLERANCE:
            copy(drive, local)
            received[top_folder] += 1

    print(f"Drive : {drive_dir}")
    print(
        "  PC -> Drive : " + (", ".join(f"{k} ({v})" for k, v in sent.items()) or "rien à envoyer")
    )
    print(
        "  Drive -> PC : "
        + (", ".join(f"{k} ({v})" for k, v in received.items()) or "rien à récupérer")
    )
    if sent:
        print("L'application Google Drive envoie les fichiers en ligne (icône près de l'horloge).")


if __name__ == "__main__":
    main()
