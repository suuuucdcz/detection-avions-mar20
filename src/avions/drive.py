"""Accès au dossier du projet sur Google Drive, via l'application « Google Drive pour ordinateur ».

Cette application fait apparaître le Drive comme un lecteur du PC (ex. G:\\Mon Drive) :
copier un fichier dedans suffit pour qu'il soit envoyé sur le Drive, et les fichiers créés
par Colab sur le Drive (modèles entraînés) apparaissent automatiquement sur le PC.

Le dossier « projet_avions » du Drive a la même arborescence que le projet sur le PC
(synchronisé par scripts/sync_drive.py). Colab y lit le code et les données, et y écrit
ses résultats dans runs/ (courbes, métriques) et models/ (poids des modèles entraînés).
"""

import os
from pathlib import Path

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
