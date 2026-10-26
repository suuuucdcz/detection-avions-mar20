"""Constantes partagées par tout le projet : chemins et liste des classes.

On les définit ici, à un seul endroit, pour que tous les modules utilisent exactement
les mêmes valeurs (sinon on risque des bugs silencieux, par exemple deux ordres de
classes différents entre l'entraînement et la prédiction).
"""

import os
from pathlib import Path

# --- Chemins ----------------------------------------------------------------------
# __file__ = src/avions/constants.py  ->  parents[2] = racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_IMAGES_DIR = RAW_DIR / "images"  # 1331 images satellites .jpg
RAW_ANNOTATIONS_DIR = RAW_DIR / "annotations"  # 1331 fichiers .xml (format Pascal VOC)
SPLITS_DIR = DATA_DIR / "splits"
# Jeu (train / val / test) de chaque image, versionné dans Git
SPLITS_FILE = SPLITS_DIR / "splits.csv"
# Dataset au format YOLO (généré par scripts/prepare_yolo.py)
YOLO_DIR = DATA_DIR / "yolo"
YOLO_DATASET_YAML = YOLO_DIR / "dataset.yaml"
# Vignettes : un carré autour de chaque avion (générées par scripts/make_crops.py)
CROPS_DIR = DATA_DIR / "crops"
CROPS_INDEX = CROPS_DIR / "index.csv"

# Sorties des entraînements. Par défaut dans le projet ; sur Colab, la variable d'environnement
# AVIONS_OUTPUT_DIR les envoie directement dans le dossier du projet sur le Drive.
OUTPUT_ROOT = Path(os.environ.get("AVIONS_OUTPUT_DIR", PROJECT_ROOT))
MODELS_DIR = OUTPUT_ROOT / "models"  # poids finaux de nos modèles
RUNS_DIR = OUTPUT_ROOT / "runs"  # sorties des entraînements (courbes, métriques, exemples)
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Classes ----------------------------------------------------------------------
# Les 20 types d'avions du dataset sont codés A1 à A20.
# On les écrit dans l'ordre NUMÉRIQUE. Attention : un tri alphabétique donnerait
# A1, A10, A11, ..., A2, ... et les numéros de classe seraient décalés sans erreur visible.
CLASSES = [f"A{i}" for i in range(1, 21)]

# Correspondance nom de classe <-> numéro utilisé par les modèles (0 à 19)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
