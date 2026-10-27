"""Conversion des annotations Pascal VOC vers le format attendu par YOLO (Ultralytics).

Format YOLO : un fichier texte par image, une ligne par avion :

    classe  x_centre  y_centre  largeur  hauteur

- classe : numéro de 0 à 19 (A1 -> 0, ..., A20 -> 19) ;
- les 4 nombres sont divisés par la taille de l'image : ils vont de 0 à 1.
  Ainsi les étiquettes restent valables même si YOLO redimensionne l'image.

Exemple : un A2 dont la boîte va de (275, 22) à (385, 149) dans une image 800 x 800
-> "1 0.412500 0.106875 0.137500 0.158750"

Organisation attendue par Ultralytics (dossiers « images » et « labels » jumeaux) :

    data/yolo/images/train/1005.jpg   <->   data/yolo/labels/train/1005.txt
"""

import os
import shutil
from pathlib import Path

import pandas as pd
import yaml

from avions.constants import CLASS_TO_INDEX, CLASSES, RAW_IMAGES_DIR, YOLO_DIR
from avions.data.split import SPLIT_NAMES


def voc_to_yolo_lines(boxes: pd.DataFrame) -> list[str]:
    """Convertit les avions d'UNE image (une ligne par avion) en lignes au format YOLO."""
    lines = []
    for box in boxes.itertuples():
        x_center = (box.xmin + box.xmax) / 2 / box.image_width
        y_center = (box.ymin + box.ymax) / 2 / box.image_height
        width = (box.xmax - box.xmin) / box.image_width
        height = (box.ymax - box.ymin) / box.image_height
        class_index = CLASS_TO_INDEX[box.class_name]
        lines.append(f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return lines


def link_or_copy(source: Path, destination: Path) -> None:
    """Crée un « lien dur » vers l'image (aucune place disque en plus), sinon la copie."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_yolo_dataset(objects: pd.DataFrame, split: pd.Series, out_dir: Path = YOLO_DIR) -> Path:
    """Construit le dataset YOLO (images, étiquettes, fichier YAML) et renvoie le chemin du YAML.

    objects : une ligne par avion (sortie de load_annotations) ; split : jeu de chaque image.
    Le dossier est entièrement régénéré à chaque appel (il ne contient que des fichiers dérivés).
    """
    shutil.rmtree(out_dir, ignore_errors=True)
    for name in SPLIT_NAMES:
        (out_dir / "images" / name).mkdir(parents=True)
        (out_dir / "labels" / name).mkdir(parents=True)

    for image_id, boxes in objects.groupby("image_id"):
        name = split[image_id]
        link_or_copy(
            RAW_IMAGES_DIR / f"{image_id}.jpg", out_dir / "images" / name / f"{image_id}.jpg"
        )
        labels = "\n".join(voc_to_yolo_lines(boxes)) + "\n"
        (out_dir / "labels" / name / f"{image_id}.txt").write_text(labels)

    # Fichier de description du dataset, lu par Ultralytics
    description = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASSES)),
    }
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(description, sort_keys=False), encoding="utf-8")
    return yaml_path
