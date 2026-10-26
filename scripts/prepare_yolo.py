"""Prépare le dataset au format YOLO dans data/yolo/ à partir des annotations et du découpage.

Lancer depuis la racine du projet :  uv run python scripts/prepare_yolo.py
(à relancer si le découpage data/splits/splits.csv change)
"""

import pandas as pd

from avions.constants import SPLITS_FILE
from avions.data.split import SPLIT_NAMES
from avions.data.to_yolo import build_yolo_dataset
from avions.data.voc import load_annotations


def main():
    objects = load_annotations()
    split = pd.read_csv(SPLITS_FILE, dtype={"image_id": str}).set_index("image_id")["split"]

    yaml_path = build_yolo_dataset(objects, split)

    print(f"Dataset YOLO écrit : {yaml_path.parent}")
    for name in SPLIT_NAMES:
        n_images = (split == name).sum()
        n_objects = objects["image_id"].map(split).eq(name).sum()
        print(f"  {name:5s} : {n_images:4d} images, {n_objects:4d} avions")


if __name__ == "__main__":
    main()
