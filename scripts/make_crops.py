"""Génère les vignettes de tous les avions annotés dans data/crops/ (étape 2 : préparation).

Lancer depuis la racine du projet :  uv run python scripts/make_crops.py

Résultat :
- data/crops/<jeu>/<image>_<numéro>.jpg : une vignette de 256 x 256 pixels par avion ;
- data/crops/index.csv : une ligne par vignette (fichier, image d'origine, jeu, classe, taille).

Les vignettes se régénèrent en moins d'une minute : elles ne sont ni dans Git ni sur le Drive.
"""

import shutil

import cv2
import pandas as pd
from tqdm import tqdm

from avions.constants import CLASS_TO_INDEX, CROPS_DIR, CROPS_INDEX, RAW_IMAGES_DIR, SPLITS_FILE
from avions.data.crops import make_crop
from avions.data.split import SPLIT_NAMES
from avions.data.voc import load_annotations


def main():
    objects = load_annotations()
    split = pd.read_csv(SPLITS_FILE, dtype={"image_id": str}).set_index("image_id")["split"]

    shutil.rmtree(CROPS_DIR, ignore_errors=True)  # dossier entièrement régénéré
    for name in SPLIT_NAMES:
        (CROPS_DIR / name).mkdir(parents=True)

    rows = []
    for image_id, boxes in tqdm(objects.groupby("image_id"), desc="Vignettes"):
        # Chaque image n'est lue qu'une seule fois, pour tous ses avions
        image = cv2.imread(str(RAW_IMAGES_DIR / f"{image_id}.jpg"))
        split_name = split[image_id]
        for number, box in enumerate(boxes.itertuples()):
            crop = make_crop(image, (box.xmin, box.ymin, box.xmax, box.ymax))
            crop_path = f"{split_name}/{image_id}_{number}.jpg"
            cv2.imwrite(str(CROPS_DIR / crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append(
                {
                    "crop_path": crop_path,
                    "image_id": image_id,
                    "split": split_name,
                    "class_name": box.class_name,
                    "label": CLASS_TO_INDEX[box.class_name],
                    "box_width": box.box_width,
                    "box_height": box.box_height,
                    "truncated": box.truncated,
                }
            )

    index = pd.DataFrame(rows)
    index.to_csv(CROPS_INDEX, index=False)
    print(f"{len(index)} vignettes écrites dans {CROPS_DIR}")
    print(index["split"].value_counts().reindex(SPLIT_NAMES).to_string())


if __name__ == "__main__":
    main()
