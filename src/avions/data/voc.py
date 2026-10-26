"""Lecture des annotations au format Pascal VOC.

Chaque image `XXXX.jpg` a un fichier `XXXX.xml` qui décrit les avions présents.
Pour chaque avion, il donne sa classe (`<name>`, ex. "A10") et sa boîte englobante
(`<bndbox>`) : coordonnées en pixels du coin haut-gauche (xmin, ymin) et du coin
bas-droit (xmax, ymax). L'origine (0, 0) est en haut à gauche de l'image.

On transforme tout ça en un tableau pandas (DataFrame) avec UNE LIGNE PAR AVION :
c'est beaucoup plus pratique pour compter, filtrer et tracer des graphiques.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from avions.constants import RAW_ANNOTATIONS_DIR


def read_voc_file(xml_path: Path) -> list[dict]:
    """Lit un fichier XML et renvoie une liste de dictionnaires, un par avion annoté."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    image_width = int(size.findtext("width"))
    image_height = int(size.findtext("height"))

    rows = []
    for obj in root.iter("object"):
        box = obj.find("bndbox")
        rows.append(
            {
                "image_id": Path(xml_path).stem,  # "1005" pour "1005.xml"
                "image_width": image_width,
                "image_height": image_height,
                "class_name": obj.findtext("name"),
                "xmin": int(box.findtext("xmin")),
                "ymin": int(box.findtext("ymin")),
                "xmax": int(box.findtext("xmax")),
                "ymax": int(box.findtext("ymax")),
                # "1" si l'avion est coupé par le bord de l'image
                "truncated": obj.findtext("truncated") == "1",
                # "1" si l'annotateur a jugé l'avion difficile à reconnaître
                "difficult": obj.findtext("difficult") == "1",
            }
        )
    return rows


def load_annotations(annotations_dir: Path = RAW_ANNOTATIONS_DIR) -> pd.DataFrame:
    """Charge tous les fichiers XML d'un dossier dans un tableau : une ligne par avion.

    On ajoute aussi des colonnes calculées à partir des coordonnées de la boîte :
    largeur, hauteur, aire, rapport largeur/hauteur et position du centre.
    """
    rows = []
    for xml_path in sorted(Path(annotations_dir).glob("*.xml")):
        rows.extend(read_voc_file(xml_path))
    df = pd.DataFrame(rows)

    df["box_width"] = df["xmax"] - df["xmin"]
    df["box_height"] = df["ymax"] - df["ymin"]
    df["box_area"] = df["box_width"] * df["box_height"]
    df["aspect_ratio"] = df["box_width"] / df["box_height"]  # > 1 : boîte plus large que haute
    # Centre de la boîte, en proportion de la taille de l'image (0 = bord gauche/haut, 1 = droit/bas)
    df["center_x"] = (df["xmin"] + df["xmax"]) / 2 / df["image_width"]
    df["center_y"] = (df["ymin"] + df["ymax"]) / 2 / df["image_height"]
    return df
