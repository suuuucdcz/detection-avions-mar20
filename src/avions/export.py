"""Export des prédictions au format Pascal VOC, le même format que les annotations du dataset.

Un fichier XML par image, une balise <object> par avion détecté, avec en plus :
- <score> : la confiance finale du modèle (détecteur × classifieur), entre 0 et 1 ;
- <a_verifier> : 1 si l'avion est « à vérifier » par un humain (type incertain, D14), 0 sinon.
Les fichiers restent lisibles par tout outil qui lit le format Pascal VOC (et par notre
fonction read_voc_file), qui ignore simplement ces deux balises supplémentaires.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

from avions.pipeline import Detection

BORDER = 2  # pixels : une boîte qui touche le bord de l'image est marquée « tronquée »


def _add(parent: ET.Element, tag: str, text) -> ET.Element:
    element = ET.SubElement(parent, tag)
    element.text = str(text)
    return element


def detections_to_voc(
    image_name: str, width: int, height: int, detections: list[Detection]
) -> ET.ElementTree:
    root = ET.Element("annotation")
    _add(root, "folder", "corpus_test_annote")
    _add(root, "filename", image_name)
    source = ET.SubElement(root, "source")
    _add(
        source, "annotation", "prédictions du modèle : détecteur YOLO11n puis classifieur ResNet-18"
    )
    size = ET.SubElement(root, "size")
    _add(size, "width", width)
    _add(size, "height", height)
    _add(size, "depth", 3)
    _add(root, "segmented", 0)

    for d in detections:
        xmin, ymin, xmax, ymax = (round(v) for v in d.box)
        touches_border = (
            xmin <= BORDER or ymin <= BORDER or xmax >= width - BORDER or ymax >= height - BORDER
        )
        obj = ET.SubElement(root, "object")
        _add(obj, "name", d.class_name)
        _add(obj, "pose", "Unspecified")
        _add(obj, "truncated", int(touches_border))
        _add(obj, "difficult", 0)
        box = ET.SubElement(obj, "bndbox")
        for tag, value in zip(
            ("xmin", "ymin", "xmax", "ymax"), (xmin, ymin, xmax, ymax), strict=True
        ):
            _add(box, tag, value)
        _add(obj, "score", f"{d.score:.4f}")
        _add(obj, "a_verifier", int(d.to_review))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t")  # même mise en forme que les fichiers du dataset
    return tree


def write_voc(path: Path, image_name: str, width: int, height: int, detections) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    detections_to_voc(image_name, width, height, detections).write(
        path, encoding="utf-8", xml_declaration=False
    )
