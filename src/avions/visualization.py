"""Fonctions d'affichage : dessiner les boîtes et les classes sur une image.

Elles servent pour l'EDA, et serviront plus tard pour afficher les prédictions du modèle.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

from avions.constants import CLASS_TO_INDEX


def class_color(class_name: str) -> tuple[int, int, int]:
    """Couleur RGB (0-255) fixe pour chaque classe, prise dans la palette « tab20 » (20 couleurs)."""
    r, g, b, _ = plt.get_cmap("tab20")(CLASS_TO_INDEX[class_name])
    return int(r * 255), int(g * 255), int(b * 255)


def draw_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    labels: list[str],
    thickness: int = 2,
    colors: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Dessine des boîtes (xmin, ymin, xmax, ymax) et leur étiquette sur une COPIE de l'image RGB.

    Sans `colors`, chaque étiquette doit commencer par un nom de classe (ex. "A10 0.93") : il
    sert à choisir la couleur de la boîte. Avec `colors`, chaque boîte prend la couleur donnée.
    """
    image = image.copy()  # on ne modifie pas l'image d'origine
    colors = colors or [class_color(label.split()[0]) for label in labels]
    for (xmin, ymin, xmax, ymax), label, color in zip(boxes, labels, colors, strict=True):
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
        cv2.putText(
            image, label, (xmin, max(ymin - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
    return image


def draw_detections(image: np.ndarray, detections) -> np.ndarray:
    """Dessine les prédictions du modèle sur une copie de l'image RGB.

    Avions sûrs : boîtes épaisses ; avions « à vérifier » : boîtes fines marquées « ? ».
    """
    for to_review, thickness, mark in ((False, 3, ""), (True, 1, " ?")):
        group = [d for d in detections if d.to_review == to_review]
        image = draw_boxes(
            image,
            [tuple(round(v) for v in d.box) for d in group],
            [f"{d.class_name}{mark} {d.score:.2f}" for d in group],
            thickness=thickness,
        )
    return image
