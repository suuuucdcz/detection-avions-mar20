"""Fonctions d'affichage : dessiner les boîtes et les classes sur une image.

draw_boxes et draw_detections servent pour l'EDA et le corpus annoté ; draw_tagged_boxes sert
pour l'interface (étiquettes lisibles sur tous les fonds, avec accents, ✓ et ✗).
"""

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from avions.constants import CLASS_TO_INDEX

# Police livrée avec matplotlib : elle contient é, à, ×, ✓ et ✗ (OpenCV n'écrit que l'ASCII)
FONT_PATH = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
HALO = (15, 23, 42)  # contour sombre sous chaque trait : la boîte reste visible sur tous les fonds
WHITE = (255, 255, 255)


def class_color(class_name: str) -> tuple[int, int, int]:
    """Couleur RGB (0-255) fixe pour chaque classe, tirée de la palette « tab20 » (20 couleurs)."""
    r, g, b, _ = plt.get_cmap("tab20")(CLASS_TO_INDEX[class_name])
    return int(r * 255), int(g * 255), int(b * 255)


def draw_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    labels: list[str],
    thickness: int = 2,
) -> np.ndarray:
    """Dessine des boîtes (xmin, ymin, xmax, ymax) et leur étiquette sur une COPIE de l'image RGB.

    Chaque étiquette doit commencer par un nom de classe (ex. "A10 0.93") : il sert à choisir
    la couleur de la boîte.
    """
    image = image.copy()  # on ne modifie pas l'image d'origine
    colors = [class_color(label.split()[0]) for label in labels]
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


@dataclass
class TaggedBox:
    """Une boîte à dessiner par draw_tagged_boxes."""

    box: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax en pixels
    color: tuple[int, int, int]  # RGB
    label: str = ""  # vide : pas d'étiquette
    dashed: bool = False  # pointillés (avion oublié par le modèle)


def _dashed_line(draw, start, end, color, width, dash=10, gap=6):
    (x0, y0), (x1, y1) = start, end
    length = max(abs(x1 - x0), abs(y1 - y0))
    for a in range(0, int(length), dash + gap):
        t0, t1 = a / length, min(a + dash, length) / length
        segment = [(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t) for t in (t0, t1)]
        draw.line(segment, fill=color, width=width)


def draw_tagged_boxes(image: np.ndarray, boxes: list[TaggedBox], font_size: int = 18) -> np.ndarray:
    """Dessine des boîtes et des étiquettes en pastille (fond de couleur, texte blanc) sur une COPIE
    de l'image RGB. Les boîtes sont dessinées dans l'ordre : les dernières passent au-dessus.
    """
    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture)
    font = ImageFont.truetype(str(FONT_PATH), font_size)
    width = picture.width
    for b in boxes:
        x0, y0, x1, y1 = (round(v) for v in b.box)
        draw.rectangle([x0, y0, x1, y1], outline=HALO, width=5)
        if b.dashed:
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            for start, end in pairwise(corners):
                _dashed_line(draw, start, end, b.color, width=3)
        else:
            draw.rectangle([x0 + 1, y0 + 1, x1 - 1, y1 - 1], outline=b.color, width=3)
        if not b.label:
            continue
        left, top, right, bottom = draw.textbbox((0, 0), b.label, font=font)
        tag_w, tag_h = right - left + 10, bottom - top + 6
        tx = max(0, min(x0, width - tag_w))  # recalée si elle sort à droite
        ty = y0 - tag_h - 2 if y0 - tag_h - 2 >= 0 else y0 + 3  # au-dessus, sinon dans la boîte
        draw.rounded_rectangle([tx, ty, tx + tag_w, ty + tag_h], radius=4, fill=b.color)
        draw.text((tx + 5 - left, ty + 3 - top), b.label, font=font, fill=WHITE)
    return np.asarray(picture)
