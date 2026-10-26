"""Chargement des images."""

from pathlib import Path

import cv2
import numpy as np


def load_image_rgb(image_path: Path) -> np.ndarray:
    """Charge une image en couleurs, au format RGB (tableau hauteur x largeur x 3).

    OpenCV lit les images en BGR (bleu, vert, rouge) pour des raisons historiques,
    alors que matplotlib et PyTorch attendent du RGB : on inverse donc les canaux.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
