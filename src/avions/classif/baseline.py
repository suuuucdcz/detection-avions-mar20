"""Caractéristiques « faites à la main » pour la baseline (forêt aléatoire).

Machine learning classique : un algorithme comme la forêt aléatoire ne sait pas regarder
une image directement. On lui donne des nombres qui la résument :
- HOG (histogrammes de gradients orientés) : on découpe l'image en petites cellules et on
  compte, dans chaque cellule, dans quelles directions vont les contours. Ça décrit la FORME
  de l'avion (ailes, fuselage) ;
- la taille de la boîte en pixels : l'EDA a montré que la taille dépend du type d'avion.

Un CNN, lui, apprend tout seul quelles caractéristiques extraire : c'est la différence
entre machine learning classique et deep learning (notion de l'oral n°1).
"""

import cv2
import numpy as np

from avions.classif.transforms import BOX_SIDE, EVAL_CROP_FACTOR
from avions.data.crops import CROP_SIZE

HOG_SIZE = 64
# Image 64 x 64, blocs de 16 x 16 qui avancent de 8 pixels, cellules de 8 x 8, 9 directions
# -> 7 x 7 blocs x 4 cellules x 9 directions = 1764 nombres par vignette
HOG = cv2.HOGDescriptor((HOG_SIZE, HOG_SIZE), (16, 16), (8, 8), (8, 8), 9)


def hog_features(crop_bgr: np.ndarray) -> np.ndarray:
    """HOG du centre de la vignette (même cadrage que pour le réseau : 1,2 × la boîte)."""
    side = round(BOX_SIDE * EVAL_CROP_FACTOR)
    start = (CROP_SIZE - side) // 2
    center = crop_bgr[start : start + side, start : start + side]
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA)
    return HOG.compute(small).ravel()


def baseline_features(crop_bgr: np.ndarray, box_width: float, box_height: float) -> np.ndarray:
    """HOG + taille de la boîte (largeur, hauteur, plus grand côté)."""
    size = [box_width, box_height, max(box_width, box_height)]
    return np.concatenate([hog_features(crop_bgr), np.array(size, dtype=np.float32)])
