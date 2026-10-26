"""Découpe des vignettes : un carré centré sur chaque avion.

C'est la MÊME fonction qui sert pour l'entraînement (boîtes des annotations) et pour
l'application finale (boîtes trouvées par le détecteur). Si les deux découpes différaient,
le classifieur verrait à l'usage des images différentes de celles de son entraînement.

Choix (voir docs/journal_decisions.md, D11) :
- côté du carré = 2 × le plus grand côté de la boîte : on garde du contexte autour de l'avion,
  et assez de marge pour pouvoir faire tourner la vignette sans faire apparaître de coins vides ;
- si le carré dépasse de l'image (avion au bord), on complète avec du gris neutre ;
- la vignette est enregistrée en 256 x 256 pixels : l'avion y occupe alors environ 128 pixels,
  à peu près sa taille d'origine (médiane : 111 pixels, voir l'EDA).
"""

import cv2
import numpy as np

CROP_SCALE = 2.0  # côté du carré / plus grand côté de la boîte
CROP_SIZE = 256  # taille des vignettes enregistrées, en pixels
PAD_VALUE = 128  # gris neutre pour compléter au-delà du bord de l'image


def make_crop(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    scale: float = CROP_SCALE,
    size: int = CROP_SIZE,
) -> np.ndarray:
    """Découpe un carré centré sur la boîte (xmin, ymin, xmax, ymax) et le redimensionne."""
    xmin, ymin, xmax, ymax = box
    center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    half_side = scale * max(xmax - xmin, ymax - ymin) / 2
    x0, y0 = round(center_x - half_side), round(center_y - half_side)
    x1, y1 = round(center_x + half_side), round(center_y + half_side)

    # Si le carré sort de l'image, on ajoute une bordure grise tout autour de l'image
    height, width = image.shape[:2]
    pad = max(0, -x0, -y0, x1 - width, y1 - height)
    if pad:
        image = cv2.copyMakeBorder(
            image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
        )
    square = image[y0 + pad : y1 + pad, x0 + pad : x1 + pad]

    # INTER_AREA pour réduire (net, sans crénelage), INTER_LINEAR pour agrandir
    shrinking = square.shape[0] > size
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(square, (size, size), interpolation=interpolation)
