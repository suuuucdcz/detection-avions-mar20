"""Transformations appliquées aux vignettes avant de les donner au réseau.

Entraînement = DATA AUGMENTATION : à chaque passage, la vignette est modifiée au hasard
(rotation, retournement, cadrage, luminosité). Le réseau voit ainsi des variantes réalistes
et apprend ce qui définit un type d'avion plutôt que d'apprendre les images par cœur.
Choix justifiés par l'EDA (question 3) : vu du ciel, l'avion peut pointer dans toutes les
directions, et la luminosité varie beaucoup d'une image à l'autre.

Validation / test / application : pas de hasard, juste un cadrage fixe au centre.
"""

import random

import torch
from torchvision.transforms import v2

from avions.data.crops import CROP_SCALE, CROP_SIZE, PAD_VALUE

IMAGE_SIZE = 224  # taille d'entrée des réseaux pré-entraînés sur ImageNet
# Moyenne et écart-type des pixels d'ImageNet : les poids pré-entraînés ont appris avec ces valeurs
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Dans une vignette de 256 px (carré de 2 × la boîte), la boîte de l'avion mesure 128 px
BOX_SIDE = CROP_SIZE / CROP_SCALE
EVAL_CROP_FACTOR = 1.2  # cadrage fixe : 1,2 × la boîte, un peu de marge autour de l'avion


class JitteredCenterCrop:
    """Cadrage aléatoire autour de l'avion : un peu plus ou moins large, un peu décalé.

    Imite les boîtes imparfaites que fournira le détecteur (trop grandes, trop petites,
    décalées), pour que le classifieur y soit habitué. Les bornes (1,1 à 1,3 × la boîte,
    5 % de décalage) restent dans le cercle laissé intact par la rotation : aucun coin vide.
    """

    def __init__(self, min_factor: float = 1.1, max_factor: float = 1.3, max_shift: float = 0.05):
        self.min_factor, self.max_factor, self.max_shift = min_factor, max_factor, max_shift

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        side = round(BOX_SIDE * random.uniform(self.min_factor, self.max_factor))
        shift_x = round(BOX_SIDE * random.uniform(-self.max_shift, self.max_shift))
        shift_y = round(BOX_SIDE * random.uniform(-self.max_shift, self.max_shift))
        top = (CROP_SIZE - side) // 2 + shift_y
        left = (CROP_SIZE - side) // 2 + shift_x
        return v2.functional.crop(image, top, left, side, side)


def _to_normalized_tensor() -> list:
    return [
        v2.Resize(IMAGE_SIZE, antialias=True),
        v2.ToDtype(torch.float32, scale=True),  # pixels 0-255 -> 0-1
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]


def train_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomRotation(degrees=180, fill=PAD_VALUE),  # n'importe quel angle
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            JitteredCenterCrop(),
            v2.ColorJitter(brightness=0.2, contrast=0.2),  # ±20 % de luminosité et de contraste
            *_to_normalized_tensor(),
        ]
    )


def eval_transform() -> v2.Compose:
    eval_side = round(BOX_SIDE * EVAL_CROP_FACTOR)
    return v2.Compose([v2.ToImage(), v2.CenterCrop(eval_side), *_to_normalized_tensor()])
