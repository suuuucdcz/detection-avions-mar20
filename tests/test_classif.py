"""Tests des vignettes et des briques de classification."""

import numpy as np
import torch
from PIL import Image

from avions.classif.baseline import baseline_features
from avions.classif.models import build_model
from avions.classif.transforms import IMAGE_SIZE, eval_transform, train_transform
from avions.data.crops import CROP_SIZE, PAD_VALUE, make_crop


def image_test():
    """Image 800 x 800 noire avec un « avion » blanc de 100 x 50 pixels centré en (400, 300)."""
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    image[275:325, 350:450] = 255
    return image


def test_vignette_centree_sur_l_avion():
    crop = make_crop(image_test(), (350, 275, 450, 325))
    assert crop.shape == (CROP_SIZE, CROP_SIZE, 3)
    # Carré de 2 × 100 = 200 px agrandi en 256 : l'avion (100 x 50) occupe le centre, 128 x 64 px
    assert crop[128, 128].min() == 255  # centre : l'avion
    assert crop[10, 10].max() == 0  # coin : le fond


def test_vignette_au_bord_completee_en_gris():
    crop = make_crop(image_test(), (0, 0, 100, 100))  # avion collé au coin de l'image
    assert crop.shape == (CROP_SIZE, CROP_SIZE, 3)
    assert (crop[0, 0] == PAD_VALUE).all()  # au-delà du bord : gris neutre


def test_transformations_donnent_la_bonne_taille():
    vignette = Image.fromarray(make_crop(image_test(), (350, 275, 450, 325)))
    for transform in (train_transform(), eval_transform()):
        tensor = transform(vignette)
        assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert tensor.dtype == torch.float32


def test_caracteristiques_de_la_baseline():
    crop = make_crop(image_test(), (350, 275, 450, 325))
    features = baseline_features(crop, 100, 50)
    assert features.shape == (1764 + 3,)  # HOG + largeur, hauteur, plus grand côté


def test_les_reseaux_ont_20_sorties():
    images = torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    for architecture in ("resnet18", "smallcnn"):
        model = build_model(architecture, pretrained=False).eval()
        assert model(images).shape == (2, 20)
