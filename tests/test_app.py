"""Tests de l'interface : elle ne propose que des images jamais vues à l'entraînement."""

import pandas as pd

from avions.app import charger_images_non_vues
from avions.constants import SPLITS_FILE


def test_seules_les_images_jamais_vues_sont_proposees():
    _, images = charger_images_non_vues()
    jeux = pd.read_csv(SPLITS_FILE, dtype={"image_id": str}).set_index("image_id")["split"]

    assert set(images["jeu"]) == {"val", "test"}
    assert len(images) == 130 + 131
    assert not (jeux[images.index] == "train").any()  # aucune image d'entraînement
