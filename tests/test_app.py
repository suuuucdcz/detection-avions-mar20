"""Tests de l'interface : elle ne propose que des images de test, jamais vues à l'entraînement."""

import pandas as pd

from avions.app import IMAGES_CONSEILLEES, charger_images_test
from avions.constants import SPLITS_FILE


def test_seules_les_images_de_test_sont_proposees():
    verite, ordre = charger_images_test()
    jeux = pd.read_csv(SPLITS_FILE, dtype={"image_id": str}).set_index("image_id")["split"]

    assert len(ordre) == len(set(ordre)) == 131  # les 131 images de test, chacune une fois
    assert (jeux[ordre] == "test").all()  # aucune image d'entraînement ni de validation
    assert set(verite["image_id"]) == set(ordre)
    assert ordre[: len(IMAGES_CONSEILLEES)] == list(IMAGES_CONSEILLEES)  # conseillées d'abord
