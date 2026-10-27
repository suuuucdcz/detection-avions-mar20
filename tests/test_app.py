"""Tests de l'interface : seulement des images de test, et une correction en oui / non."""

import numpy as np
import pandas as pd

from avions.app import EXEMPLE, charger_images_test, corriger, f1_md
from avions.constants import SPLITS_FILE
from avions.pipeline import Detection
from avions.visualization import TaggedBox, draw_tagged_boxes


def test_seules_les_images_de_test_sont_proposees():
    verite, ordre = charger_images_test()
    jeux = pd.read_csv(SPLITS_FILE, dtype={"image_id": str}).set_index("image_id")["split"]

    assert len(ordre) == len(set(ordre)) == 131  # les 131 images de test, chacune une fois
    assert (jeux[ordre] == "test").all()  # aucune image d'entraînement ni de validation
    assert set(verite["image_id"]) == set(ordre)
    assert ordre[0] == EXEMPLE  # l'exemple de l'oral s'affiche en premier


def test_correction_oui_ou_non():
    # 3 vrais avions : un A1, un A2, un A3 (loin des autres)
    verite = pd.DataFrame(
        [[0, 0, 10, 10, "A1"], [20, 0, 30, 10, "A2"], [40, 0, 50, 10, "A3"]],
        columns=["xmin", "ymin", "xmax", "ymax", "class_name"],
    )
    reponses = [
        Detection((0, 0, 10, 10), "A1", 0.9, 0.9, 1.0),  # juste
        Detection((0, 0, 10, 9), "A1", 0.85, 0.9, 0.95),  # 2e boîte sur le même A1 : doublon
        Detection((20, 0, 30, 10), "A5", 0.8, 0.9, 0.9),  # mauvais type (vrai : A2)
        Detection((80, 80, 90, 90), "A1", 0.7, 0.8, 0.9),  # aucun avion annoté ici
    ]
    c = corriger(verite, reponses)

    assert [d.class_name for d in c.justes] == ["A1"]
    assert [raison for _, raison in c.fausses] == ["doublon", "vrai : A2", "aucun avion annoté"]
    assert [vrai for _, vrai in c.oublies] == ["A3"]  # l'A3 n'a reçu aucune réponse
    # 1 juste, 4 réponses, 3 vrais avions
    assert c.comptes() == (1, 4, 3)


def test_f1_cumule_compte_des_avions_pas_des_images():
    # deux images : l'une sans faute (2 avions), l'autre ratée (1 juste sur 50 réponses et avions)
    cumul = {"a": (2, 2, 2), "b": (1, 50, 50)}
    # on additionne les avions : 2 × 3 / (52 + 52), pas la moyenne des deux F1 (1 et 0,02)
    assert "0,058" in f1_md(cumul)
    # avec les comptes officiels du test (729 justes, 741 réponses, 788 avions) : 0,954
    assert "0,954" in f1_md({"tout": (729, 741, 788)})


def test_dessin_ne_modifie_pas_l_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    boites = [
        TaggedBox((10, 10, 50, 50), (22, 163, 74), "✓ A1"),
        TaggedBox((60, 60, 95, 95), (220, 38, 38), "✗ oublié : A2", dashed=True),
    ]
    resultat = draw_tagged_boxes(image, boites)
    assert image.sum() == 0  # l'original est intact
    assert resultat.shape == image.shape and resultat.sum() > 0
