"""Tests des constantes : on vérifie automatiquement des hypothèses dont tout le projet dépend.

Lancer les tests :  uv run pytest
"""

from avions.constants import CLASS_TO_INDEX, CLASSES, RAW_ANNOTATIONS_DIR, RAW_IMAGES_DIR


def test_il_y_a_20_classes_dans_l_ordre_numerique():
    assert len(CLASSES) == 20
    assert CLASSES[0] == "A1"
    assert CLASSES[1] == "A2"  # et non "A10" (piège du tri alphabétique)
    assert CLASSES[-1] == "A20"


def test_correspondance_classe_index():
    assert CLASS_TO_INDEX["A1"] == 0
    assert CLASS_TO_INDEX["A20"] == 19


def test_chaque_image_a_son_annotation():
    images = {p.stem for p in RAW_IMAGES_DIR.glob("*.jpg")}
    annotations = {p.stem for p in RAW_ANNOTATIONS_DIR.glob("*.xml")}
    assert len(images) == 1331
    assert images == annotations
