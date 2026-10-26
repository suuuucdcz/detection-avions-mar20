"""Tests du découpage train / validation / test."""

import pandas as pd
import pytest

from avions.constants import CLASSES, SPLITS_FILE
from avions.data.split import split_images
from avions.data.voc import load_annotations


def faux_objets(n_images=60):
    """Petit jeu de données inventé : 3 avions par image, 4 classes."""
    rows = []
    for i in range(n_images):
        for _ in range(3):
            rows.append({"image_id": f"img{i:03d}", "class_name": f"A{i % 4 + 1}"})
    return pd.DataFrame(rows)


def test_chaque_image_a_un_seul_jeu():
    objets = faux_objets()
    groupes = pd.Series(range(60), index=sorted(objets["image_id"].unique()))
    split = split_images(objets, groupes)
    assert len(split) == 60  # toutes les images sont réparties, une seule fois chacune
    assert set(split) == {"train", "val", "test"}


def test_les_quasi_doublons_restent_ensemble():
    objets = faux_objets()
    groupes = pd.Series(range(60), index=sorted(objets["image_id"].unique()))
    groupes["img001"] = groupes["img000"]  # img000 et img001 sont déclarées quasi identiques
    for seed in range(10):
        split = split_images(objets, groupes, seed)
        assert split["img000"] == split["img001"]


fichier_absent = pytest.mark.skipif(
    not SPLITS_FILE.exists(), reason="lancer scripts/make_splits.py"
)


@fichier_absent
def test_fichier_de_split_complet_et_coherent():
    table = pd.read_csv(SPLITS_FILE, dtype={"image_id": str}).set_index("image_id")
    objets = load_annotations()
    assert set(table.index) == set(objets["image_id"])  # les 1331 images, ni plus ni moins
    assert table.index.is_unique  # aucune image dans deux jeux
    # Les paires de quasi-doublons vues pendant l'EDA sont dans le même jeu
    for a, b in [("2286", "2787"), ("2050", "2967"), ("593", "602")]:
        assert table.loc[a, "split"] == table.loc[b, "split"]
    # Un même groupe n'est jamais coupé entre deux jeux
    assert (table.groupby("group")["split"].nunique() == 1).all()


@fichier_absent
def test_chaque_classe_est_presente_dans_chaque_jeu():
    table = pd.read_csv(SPLITS_FILE, dtype={"image_id": str}).set_index("image_id")
    objets = load_annotations()
    objets["split"] = objets["image_id"].map(table["split"])
    for nom in ["train", "val", "test"]:
        assert set(objets.loc[objets["split"] == nom, "class_name"]) == set(CLASSES)
