"""Découpage des images en jeux d'entraînement (train), de validation (val) et de test.

Trois règles (voir docs/journal_decisions.md, D7) :
1. on découpe PAR IMAGE (tous les avions d'une image vont dans le même jeu) ;
2. on garde les quasi-doublons ensemble (même groupe -> même jeu) ;
3. on garde à peu près les mêmes proportions de classes dans chaque jeu (stratification).

Méthode : StratifiedGroupKFold (scikit-learn) coupe les données en N paquets (« folds »)
de tailles proches, sans jamais couper un groupe, en équilibrant les classes. Avec 10 paquets :
paquet 0 -> test (10 %), paquet 1 -> validation (10 %), les 8 autres -> train (80 %).
Le résultat dépend du tirage au hasard (la « graine ») : on essaie plusieurs graines et on garde
le découpage où chaque classe est la mieux répartie.
"""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from avions.constants import CLASSES

SPLIT_NAMES = ["train", "val", "test"]
N_FOLDS = 10  # 10 paquets -> 80 / 10 / 10
MIN_IMAGES_PER_CLASS = 3  # chaque classe doit avoir au moins 3 images en validation et en test


def split_images(objects: pd.DataFrame, groups: pd.Series, seed: int = 0) -> pd.Series:
    """Renvoie le jeu ("train", "val" ou "test") de chaque image (index = image_id).

    objects : une ligne par avion, avec les colonnes image_id et class_name.
    groups : numéro de groupe de chaque image (index = image_id).
    """
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    # Une ligne par avion : la classe sert à équilibrer, le groupe de son image empêche de couper
    folds = splitter.split(objects, objects["class_name"], objects["image_id"].map(groups))
    split = {}
    for fold, (_, indices) in enumerate(folds):
        name = "test" if fold == 0 else "val" if fold == 1 else "train"
        for image_id in objects["image_id"].iloc[indices]:
            split[image_id] = name
    return pd.Series(split, name="split").sort_index()


def split_report(objects: pd.DataFrame, split: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deux tableaux (une ligne par classe, une colonne par jeu) : nombre d'avions et d'images."""
    with_split = objects.assign(split=objects["image_id"].map(split))
    n_objects = pd.crosstab(with_split["class_name"], with_split["split"])
    n_images = with_split.groupby(["class_name", "split"])["image_id"].nunique().unstack()
    shape = {"index": CLASSES, "columns": SPLIT_NAMES, "fill_value": 0}
    return n_objects.reindex(**shape), n_images.reindex(**shape)


def split_score(objects: pd.DataFrame, split: pd.Series) -> float:
    """Plus le score est petit, mieux c'est.

    Score = le plus grand écart, toutes classes confondues, entre la part des avions d'une classe
    en validation (ou en test) et la part visée (10 %). On ajoute une grosse pénalité si une classe
    a moins de MIN_IMAGES_PER_CLASS images en validation ou en test.
    """
    n_objects, n_images = split_report(objects, split)
    shares = n_objects.div(n_objects.sum(axis=1), axis=0)  # part de chaque jeu, classe par classe
    gap = (shares[["val", "test"]] - 1 / N_FOLDS).abs().to_numpy().max()
    too_few_images = (n_images[["val", "test"]] < MIN_IMAGES_PER_CLASS).to_numpy().any()
    return gap + (1.0 if too_few_images else 0.0)


def best_split(
    objects: pd.DataFrame, groups: pd.Series, seeds: range = range(50)
) -> tuple[pd.Series, int, float]:
    """Essaie plusieurs graines et renvoie (meilleur découpage, sa graine, son score)."""
    candidates = ((split_images(objects, groups, seed), seed) for seed in seeds)
    scored = ((split, seed, split_score(objects, split)) for split, seed in candidates)
    return min(scored, key=lambda result: result[2])
