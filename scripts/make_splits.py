"""Découpe les images en train / validation / test (80 / 10 / 10) et enregistre data/splits/splits.csv.

Lancer depuis la racine du projet :  uv run python scripts/make_splits.py

Le fichier produit est versionné dans Git : tout le monde (le binôme, les scripts
d'entraînement, l'évaluation finale) utilise exactement le même découpage.
"""

from avions.constants import SPLITS_FILE
from avions.data.duplicates import near_duplicate_groups
from avions.data.split import SPLIT_NAMES, best_split, split_report
from avions.data.voc import load_annotations


def main():
    objects = load_annotations()
    image_ids = sorted(objects["image_id"].unique())

    print("Recherche des quasi-doublons...")
    groups = near_duplicate_groups(image_ids)
    group_sizes = groups.value_counts()
    print(
        f"  {(group_sizes > 1).sum()} groupes de quasi-doublons (au plus {group_sizes.max()} images)"
    )

    print("Découpage (50 graines essayées, on garde la meilleure)...")
    split, seed, score = best_split(objects, groups)
    print(
        f"  graine retenue : {seed} | plus grand écart à 10 % pour une classe : {100 * score:.1f} points"
    )

    SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    table = split.to_frame().join(groups)
    table.index.name = "image_id"
    table.to_csv(SPLITS_FILE)
    print(f"  écrit : {SPLITS_FILE}")

    n_objects, n_images = split_report(objects, split)
    print("\nImages et avions par jeu :")
    for name in SPLIT_NAMES:
        print(
            f"  {name:5s} : {(split == name).sum():4d} images ({100 * (split == name).mean():.0f} %)"
            f" | {n_objects[name].sum():4d} avions"
        )
    print("\nAvions par classe et par jeu (entre parenthèses : nombre d'images) :")
    for class_name in n_objects.index:
        cells = [
            f"{n_objects.loc[class_name, s]:4d} ({n_images.loc[class_name, s]:3d})"
            for s in SPLIT_NAMES
        ]
        print(f"  {class_name:4s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
