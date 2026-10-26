"""Baseline de classification : forêt aléatoire sur des caractéristiques HOG + taille de la boîte.

Lancer depuis la racine du projet (après scripts/make_crops.py) :
    uv run python scripts/train_baseline.py

Le modèle apprend sur le jeu d'entraînement et il est évalué sur la VALIDATION (jamais sur ses
données d'entraînement). Résultats :
- runs/baseline_rf/ : métriques, rapport par classe, matrice de confusion ;
- models/baseline_rf.joblib : le modèle entraîné.
"""

import json

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from avions.classif.baseline import baseline_features
from avions.classif.dataset import load_crops_index
from avions.classif.metrics import save_report, scores
from avions.constants import CROPS_DIR, MODELS_DIR, RUNS_DIR

NAME = "baseline_rf"


def features_of(split: str) -> tuple[np.ndarray, np.ndarray]:
    index = load_crops_index(split)
    features = [
        baseline_features(cv2.imread(str(CROPS_DIR / row.crop_path)), row.box_width, row.box_height)
        for row in tqdm(index.itertuples(), total=len(index), desc=f"Caractéristiques {split}")
    ]
    return np.stack(features), index["label"].to_numpy()


def main():
    x_train, y_train = features_of("train")
    x_val, y_val = features_of("val")

    # 300 arbres de décision, chacun appris sur un tirage au hasard des vignettes et des
    # caractéristiques ; la forêt vote. class_weight compense les classes rares.
    forest = RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=0
    )
    print("Entraînement de la forêt aléatoire...")
    forest.fit(x_train, y_train)

    results = {"train": scores(y_train, forest.predict(x_train))}
    y_pred = forest.predict(x_val)
    results["val"] = scores(y_val, y_pred)

    out_dir = RUNS_DIR / NAME
    save_report(y_val, y_pred, out_dir, "Forêt aléatoire (HOG + taille) - validation")
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Fichier compressé : 300 arbres entièrement développés pèsent environ 200 Mo sinon
    joblib.dump(forest, MODELS_DIR / f"{NAME}.joblib", compress=3)

    for split, values in results.items():
        print(f"{split:5s} : accuracy {values['accuracy']:.3f} | macro-F1 {values['macro_f1']:.3f}")
    print(f"Résultats : {out_dir}")


if __name__ == "__main__":
    main()
