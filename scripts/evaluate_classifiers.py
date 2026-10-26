"""Évalue tous les classifieurs de vignettes sur un même jeu (validation par défaut).

Lancer depuis la racine du projet (après scripts/make_crops.py) :
    uv run python scripts/evaluate_classifiers.py

Le jeu de TEST ne sert qu'une seule fois, à la toute fin du projet :
    uv run python scripts/evaluate_classifiers.py --split test --je-confirme-le-test

Résultats dans runs/classifieurs_<jeu>/ : un tableau des scores, et pour chaque modèle son
rapport par classe et sa matrice de confusion. Tout tourne sur le processeur (CPU).
"""

import argparse
import json

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from avions.classif.baseline import baseline_features
from avions.classif.dataset import CropDataset, load_crops_index
from avions.classif.metrics import save_report, scores
from avions.classif.transforms import eval_transform
from avions.constants import CROPS_DIR, MODELS_DIR, RUNS_DIR
from avions.pipeline import load_classifier

NETWORKS = ["classifieur_resnet18", "classifieur_resnet18_sans_pretrain", "classifieur_smallcnn"]


@torch.no_grad()
def predict_network(name: str, index: pd.DataFrame) -> np.ndarray:
    model = load_classifier(MODELS_DIR / f"{name}.pt", device="cpu")
    loader = DataLoader(CropDataset(index, eval_transform()), batch_size=64)
    return torch.cat([model(images).argmax(dim=1) for images, _ in loader]).numpy()


def predict_forest(index: pd.DataFrame) -> np.ndarray:
    forest = joblib.load(MODELS_DIR / "baseline_rf.joblib")
    features = [
        baseline_features(cv2.imread(str(CROPS_DIR / row.crop_path)), row.box_width, row.box_height)
        for row in index.itertuples()
    ]
    return forest.predict(np.stack(features))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--je-confirme-le-test", action="store_true", dest="confirm_test")
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_test:
        raise SystemExit(
            "Le jeu de test ne sert qu'une fois, à la fin : ajouter --je-confirme-le-test."
        )

    index = load_crops_index(args.split)
    y_true = index["label"].to_numpy()
    out_dir = RUNS_DIR / f"classifieurs_{args.split}"
    results = {}
    for name in ["baseline_rf", *NETWORKS]:
        print(f"Évaluation de {name}...")
        y_pred = predict_forest(index) if name == "baseline_rf" else predict_network(name, index)
        results[name] = scores(y_true, y_pred)
        results[name]["erreurs"] = int((y_pred != y_true).sum())
        save_report(y_true, y_pred, out_dir / name, f"{name} - {args.split}")

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n{len(index)} vignettes ({args.split}) :")
    print(pd.DataFrame(results).T.round(3).to_string())
    print(f"\nRésultats : {out_dir}")


if __name__ == "__main__":
    main()
