"""Compare nos deux systèmes complets sur le même jeu d'images, avec le même code d'évaluation :

- « deux étages » : détecteur « avion » puis classifieur ResNet-18 (notre architecture, D1) ;
- « témoin » : YOLO à 20 classes, qui détecte et classe en un seul passage.

Lancer depuis la racine du projet :
    uv run python scripts/evaluate_systems.py                 (validation)

Le jeu de TEST ne doit servir qu'une seule fois, à la toute fin du projet. Il faut donc le
demander explicitement :
    uv run python scripts/evaluate_systems.py --split test --je-confirme-le-test

Deux façons de mesurer, toutes deux enregistrées :
- la mAP, sur toute la courbe précision-rappel (tous les seuils de confiance) ;
- le décompte AU SEUIL DE L'APPLICATION (0,5) : avions bien classés, mal classés, FAUSSES ALARMES
  (faux positifs) et avions manqués, ainsi que le temps de calcul par image.

Résultats dans runs/comparaison_<jeu>/. Tout tourne sur le processeur (CPU).
"""

import argparse
import json
import time
from collections import Counter

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from avions.constants import CLASS_TO_INDEX, MODELS_DIR, RAW_IMAGES_DIR, RUNS_DIR, SPLITS_FILE
from avions.data.voc import load_annotations
from avions.evaluation import DetectionEvaluator, count_outcomes
from avions.pipeline import DEFAULT_MIN_SCORE, AircraftPipeline, predict_baseline_yolo

# Confiance minimale très basse pour l'évaluation : la mAP a besoin de toute la courbe
# précision-rappel, y compris les prédictions peu sûres (l'application, elle, utilise 0,5).
EVAL_MIN_CONFIDENCE = 0.01


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--je-confirme-le-test", action="store_true", dest="confirm_test")
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_test:
        raise SystemExit(
            "Le jeu de test ne sert qu'une fois, à la fin : ajouter --je-confirme-le-test."
        )

    objects = load_annotations()
    split = pd.read_csv(SPLITS_FILE, dtype={"image_id": str}).set_index("image_id")["split"]
    objects = objects[objects["image_id"].map(split) == args.split]

    two_stage = AircraftPipeline(min_score=EVAL_MIN_CONFIDENCE)
    two_stage_app = AircraftPipeline(min_score=DEFAULT_MIN_SCORE)  # réglage de l'application
    baseline = YOLO(str(MODELS_DIR / "detecteur_20classes.pt"))
    evaluators = {"deux_etages": DetectionEvaluator(), "temoin_yolo20": DetectionEvaluator()}
    seconds = {name: 0.0 for name in evaluators}
    outcomes = {name: Counter() for name in evaluators}

    for image_id, truth in tqdm(objects.groupby("image_id"), desc=f"Images ({args.split})"):
        image = cv2.imread(str(RAW_IMAGES_DIR / f"{image_id}.jpg"))
        true_boxes = truth[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
        true_classes = truth["class_name"].map(CLASS_TO_INDEX).to_numpy()

        # Notre système : mAP sur toutes les prédictions, décompte au seuil de l'application...
        detections = two_stage.predict(image)
        boxes = [d.box for d in detections]
        classes = [CLASS_TO_INDEX[d.class_name] for d in detections]
        scores = [d.score for d in detections]
        evaluators["deux_etages"].add_image(true_boxes, true_classes, boxes, classes, scores)
        outcomes["deux_etages"].update(
            count_outcomes(true_boxes, true_classes, boxes, classes, scores, DEFAULT_MIN_SCORE)
        )
        # ... et temps de calcul avec le réglage de l'application
        start = time.perf_counter()
        two_stage_app.predict(image)
        seconds["deux_etages"] += time.perf_counter() - start

        # Le témoin, mesuré exactement de la même façon
        boxes, classes, scores = predict_baseline_yolo(
            baseline, image, EVAL_MIN_CONFIDENCE, device="cpu"
        )
        evaluators["temoin_yolo20"].add_image(true_boxes, true_classes, boxes, classes, scores)
        outcomes["temoin_yolo20"].update(
            count_outcomes(true_boxes, true_classes, boxes, classes, scores, DEFAULT_MIN_SCORE)
        )
        start = time.perf_counter()
        predict_baseline_yolo(baseline, image, DEFAULT_MIN_SCORE, device="cpu")
        seconds["temoin_yolo20"] += time.perf_counter() - start

    n_images = objects["image_id"].nunique()
    results = {}
    for name, evaluator in evaluators.items():
        results[name] = evaluator.results()
        results[name]["ms_par_image_cpu"] = 1000 * seconds[name] / n_images
        c = outcomes[name]
        proposed = c["bon_type"] + c["mauvais_type"] + c["fausses_alarmes"]
        precision, recall = c["bon_type"] / max(proposed, 1), c["bon_type"] / len(objects)
        results[name]["au_seuil"] = {
            "seuil": DEFAULT_MIN_SCORE,
            **dict(c),
            "precision": precision,
            "rappel": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        }

    out_dir = RUNS_DIR / f"comparaison_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    per_class = pd.DataFrame({name: r["ap50_par_classe"] for name, r in results.items()})
    per_class["ecart"] = per_class["deux_etages"] - per_class["temoin_yolo20"]
    per_class.round(3).to_csv(out_dir / "ap50_par_classe.csv", index_label="classe")

    columns = ["mAP50", "mAP50-95", "precision", "recall", "f1", "ms_par_image_cpu"]
    summary = pd.DataFrame(results).T[columns].astype(float)
    print(f"\n{n_images} images, {len(objects)} avions ({args.split}) :")
    print(summary.round(3).to_string())
    at_threshold = pd.DataFrame({name: r["au_seuil"] for name, r in results.items()}).T
    print(f"\nAu seuil de l'application ({DEFAULT_MIN_SCORE}) :")
    print(at_threshold.drop(columns="seuil").round(3).to_string())
    print("\nAP50 par classe (les plus grands écarts) :")
    print(per_class.reindex(per_class["ecart"].abs().sort_values(ascending=False).index).head(6))
    print(f"\nRésultats : {out_dir}")


if __name__ == "__main__":
    main()
