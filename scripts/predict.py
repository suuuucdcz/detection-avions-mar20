"""Démo : détecte et classe les avions d'une image satellite, et enregistre l'image annotée.

Lancer depuis la racine du projet :
    uv run python scripts/predict.py data/raw/images/1453.jpg

Résultat : l'image annotée dans reports/figures/predictions/, et la liste des avions trouvés.
"""

import argparse
from pathlib import Path

import cv2

from avions.constants import FIGURES_DIR
from avions.pipeline import DEFAULT_MIN_SCORE, AircraftPipeline
from avions.visualization import draw_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path, help="image satellite à analyser")
    parser.add_argument(
        "--seuil", type=float, default=DEFAULT_MIN_SCORE, help="confiance finale minimale"
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Image illisible : {args.image}")
    detections = AircraftPipeline(min_score=args.seuil).predict(image)

    print(f"{len(detections)} avion(s) trouvé(s) dans {args.image.name} :")
    for d in sorted(detections, key=lambda d: -d.score):
        xmin, ymin, xmax, ymax = (round(v) for v in d.box)
        print(
            f"  {d.class_name:4s} confiance {d.score:.2f}"
            f" (détecteur {d.detector_score:.2f} x classifieur {d.classifier_score:.2f})"
            f" | boîte ({xmin}, {ymin}) - ({xmax}, {ymax})"
        )

    # draw_boxes travaille en RGB : on convertit, puis on revient en BGR pour enregistrer
    boxes = [tuple(round(v) for v in d.box) for d in detections]
    labels = [f"{d.class_name} {d.score:.2f}" for d in detections]
    annotated = draw_boxes(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), boxes, labels)
    out_path = FIGURES_DIR / "predictions" / f"{args.image.stem}_annote.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print(f"Image annotée : {out_path}")


if __name__ == "__main__":
    main()
