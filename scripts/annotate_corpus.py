"""Annote un ensemble d'images avec le modèle final : XML Pascal VOC, aperçus et tableau.

Par défaut, le jeu de TEST, c'est-à-dire le « corpus de test annoté par le modèle » demandé pour
le rapport :
    uv run python scripts/annotate_corpus.py

N'importe quel dossier d'images :
    uv run python scripts/annotate_corpus.py --images chemin/vers/images --sortie reports/autre

Contenu du dossier de sortie :
- annotations/<image>.xml : prédictions au format Pascal VOC (même format que le dataset),
  avec la confiance (<score>) et le marquage « à vérifier » (<a_verifier>) ;
- images/ : les images d'origine ;
- apercus/ : les images avec les boîtes dessinées (épaisses : sûr ; fines « ? » : à vérifier) ;
- detections.csv : toutes les détections dans un seul tableau ;
- LISEZMOI.md : description, réglages et scores obtenus.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from avions.constants import RAW_IMAGES_DIR, REPORTS_DIR, RUNS_DIR, SPLITS_FILE
from avions.export import write_voc
from avions.pipeline import DEFAULT_MIN_SCORE, REVIEW_MIN_DETECTOR, AircraftPipeline
from avions.visualization import draw_detections


def write_readme(out_dir: Path, n_images: int, table: pd.DataFrame, test_corpus: bool) -> None:
    surs = table[table["statut"] == "sûr"]
    lines = [
        "# Corpus annoté par le modèle",
        "",
        f"{n_images} images satellites annotées automatiquement par notre système en deux étages :",
        "un détecteur YOLO11n (« où sont les avions ? »), puis un classifieur ResNet-18",
        "(« quel type, de A1 à A20 ? »).",
        "",
        "## Réglages",
        "",
        f"- **Avion sûr** : confiance finale (détecteur × classifieur) ≥ {DEFAULT_MIN_SCORE}.",
        f"- **Avion à vérifier** : le détecteur est sûr de voir un avion (≥ {REVIEW_MIN_DETECTOR}),",
        "  mais la confiance finale est plus basse, le plus souvent parce que le type est incertain.",
        "  Il revient alors à un humain de trancher.",
        "",
        "## Contenu",
        "",
        "| Dossier ou fichier | Contenu |",
        "|---|---|",
        "| `annotations/` | un fichier XML par image, au format Pascal VOC (même format que le dataset) |",
        "| `images/` | les images d'origine |",
        "| `apercus/` | les images avec les boîtes : épaisses pour les avions sûrs, fines avec « ? » pour les avions à vérifier |",
        "| `detections.csv` | toutes les détections : image, type, confiances, boîte, statut |",
        "",
        "Dans les fichiers XML, chaque `<object>` contient :",
        "- `<name>` : le type d'avion ;",
        "- `<bndbox>` : la boîte, en pixels ;",
        "- `<score>` : la confiance finale ;",
        "- `<a_verifier>` : 1 si l'avion est à vérifier, 0 sinon.",
        "",
        "## Ce que le modèle a trouvé",
        "",
        f"- **{len(surs)} avions sûrs** et **{len(table) - len(surs)} avions à vérifier**.",
        "- Types les plus fréquents parmi les avions sûrs : "
        + ", ".join(f"{t} ({n})" for t, n in surs["type"].value_counts().head(5).items())
        + ".",
    ]
    metrics_file = RUNS_DIR / "comparaison_test" / "metrics.json"
    if test_corpus and metrics_file.exists():
        m = json.loads(metrics_file.read_text(encoding="utf-8"))["deux_etages"]
        s = m["au_seuil"]
        n_avions = int(s["bon_type"] + s["mauvais_type"] + s["manques"])
        lines += [
            "",
            "## Scores sur ce corpus (comparaison avec la vérité terrain)",
            "",
            "Ces images forment le **jeu de test** : elles n'ont jamais servi ni à entraîner, ni à",
            "régler le modèle. Elles ont été évaluées une seule fois, à la fin du projet.",
            "",
            "| Mesure | Valeur |",
            "|---|---|",
            f"| Avions réels | {n_avions} |",
            f"| Avions sûrs du bon type | {int(s['bon_type'])} |",
            f"| Avions sûrs du mauvais type | {int(s['mauvais_type'])} |",
            f"| Fausses alarmes (faux positifs) | {int(s['fausses_alarmes'])} |",
            f"| Avions non détectés comme sûrs | {int(s['manques'])} |",
            (
                f"| Précision / rappel / F1 au seuil {DEFAULT_MIN_SCORE} | "
                f"{s['precision']:.3f} / {s['rappel']:.3f} / {s['f1']:.3f} |"
            ),
            f"| mAP50 (tous les seuils) | {m['mAP50']:.3f} |",
        ]
    (out_dir / "LISEZMOI.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--images", type=Path, help="dossier d'images (par défaut : le jeu de test)"
    )
    parser.add_argument("--sortie", type=Path, default=REPORTS_DIR / "corpus_test_annote")
    args = parser.parse_args()

    if args.images:
        paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    else:
        split = pd.read_csv(SPLITS_FILE, dtype={"image_id": str})
        test_ids = split.loc[split["split"] == "test", "image_id"]
        paths = [RAW_IMAGES_DIR / f"{image_id}.jpg" for image_id in sorted(test_ids)]

    shutil.rmtree(args.sortie, ignore_errors=True)  # dossier entièrement régénéré
    for folder in ("annotations", "images", "apercus"):
        (args.sortie / folder).mkdir(parents=True)

    pipeline = AircraftPipeline(review_min_detector=REVIEW_MIN_DETECTOR)
    rows = []
    for path in tqdm(paths, desc="Annotation"):
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        detections = sorted(pipeline.predict(image), key=lambda d: (d.to_review, -d.score))

        write_voc(
            args.sortie / "annotations" / f"{path.stem}.xml", path.name, width, height, detections
        )
        shutil.copy2(path, args.sortie / "images" / path.name)
        preview = draw_detections(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections)
        cv2.imwrite(
            str(args.sortie / "apercus" / f"{path.stem}.jpg"),
            cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
        )
        for d in detections:
            xmin, ymin, xmax, ymax = (round(v) for v in d.box)
            rows.append(
                {
                    "image": path.name,
                    "type": d.class_name,
                    "confiance": round(d.score, 4),
                    "confiance_detecteur": round(d.detector_score, 4),
                    "confiance_classifieur": round(d.classifier_score, 4),
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "statut": "à vérifier" if d.to_review else "sûr",
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(args.sortie / "detections.csv", index=False)
    write_readme(args.sortie, len(paths), table, test_corpus=args.images is None)
    print(f"{len(paths)} images annotées dans {args.sortie}")
    print(table["statut"].value_counts().to_string())


if __name__ == "__main__":
    main()
