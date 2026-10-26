"""Entraîne un détecteur YOLO à partir d'un fichier de configuration (dossier configs/).

Lancer depuis la racine du projet :
    uv run python scripts/train_detector.py configs/detecteur_1classe.yaml

On peut remplacer un paramètre sans modifier le fichier, par exemple si la mémoire GPU sature :
    uv run python scripts/train_detector.py configs/detecteur_1classe.yaml batch=8

Si l'entraînement a été interrompu (crash, coupure de courant), on le reprend là où il s'était arrêté :
    uv run python scripts/train_detector.py configs/detecteur_1classe.yaml --resume batch=8

Résultats :
- runs/<name>/ : courbes d'apprentissage (results.png), métriques, exemples de prédictions ;
- models/<name>.pt : les poids du meilleur modèle (meilleure validation), utilisés ensuite.
"""

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from avions.constants import MODELS_DIR, RUNS_DIR, YOLO_DATASET_YAML


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="fichier YAML de configuration")
    parser.add_argument("overrides", nargs="*", help="paramètres à remplacer, ex. batch=8")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reprendre un entraînement interrompu là où il s'était arrêté",
    )
    args = parser.parse_intermixed_args()  # accepte --resume avant ou après les paramètres

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    overrides = {}
    for override in args.overrides:
        key, value = override.split("=", 1)
        overrides[key] = yaml.safe_load(value)  # "8" -> 8, "true" -> True, etc.
    config.update(overrides)

    if not YOLO_DATASET_YAML.exists():
        raise SystemExit(
            "Dataset YOLO absent : lancer d'abord  uv run python scripts/prepare_yolo.py"
        )

    if args.resume:
        # Reprise après une interruption (crash, coupure) : on repart du dernier point de
        # sauvegarde (last.pt, enregistré à chaque epoch) avec l'epoch, l'optimiseur et les
        # réglages d'origine. Seuls quelques paramètres peuvent changer (ex. batch=8).
        last = Path(config.get("project", RUNS_DIR)) / config["name"] / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"Rien à reprendre : {last} n'existe pas.")
        # Garde-fou (incident D16) : un entraînement TERMINÉ n'a plus d'optimiseur enregistré
        # (epoch = -1). Sans cette vérification, YOLO lançait un NOUVEL entraînement avec ses
        # réglages par défaut (sur coco8.yaml, 8 images hors projet) et écrasait models/.
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        if checkpoint.get("epoch", -1) < 0 or checkpoint.get("optimizer") is None:
            raise SystemExit(f"Rien à reprendre : l'entraînement {config['name']} est terminé.")
        model = YOLO(last)
        model.train(resume=True, data=str(YOLO_DATASET_YAML), **overrides)
    else:
        # Chargement du modèle pré-entraîné (téléchargé automatiquement la première fois)
        model = YOLO(config.pop("model"))
        model.train(
            data=str(YOLO_DATASET_YAML),
            project=str(config.pop("project", RUNS_DIR)),
            exist_ok=True,  # relancer avec le même nom remplace le dossier au lieu d'en créer un nouveau
            **config,
        )

    # On garde une copie des meilleurs poids dans models/ sous un nom simple
    save_dir = Path(model.trainer.save_dir)
    if save_dir.name != config["name"]:  # garde-fou : ne jamais écraser un modèle par un autre
        raise SystemExit(f"Entraînement inattendu dans {save_dir} : models/ n'est pas modifié.")
    best_weights = save_dir / "weights" / "best.pt"
    MODELS_DIR.mkdir(exist_ok=True)
    destination = MODELS_DIR / f"{config['name']}.pt"
    shutil.copy2(best_weights, destination)
    print(f"\nMeilleurs poids copiés dans : {destination}")


if __name__ == "__main__":
    main()
