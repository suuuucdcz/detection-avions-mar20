"""Entraîne un classifieur de vignettes (réseau de neurones) à partir d'un fichier de configuration.

Lancer depuis la racine du projet (après scripts/make_crops.py) :
    uv run python scripts/train_classifier.py configs/classifieur_resnet18.yaml
On peut remplacer un paramètre sans modifier le fichier :  ... epochs=2 device=cpu

À chaque epoch :
1. on parcourt toutes les vignettes d'entraînement par paquets (batch) : le réseau prédit,
   on mesure l'erreur (la loss), on calcule dans quel sens corriger chaque poids (le gradient)
   et on corrige un peu (DESCENTE DE GRADIENT) ;
2. on mesure la macro-F1 sur la VALIDATION. On garde le modèle de la meilleure epoch, et on
   s'arrête si la validation ne progresse plus pendant `patience` epochs (EARLY STOPPING).

Résultats :
- runs/<name>/ : results.csv et results.png (courbes), rapport par classe, matrice de confusion ;
- models/<name>.pt : les poids du meilleur modèle.
"""

import argparse
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from avions.classif.dataset import CropDataset, load_crops_index
from avions.classif.metrics import save_report, scores
from avions.classif.models import build_model, head_parameters
from avions.classif.transforms import eval_transform, train_transform
from avions.constants import CLASSES, MODELS_DIR, RUNS_DIR


def load_config() -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="fichier YAML de configuration")
    parser.add_argument("overrides", nargs="*", help="paramètres à remplacer, ex. epochs=2")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for override in args.overrides:
        key, value = override.split("=", 1)
        config[key] = yaml.safe_load(value)
    return config


def class_weights(labels: list[int]) -> torch.Tensor:
    """Poids 1 / racine(effectif) : une erreur sur une classe rare compte davantage."""
    counts = np.bincount(labels, minlength=len(CLASSES))
    weights = 1 / np.sqrt(counts)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Gèle (trainable=False) ou dégèle tout le réseau sauf sa dernière couche."""
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in head_parameters(model):
        parameter.requires_grad = True


def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> float:
    model.train()
    total_loss, total = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = criterion(model(images), labels)  # 1. prédiction et erreur
        scaler.scale(loss).backward()  # 2. gradient de l'erreur par rapport à chaque poids
        scaler.step(optimizer)  # 3. correction des poids
        scaler.update()
        total_loss += loss.item() * len(labels)
        total += len(labels)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, np.ndarray, np.ndarray]:
    """Loss moyenne, vraies classes et classes prédites sur un jeu (sans apprendre)."""
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * len(labels)
        y_true.append(labels.cpu())
        y_pred.append(logits.argmax(dim=1).cpu())
    y_true, y_pred = torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()
    return total_loss / len(y_true), y_true, y_pred


def plot_curves(history: pd.DataFrame, out_path: Path) -> None:
    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss.plot(history["epoch"], history["train_loss"], label="entraînement")
    ax_loss.plot(history["epoch"], history["val_loss"], label="validation")
    ax_loss.set_title("Loss (erreur)")
    ax_score.plot(history["epoch"], history["val_macro_f1"], label="macro-F1")
    ax_score.plot(history["epoch"], history["val_accuracy"], label="accuracy")
    ax_score.set_title("Scores sur la validation")
    for ax in (ax_loss, ax_score):
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    config = load_config()
    name = config["name"]
    seed = config.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_name = config.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Entraînement de {name} sur {device}")

    train_index = load_crops_index("train")
    if config.get("fraction", 1.0) < 1.0:  # pour les essais rapides seulement
        train_index = train_index.sample(frac=config["fraction"], random_state=seed)
    val_index = load_crops_index("val")
    workers = config.get("workers", 2)
    train_loader = DataLoader(
        CropDataset(train_index, train_transform()),
        batch_size=config["batch"],
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        CropDataset(val_index, eval_transform()), batch_size=config["batch"], num_workers=workers
    )

    model = build_model(config["architecture"], config["pretrained"]).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_index["label"].tolist()).to(device),
        label_smoothing=config.get("label_smoothing", 0.0),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    out_dir = RUNS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, history = -1.0, 0, []
    freeze_epochs = config.get("freeze_epochs", 0)
    start = time.time()

    for epoch in range(1, config["epochs"] + 1):
        # Transfer learning : on n'entraîne d'abord que la nouvelle tête (backbone gelé),
        # puis tout le réseau, pour ne pas abîmer d'emblée ce que le backbone sait déjà
        set_backbone_trainable(model, trainable=epoch > freeze_epochs)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        val_scores = scores(y_true, y_pred)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_scores["accuracy"],
                "val_macro_f1": val_scores["macro_f1"],
                "time": time.time() - start,
            }
        )
        print(
            f"epoch {epoch:3d} | loss entraînement {train_loss:.3f}"
            f" | loss validation {val_loss:.3f}"
            f" | accuracy {val_scores['accuracy']:.3f} | macro-F1 {val_scores['macro_f1']:.3f}"
        )
        pd.DataFrame(history).to_csv(out_dir / "results.csv", index=False)

        if val_scores["macro_f1"] > best_f1:  # meilleur modèle jusqu'ici : on le garde
            best_f1, best_epoch = val_scores["macro_f1"], epoch
            torch.save(
                {"state_dict": model.state_dict(), "classes": CLASSES, "config": config},
                MODELS_DIR / f"{name}.pt",
            )
            save_report(y_true, y_pred, out_dir, f"{name} - validation (epoch {epoch})")
        elif epoch - best_epoch >= config.get("patience", config["epochs"]):
            print(f"Early stopping : pas de progrès depuis {config['patience']} epochs.")
            break

    plot_curves(pd.DataFrame(history), out_dir / "results.png")
    print(f"\nMeilleure epoch : {best_epoch} | macro-F1 en validation : {best_f1:.3f}")
    print(f"Poids : {MODELS_DIR / f'{name}.pt'} | courbes et rapport : {out_dir}")


if __name__ == "__main__":
    main()
