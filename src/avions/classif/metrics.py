"""Mesures de performance d'un classifieur et leur enregistrement (rapport, matrice de confusion).

- accuracy : part des vignettes bien classées ;
- macro-F1 : moyenne des F1 de chaque classe -> chaque classe compte autant, même les rares
  (A6, A11, A18...). C'est notre métrique principale (EDA, question 1).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures enregistrées dans des fichiers, sans fenêtre (Colab, serveurs)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

from avions.constants import CLASSES

LABELS = list(range(len(CLASSES)))


def scores(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
    }


def save_report(y_true, y_pred, out_dir: Path, title: str) -> None:
    """Enregistre le rapport par classe (précision, rappel, F1) et la matrice de confusion."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        y_true, y_pred, labels=LABELS, target_names=CLASSES, digits=3, zero_division=0
    )
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    # Matrice normalisée par ligne : pour chaque vraie classe, la part prédite dans chaque classe
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=LABELS,
        display_labels=CLASSES,
        normalize="true",
        values_format=".2f",
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    for text in ax.texts:
        text.set_fontsize(6)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Vraie classe")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
