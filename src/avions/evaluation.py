"""Évaluation d'un système de détection + classification : mAP, précision, rappel, F1.

On évalue nos deux systèmes (deux étages, et témoin YOLO à 20 classes) avec CE MÊME code,
pour une comparaison équitable (décision D13).

Principe :
1. IoU (« intersection sur union ») : à quel point une boîte prédite recouvre une vraie boîte
   (0 = pas du tout, 1 = parfaitement) ;
2. une prédiction est un VRAI POSITIF si elle a la bonne classe et recouvre une vraie boîte avec
   une IoU suffisante, et que cette vraie boîte n'a pas déjà été « prise » par une meilleure
   prédiction. Sinon c'est un FAUX POSITIF. Une vraie boîte jamais trouvée est un FAUX NÉGATIF ;
3. en classant les prédictions de la plus sûre à la moins sûre, on trace la courbe
   précision-rappel de chaque classe. Son aire est l'AP (average precision) ; la moyenne sur
   les 20 classes est la mAP. mAP50 : seuil d'IoU de 0,5 ; mAP50-95 : moyenne pour des seuils
   de 0,5 à 0,95 (plus exigeant sur la justesse des boîtes).

Le calcul de l'aire des courbes réutilise la fonction d'Ultralytics (ap_per_class), celle-là
même qui donne les scores affichés pendant l'entraînement de YOLO.
"""

import numpy as np
from ultralytics.utils.metrics import ap_per_class

from avions.constants import CLASSES

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)  # 0,50 ; 0,55 ; ... ; 0,95


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU de chaque boîte de A avec chaque boîte de B (boîtes au format xmin, ymin, xmax, ymax)."""
    a, b = boxes_a[:, None, :], boxes_b[None, :, :]
    inter_w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    inter_h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    intersection = inter_w * inter_h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return intersection / (area_a + area_b - intersection + 1e-9)


def match_predictions(
    true_boxes: np.ndarray,
    true_classes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
) -> np.ndarray:
    """Pour chaque prédiction et chaque seuil d'IoU : True si c'est un vrai positif.

    Chaque vraie boîte ne peut valider qu'UNE prédiction (la mieux superposée) :
    deux boîtes sur le même avion = un vrai positif + un faux positif.
    """
    correct = np.zeros((len(pred_boxes), len(IOU_THRESHOLDS)), dtype=bool)
    if len(true_boxes) == 0 or len(pred_boxes) == 0:
        return correct
    iou = box_iou(true_boxes, pred_boxes)  # lignes : vraies boîtes, colonnes : prédictions
    iou = iou * (true_classes[:, None] == pred_classes[None, :])  # mauvaise classe -> IoU 0
    for t, threshold in enumerate(IOU_THRESHOLDS):
        pairs = np.argwhere(iou >= threshold)  # couples (vraie boîte, prédiction) candidats
        if len(pairs) == 0:
            continue
        pairs = pairs[np.argsort(-iou[pairs[:, 0], pairs[:, 1]])]  # meilleures IoU d'abord
        pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]  # 1 vraie boîte par prédiction
        pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]  # 1 prédiction par vraie boîte
        correct[pairs[:, 1], t] = True
    return correct


def count_outcomes(
    true_boxes, true_classes, pred_boxes, pred_classes, pred_scores, min_score: float
) -> dict:
    """Décompte lisible à un seuil de confiance donné (celui de l'application), voir match_details."""
    pred_scores = np.asarray(pred_scores, float)
    keep = pred_scores >= min_score
    statuses, _, missed = match_details(
        true_boxes,
        true_classes,
        np.asarray(pred_boxes, float).reshape(-1, 4)[keep],
        np.asarray(pred_classes)[keep],
        pred_scores[keep],
    )
    return {
        "bon_type": statuses.count("bon_type"),
        "mauvais_type": statuses.count("mauvais_type"),
        "fausses_alarmes": statuses.count("fausse_alarme"),
        "manques": len(missed),
    }


def match_details(true_boxes, true_classes, pred_boxes, pred_classes, pred_scores):
    """Le verdict de CHAQUE prédiction face à la vérité (utilisé par l'interface « Vérifier »).

    Les prédictions sont prises de la plus sûre à la moins sûre ; chacune est associée à l'avion
    encore libre qu'elle recouvre le mieux (IoU >= 0,5) :
    - "bon_type" / "mauvais_type" selon la classe prédite ;
    - "fausse_alarme" (faux positif) si elle ne recouvre aucun avion libre.
    Un avion qu'aucune prédiction ne recouvre est MANQUÉ (faux négatif).
    Un mauvais type compte à la fois comme faux positif et comme faux négatif.

    Renvoie : le statut de chaque prédiction, l'indice de l'avion réel associé (ou None), et la
    liste des indices des avions manqués.
    """
    true_boxes = np.asarray(true_boxes, float).reshape(-1, 4)
    pred_boxes = np.asarray(pred_boxes, float).reshape(-1, 4)
    true_classes, pred_classes = np.asarray(true_classes), np.asarray(pred_classes)
    pred_scores = np.asarray(pred_scores, float)
    iou = box_iou(pred_boxes, true_boxes) if len(true_boxes) else np.zeros((len(pred_boxes), 0))
    statuses, matched = ["fausse_alarme"] * len(pred_boxes), [None] * len(pred_boxes)
    taken = set()
    for p in np.argsort(-pred_scores):
        free = [t for t in np.argsort(-iou[p]) if iou[p, t] >= 0.5 and t not in taken]
        if free:
            taken.add(free[0])
            matched[p] = int(free[0])
            same_type = true_classes[free[0]] == pred_classes[p]
            statuses[p] = "bon_type" if same_type else "mauvais_type"
    missed = [t for t in range(len(true_boxes)) if t not in taken]
    return statuses, matched, missed


class DetectionEvaluator:
    """Accumule les prédictions image par image, puis calcule les scores sur tout le jeu."""

    def __init__(self):
        self.correct, self.scores, self.pred_classes, self.true_classes = [], [], [], []

    def add_image(self, true_boxes, true_classes, pred_boxes, pred_classes, pred_scores) -> None:
        true_boxes, pred_boxes = np.asarray(true_boxes, float), np.asarray(pred_boxes, float)
        true_classes, pred_classes = np.asarray(true_classes, int), np.asarray(pred_classes, int)
        self.correct.append(
            match_predictions(
                true_boxes.reshape(-1, 4), true_classes, pred_boxes.reshape(-1, 4), pred_classes
            )
        )
        self.scores.append(np.asarray(pred_scores, float))
        self.pred_classes.append(pred_classes)
        self.true_classes.append(true_classes)

    def results(self) -> dict:
        # p, r, f1 : par classe, au seuil de confiance qui maximise la F1 moyenne
        # ap : par classe et par seuil d'IoU (colonne 0 : IoU 0,5)
        _, _, p, r, f1, ap, classes, *_ = ap_per_class(
            np.concatenate(self.correct),
            np.concatenate(self.scores),
            np.concatenate(self.pred_classes),
            np.concatenate(self.true_classes),
        )
        return {
            "mAP50": float(ap[:, 0].mean()),
            "mAP50-95": float(ap.mean()),
            "precision": float(p.mean()),
            "recall": float(r.mean()),
            "f1": float(f1.mean()),
            "ap50_par_classe": {CLASSES[c]: float(ap[i, 0]) for i, c in enumerate(classes)},
        }
