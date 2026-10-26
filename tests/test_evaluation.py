"""Tests du code d'évaluation (IoU, appariement des prédictions, mAP)."""

import numpy as np
import pytest

from avions.evaluation import (
    DetectionEvaluator,
    box_iou,
    count_outcomes,
    match_details,
    match_predictions,
)


def test_iou():
    a = np.array([[0, 0, 10, 10]])
    assert box_iou(a, np.array([[0, 0, 10, 10]]))[0, 0] == pytest.approx(1.0)  # identiques
    assert box_iou(a, np.array([[20, 20, 30, 30]]))[0, 0] == 0  # disjointes
    # Moitié recouverte : intersection 50, union 150 -> 1/3
    assert box_iou(a, np.array([[5, 0, 15, 10]]))[0, 0] == pytest.approx(1 / 3)


def test_une_seule_prediction_validee_par_avion():
    true_boxes, true_classes = np.array([[0, 0, 10, 10]]), np.array([3])
    pred_boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10]])  # deux boîtes sur le même avion
    correct = match_predictions(true_boxes, true_classes, pred_boxes, np.array([3, 3]))
    assert correct[:, 0].tolist() == [True, False]  # la mieux placée seulement (IoU 0,5)


def test_mauvaise_classe_jamais_validee():
    correct = match_predictions(
        np.array([[0, 0, 10, 10]]), np.array([3]), np.array([[0, 0, 10, 10]]), np.array([4])
    )
    assert not correct.any()


def test_predictions_parfaites_donnent_une_map_maximale():
    evaluator = DetectionEvaluator()
    for i in range(5):
        boxes = np.array([[0, 0, 10, 10], [50, 50, 70, 60]]) + i
        classes = np.array([0, 1])
        evaluator.add_image(boxes, classes, boxes, classes, [0.9, 0.8])
    results = evaluator.results()
    # Ultralytics intègre la courbe sur 101 points : une détection parfaite donne environ 0,995
    assert results["mAP50"] > 0.99
    assert results["mAP50-95"] > 0.99


def test_decompte_des_faux_positifs():
    true_boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [100, 100, 110, 110]])
    true_classes = np.array([1, 2, 3])
    pred_boxes = np.array(
        [
            [0, 0, 10, 10],  # bon type
            [50, 50, 60, 60],  # mauvais type
            [200, 200, 210, 210],  # rien à cet endroit : fausse alarme
            [100, 100, 110, 110],  # trop peu sûr : ignoré au seuil 0,5 -> avion manqué
        ]
    )
    counts = count_outcomes(
        true_boxes, true_classes, pred_boxes, np.array([1, 5, 1, 3]), [0.9, 0.8, 0.7, 0.3], 0.5
    )
    assert counts == {"bon_type": 1, "mauvais_type": 1, "fausses_alarmes": 1, "manques": 1}


def test_verdict_de_chaque_boite():
    true_boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [100, 100, 110, 110]])
    pred_boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [200, 200, 210, 210]])
    statuses, matched, missed = match_details(
        true_boxes, np.array([1, 2, 3]), pred_boxes, np.array([1, 5, 1]), [0.9, 0.8, 0.7]
    )
    assert statuses == ["bon_type", "mauvais_type", "fausse_alarme"]
    assert matched == [0, 1, None]  # avion réel associé à chaque prédiction
    assert missed == [2]  # le 3e avion n'est recouvert par aucune prédiction
