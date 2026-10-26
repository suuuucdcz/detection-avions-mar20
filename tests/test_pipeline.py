"""Test de bout en bout du système complet (seulement si les modèles entraînés sont présents)."""

import cv2
import pandas as pd
import pytest

from avions.constants import CLASSES, MODELS_DIR, RAW_IMAGES_DIR, SPLITS_FILE

modeles_absents = pytest.mark.skipif(
    not (MODELS_DIR / "detecteur_1classe.pt").exists()
    or not (MODELS_DIR / "classifieur_resnet18.pt").exists(),
    reason="modèles entraînés absents (voir notebooks/colab_*.ipynb)",
)


@modeles_absents
def test_le_systeme_complet_trouve_et_classe_des_avions():
    from avions.pipeline import AircraftPipeline

    # Une image de VALIDATION (jamais vue à l'entraînement)
    split = pd.read_csv(SPLITS_FILE, dtype={"image_id": str})
    image_id = split.loc[split["split"] == "val", "image_id"].iloc[0]
    image = cv2.imread(str(RAW_IMAGES_DIR / f"{image_id}.jpg"))

    detections = AircraftPipeline().predict(image)

    assert len(detections) > 0
    for d in detections:
        assert d.class_name in CLASSES
        assert 0.5 <= d.score <= 1  # seuil par défaut de l'application
        assert d.score == pytest.approx(d.detector_score * d.classifier_score)
        xmin, ymin, xmax, ymax = d.box
        assert 0 <= xmin < xmax <= image.shape[1] and 0 <= ymin < ymax <= image.shape[0]


@modeles_absents
def test_avions_a_verifier():
    from avions.pipeline import DEFAULT_MIN_SCORE, REVIEW_MIN_DETECTOR, AircraftPipeline

    # Image 1453 (validation) : des A1 / A19 dont le type est incertain
    image = cv2.imread(str(RAW_IMAGES_DIR / "1453.jpg"))
    detections = AircraftPipeline(review_min_detector=REVIEW_MIN_DETECTOR).predict(image)

    assert any(d.to_review for d in detections)
    for d in detections:
        if d.to_review:  # pas assez sûr globalement, mais le détecteur voit bien un avion
            assert d.score < DEFAULT_MIN_SCORE and d.detector_score >= REVIEW_MIN_DETECTOR
        else:
            assert d.score >= DEFAULT_MIN_SCORE
