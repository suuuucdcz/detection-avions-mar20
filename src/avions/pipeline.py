"""Notre système complet, en deux étages (décision D1) :

    image  ->  1. détecteur « avion » (YOLO11n)  ->  boîtes
           ->  vignette autour de chaque boîte (make_crop, la même qu'à l'entraînement)
           ->  2. classifieur (ResNet-18)  ->  type d'avion A1…A20

Confiance finale d'un avion = confiance du détecteur × confiance du classifieur :
un avion n'est sûr que si on est sûr à la fois que c'est un avion ET de son type.

Par défaut, tout tourne sur le processeur (CPU) : c'est suffisant pour l'application
(moins d'une demi-seconde par image) et la carte graphique du PC est instable (D10).
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from avions.classif.models import build_model
from avions.classif.transforms import eval_transform
from avions.constants import CLASSES, MODELS_DIR
from avions.data.crops import make_crop

DETECTOR_IMGSZ = 800  # même résolution qu'à l'entraînement du détecteur
NMS_IOU = 0.7  # deux boîtes qui se recouvrent à plus de 70 % : on ne garde que la plus sûre
# Seuil sur la confiance FINALE, choisi sur la validation (D14) : meilleure F1 (0,985) avec le
# moins de faux positifs (4 fausses alarmes et 1 mauvais type pour 787 avions)
DEFAULT_MIN_SCORE = 0.5
# Avions « à vérifier » (humain dans la boucle, D14) : le détecteur est sûr que c'est un avion
# (>= 0,6), mais la confiance finale reste sous le seuil, le plus souvent parce que le type est
# incertain. On les montre à part au lieu de les cacher.
REVIEW_MIN_DETECTOR = 0.6


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax en pixels
    class_name: str
    score: float  # confiance finale = détecteur × classifieur
    detector_score: float
    classifier_score: float
    to_review: bool = False  # True : à faire vérifier par un humain


def load_classifier(path: Path, device: str) -> torch.nn.Module:
    """Recrée le réseau décrit dans le fichier de poids, puis charge ses poids entraînés."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = build_model(config["architecture"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


class AircraftPipeline:
    """Détecte et classe les avions d'une image satellite."""

    def __init__(
        self,
        detector_path: Path = MODELS_DIR / "detecteur_1classe.pt",
        classifier_path: Path = MODELS_DIR / "classifieur_resnet18.pt",
        min_score: float = DEFAULT_MIN_SCORE,
        review_min_detector: float | None = None,
        device: str = "cpu",
    ):
        self.detector = YOLO(str(detector_path))
        self.classifier = load_classifier(classifier_path, device)
        self.transform = eval_transform()
        self.min_score = min_score  # confiance finale minimale pour un avion « sûr »
        # None : pas d'avions « à vérifier » (évaluation automatique) ; sinon seuil du détecteur
        self.review_min_detector = review_min_detector
        self.device = device

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        """Les avions d'une image (lue avec cv2.imread, donc en BGR).

        - confiance finale >= min_score : avion « sûr » ;
        - sinon, si review_min_detector est défini et que le détecteur est assez sûr : avion
          « à vérifier » (to_review=True) ;
        - sinon : écarté. Le classifieur sert ainsi de second filtre : une boîte sur le fond ou
          sur un bout d'avion ne ressemble à aucun type, sa confiance finale est basse.
        """
        # Étage 1 : où sont les avions ? La confiance finale ne peut pas dépasser celle du
        # détecteur, donc on écarte d'emblée les boîtes dont le détecteur doute trop.
        min_detector = self.min_score
        if self.review_min_detector is not None:
            min_detector = min(min_detector, self.review_min_detector)
        result = self.detector.predict(
            image_bgr,
            imgsz=DETECTOR_IMGSZ,
            conf=min_detector,
            iou=NMS_IOU,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        detector_scores = result.boxes.conf.cpu().numpy()
        if len(boxes) == 0:
            return []

        # Étage 2 : quel type d'avion ? Une vignette par boîte, toutes classées d'un coup.
        # Les vignettes d'entraînement étaient en RGB : on convertit (OpenCV lit en BGR).
        crops = [cv2.cvtColor(make_crop(image_bgr, tuple(box)), cv2.COLOR_BGR2RGB) for box in boxes]
        batch = torch.stack([self.transform(crop) for crop in crops]).to(self.device)
        probabilities = self.classifier(batch).softmax(dim=1).cpu().numpy()

        detections = []
        for box, detector_score, probs in zip(boxes, detector_scores, probabilities, strict=True):
            class_index = int(probs.argmax())
            score = float(detector_score * probs[class_index])
            to_review = score < self.min_score  # pas assez sûr d'être un avion ET de son type
            if to_review and (
                self.review_min_detector is None or detector_score < self.review_min_detector
            ):
                continue
            detections.append(
                Detection(
                    box=tuple(float(v) for v in box),
                    class_name=CLASSES[class_index],
                    score=score,
                    detector_score=float(detector_score),
                    classifier_score=float(probs[class_index]),
                    to_review=to_review,
                )
            )
        return detections


def predict_baseline_yolo(model: YOLO, image_bgr: np.ndarray, min_confidence: float, device: str):
    """Le témoin (YOLO à 20 classes) : boîtes, numéros de classe et confiances, en un seul passage."""
    result = model.predict(
        image_bgr,
        imgsz=DETECTOR_IMGSZ,
        conf=min_confidence,
        iou=NMS_IOU,
        device=device,
        verbose=False,
    )[0]
    return (
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.cls.cpu().numpy().astype(int),
        result.boxes.conf.cpu().numpy(),
    )
