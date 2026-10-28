"""Démonstration : tester le modèle sur les images de TEST, jamais vues pendant l'entraînement.

Lancer depuis la racine du projet :  uv run python app.py

1. On choisit une des 131 images de test (aucune image extérieure au dataset).
2. « Détecter les avions » : le modèle répond par des boîtes bleues, là où sa confiance finale
   (détecteur × classifieur) atteint le seuil, 0,5 par défaut comme pour nos scores officiels.
3. « Corriger avec la vérité » : c'est oui ou non. Chaque réponse est juste (vert) ou fausse
   (rouge, avec la raison), et chaque vrai avion sans réponse est oublié (rouge pointillé).
   En haut, le F1 cumulé des images corrigées : après les 131, il vaut exactement 0,954.

Ce fichier est autonome : il contient toute la chaîne en deux étages, la même que dans le
notebook etapes/5_integration.ipynb (lecture des XML, vignettes, détecteur, classifieur, correction).
"""

import random
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import cv2
import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2
from ultralytics import YOLO

# --- Chemins et classes -------------------------------------------------------------------


RACINE = Path(__file__).resolve().parent  # app.py est à la racine du projet
RAW_IMAGES_DIR = RACINE / "data" / "raw" / "images"  # les images du dataset du prof
RAW_ANNOTATIONS_DIR = RACINE / "data" / "raw" / "annotations"  # leurs fichiers XML (Pascal VOC)
SPLITS_FILE = RACINE / "data" / "splits" / "splits.csv"  # le jeu (train, val, test) de chaque image
MODELS_DIR = RACINE / "models"  # les deux modèles finaux

# Les 20 types d'avions, dans l'ordre NUMÉRIQUE (un tri alphabétique donnerait A1, A10, A11...
# et les numéros de classe seraient décalés sans erreur visible)
CLASSES = [f"A{i}" for i in range(1, 21)]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}


# --- Lecture des données (étape 1) --------------------------------------------------------


def read_voc_file(xml_path: Path) -> list[dict]:
    """Lit un fichier XML et renvoie une liste de dictionnaires, un par avion annoté."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    image_width = int(size.findtext("width"))
    image_height = int(size.findtext("height"))

    rows = []
    for obj in root.iter("object"):
        box = obj.find("bndbox")
        rows.append(
            {
                "image_id": Path(xml_path).stem,  # "1005" pour "1005.xml"
                "image_width": image_width,
                "image_height": image_height,
                "class_name": obj.findtext("name"),
                "xmin": int(box.findtext("xmin")),
                "ymin": int(box.findtext("ymin")),
                "xmax": int(box.findtext("xmax")),
                "ymax": int(box.findtext("ymax")),
                # "1" si l'avion est coupé par le bord de l'image
                "truncated": obj.findtext("truncated") == "1",
                # "1" si l'annotateur a jugé l'avion difficile à reconnaître
                "difficult": obj.findtext("difficult") == "1",
            }
        )
    return rows


def load_annotations(annotations_dir: Path = RAW_ANNOTATIONS_DIR) -> pd.DataFrame:
    """Charge tous les fichiers XML d'un dossier dans un tableau : une ligne par avion.

    On ajoute aussi des colonnes calculées à partir des coordonnées de la boîte :
    largeur, hauteur, aire, rapport largeur/hauteur et position du centre.
    """
    rows = []
    for xml_path in sorted(Path(annotations_dir).glob("*.xml")):
        rows.extend(read_voc_file(xml_path))
    df = pd.DataFrame(rows)

    df["box_width"] = df["xmax"] - df["xmin"]
    df["box_height"] = df["ymax"] - df["ymin"]
    df["box_area"] = df["box_width"] * df["box_height"]
    df["aspect_ratio"] = df["box_width"] / df["box_height"]  # > 1 : boîte plus large que haute
    # Centre de la boîte, en proportion de l'image (0 = bord gauche/haut, 1 = bord droit/bas)
    df["center_x"] = (df["xmin"] + df["xmax"]) / 2 / df["image_width"]
    df["center_y"] = (df["ymin"] + df["ymax"]) / 2 / df["image_height"]
    return df


def load_image_rgb(image_path: Path) -> np.ndarray:
    """Charge une image en couleurs, au format RGB (tableau hauteur x largeur x 3).

    OpenCV lit les images en BGR (bleu, vert, rouge) pour des raisons historiques,
    alors que matplotlib et PyTorch attendent du RGB : on inverse donc les canaux.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# --- Les vignettes et le classifieur (étapes 2 et 4) --------------------------------------


CROP_SCALE = 2.0  # côté du carré / plus grand côté de la boîte
CROP_SIZE = 256  # taille des vignettes enregistrées, en pixels
PAD_VALUE = 128  # gris neutre pour compléter au-delà du bord de l'image


def make_crop(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Découpe un carré centré sur la boîte (xmin, ymin, xmax, ymax) et le redimensionne."""
    xmin, ymin, xmax, ymax = box
    center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    half_side = CROP_SCALE * max(xmax - xmin, ymax - ymin) / 2
    x0, y0 = round(center_x - half_side), round(center_y - half_side)
    x1, y1 = round(center_x + half_side), round(center_y + half_side)

    # Si le carré sort de l'image, on ajoute une bordure grise tout autour de l'image
    height, width = image.shape[:2]
    pad = max(0, -x0, -y0, x1 - width, y1 - height)
    if pad:
        image = cv2.copyMakeBorder(
            image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
        )
    square = image[y0 + pad : y1 + pad, x0 + pad : x1 + pad]

    # INTER_AREA pour réduire (net, sans crénelage), INTER_LINEAR pour agrandir
    shrinking = square.shape[0] > CROP_SIZE
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(square, (CROP_SIZE, CROP_SIZE), interpolation=interpolation)


IMAGE_SIZE = 224  # taille d'entrée des réseaux pré-entraînés sur ImageNet
# Moyenne et écart-type des pixels d'ImageNet : les poids pré-entraînés ont appris avec ces valeurs
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# Dans une vignette de 256 px (carré de 2 × la boîte), la boîte de l'avion mesure 128 px
BOX_SIDE = CROP_SIZE / CROP_SCALE
EVAL_CROP_FACTOR = 1.2  # cadrage fixe : 1,2 × la boîte, un peu de marge autour de l'avion


def _to_normalized_tensor() -> list:
    return [
        v2.Resize(IMAGE_SIZE, antialias=True),
        v2.ToDtype(torch.float32, scale=True),  # pixels 0-255 -> 0-1
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]


def eval_transform() -> v2.Compose:
    """Validation, test et application : pas de hasard, juste un cadrage fixe au centre."""
    eval_side = round(BOX_SIDE * EVAL_CROP_FACTOR)
    return v2.Compose([v2.ToImage(), v2.CenterCrop(eval_side), *_to_normalized_tensor()])


def build_model(pretrained: bool) -> nn.Module:
    """ResNet-18 avec une sortie par type d'avion (20) au lieu des 1000 classes d'ImageNet."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))  # nouvelle « tête » à 20 sorties
    return model


def load_classifier(path: Path, device: str = "cpu") -> nn.Module:
    """Recrée le réseau, puis charge les poids appris pendant l'entraînement."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(pretrained=False)  # les poids viennent du fichier, pas d'ImageNet
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


# --- Les deux étages (étape 5) ------------------------------------------------------------


DETECTOR_IMGSZ = 800  # même résolution qu'à l'entraînement du détecteur
NMS_IOU = 0.7  # deux boîtes qui se recouvrent à plus de 70 % : on garde la plus sûre
DEFAULT_MIN_SCORE = 0.5  # seuil sur la confiance FINALE, choisi sur la validation


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax en pixels
    class_name: str
    score: float  # confiance finale = détecteur × classifieur
    detector_score: float
    classifier_score: float


class AircraftPipeline:
    """Détecte et classe les avions d'une image satellite."""

    def __init__(self, min_score: float = DEFAULT_MIN_SCORE, device: str = "cpu"):
        self.detector = YOLO(str(MODELS_DIR / "detecteur_1classe.pt"))
        self.classifier = load_classifier(MODELS_DIR / "classifieur_resnet18.pt", device)
        self.transform = eval_transform()
        self.min_score = min_score
        self.device = device

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        """Les avions d'une image lue avec cv2.imread (donc en BGR)."""
        # Étage 1 : où sont les avions ? La confiance finale ne peut pas dépasser celle
        # du détecteur : on écarte d'emblée les boîtes dont il doute trop.
        result = self.detector.predict(
            image_bgr,
            imgsz=DETECTOR_IMGSZ,
            conf=self.min_score,
            iou=NMS_IOU,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        detector_scores = result.boxes.conf.cpu().numpy()
        if len(boxes) == 0:
            return []

        # Étage 2 : quel type ? Une vignette par boîte, toutes classées d'un coup.
        # Les vignettes d'entraînement étaient en RGB : on convertit (OpenCV lit en BGR).
        crops = [cv2.cvtColor(make_crop(image_bgr, tuple(box)), cv2.COLOR_BGR2RGB) for box in boxes]
        batch = torch.stack([self.transform(crop) for crop in crops]).to(self.device)
        probabilities = self.classifier(batch).softmax(dim=1).cpu().numpy()

        detections = []
        for box, detector_score, probs in zip(boxes, detector_scores, probabilities, strict=True):
            class_index = int(probs.argmax())
            score = float(detector_score * probs[class_index])
            if score >= self.min_score:  # sûr que c'est un avion ET de son type
                detections.append(
                    Detection(
                        box=tuple(float(v) for v in box),
                        class_name=CLASSES[class_index],
                        score=score,
                        detector_score=float(detector_score),
                        classifier_score=float(probs[class_index]),
                    )
                )
        return detections


# --- La correction : IoU et appariement (étape 5) -----------------------------------------


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU de chaque boîte de A avec chaque boîte de B (boîtes au format xmin, ymin, xmax, ymax)."""
    a, b = boxes_a[:, None, :], boxes_b[None, :, :]
    inter_w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    inter_h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    intersection = inter_w * inter_h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return intersection / (area_a + area_b - intersection + 1e-9)


def match_details(true_boxes, true_classes, pred_boxes, pred_classes, pred_scores):
    """Le verdict de CHAQUE prédiction face à la vérité (utilisé par corriger() dans l'interface).

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
        if free:  # un vrai avion encore libre, avec une IoU d'au moins 0,5
            taken.add(free[0])
            matched[p] = int(free[0])  # indice du vrai avion associé
            same_type = true_classes[free[0]] == pred_classes[p]
            statuses[p] = "bon_type" if same_type else "mauvais_type"
    missed = [t for t in range(len(true_boxes)) if t not in taken]
    return statuses, matched, missed


# --- Le dessin des boîtes -----------------------------------------------------------------


# Police livrée avec matplotlib : elle contient é, à, ×, ✓ et ✗ (OpenCV n'écrit que l'ASCII)
FONT_PATH = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"


HALO = (15, 23, 42)  # contour sombre sous chaque trait : la boîte reste visible sur tous les fonds


WHITE = (255, 255, 255)


@dataclass
class TaggedBox:
    """Une boîte à dessiner par draw_tagged_boxes."""

    box: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax en pixels
    color: tuple[int, int, int]  # RGB
    label: str = ""  # vide : pas d'étiquette
    dashed: bool = False  # pointillés (avion oublié par le modèle)


def _dashed_line(draw, start, end, color, width, dash=10, gap=6):
    (x0, y0), (x1, y1) = start, end
    length = max(abs(x1 - x0), abs(y1 - y0))
    for a in range(0, int(length), dash + gap):
        t0, t1 = a / length, min(a + dash, length) / length
        segment = [(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t) for t in (t0, t1)]
        draw.line(segment, fill=color, width=width)


def draw_tagged_boxes(image: np.ndarray, boxes: list[TaggedBox], font_size: int = 18) -> np.ndarray:
    """Dessine des boîtes et des étiquettes en pastille (fond de couleur, texte blanc) sur une COPIE
    de l'image RGB. Les boîtes sont dessinées dans l'ordre : les dernières passent au-dessus.
    """
    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture)
    font = ImageFont.truetype(str(FONT_PATH), font_size)
    width = picture.width
    for b in boxes:
        x0, y0, x1, y1 = (round(v) for v in b.box)
        draw.rectangle([x0, y0, x1, y1], outline=HALO, width=5)
        if b.dashed:
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            for start, end in pairwise(corners):
                _dashed_line(draw, start, end, b.color, width=3)
        else:
            draw.rectangle([x0 + 1, y0 + 1, x1 - 1, y1 - 1], outline=b.color, width=3)
        if not b.label:
            continue
        left, top, right, bottom = draw.textbbox((0, 0), b.label, font=font)
        tag_w, tag_h = right - left + 10, bottom - top + 6
        tx = max(0, min(x0, width - tag_w))  # recalée si elle sort à droite
        ty = y0 - tag_h - 2 if y0 - tag_h - 2 >= 0 else y0 + 3  # au-dessus, sinon dans la boîte
        draw.rounded_rectangle([tx, ty, tx + tag_w, ty + tag_h], radius=4, fill=b.color)
        draw.text((tx + 5 - left, ty + 3 - top), b.label, font=font, fill=WHITE)
    return np.asarray(picture)


# --- L'interface --------------------------------------------------------------------------


EXEMPLE = "3803"  # image affichée à l'ouverture : des A1 pris pour des A19, deux chasseurs proches
BLEU, VERT, ROUGE = (37, 99, 235), (22, 163, 74), (220, 38, 38)  # couleurs des boîtes (RGB)
DENSE = 30  # au-delà de 30 boîtes, les réponses justes n'ont pas d'étiquette (image lisible)
THEME = gr.themes.Soft(primary_hue="blue")


def charger_images_test() -> tuple[pd.DataFrame, list[str]]:
    """Les vrais avions des images de test, et l'ordre d'affichage (l'exemple en premier)."""
    jeux = pd.read_csv(SPLITS_FILE, dtype={"image_id": str}).set_index("image_id")["split"]
    avions = load_annotations()
    avions = avions[avions["image_id"].map(jeux) == "test"]
    autres = sorted(set(avions["image_id"]) - {EXEMPLE}, key=int)
    return avions, [EXEMPLE, *autres]


def lire_image(image_id: str) -> np.ndarray:
    return load_image_rgb(RAW_IMAGES_DIR / f"{image_id}.jpg")


def virgule(x: float, decimales: int = 2) -> str:
    """0.954 -> « 0,95 » : les nombres à la française."""
    return f"{x:.{decimales}f}".replace(".", ",")


def f1_score(justes: int, reponses: int, n_avions: int) -> float:
    """F1 = 2 × justes / (réponses + vrais avions), ce qui revient à 2PR / (P + R).

    On compte des AVIONS, pas des images : c'est ainsi qu'est calculé le F1 officiel (0,954).
    """
    return 2 * justes / (reponses + n_avions)


@dataclass
class Correction:
    """La correction OUI / NON d'une image."""

    justes: list[Detection]
    fausses: list[tuple[Detection, str]]  # (réponse, raison : « vrai : A1 », « doublon »…)
    oublies: list[tuple[tuple, str]]  # vrais avions sans aucune réponse : (boîte, vrai type)
    n_avions: int  # vrais avions de l'image

    def comptes(self) -> tuple[int, int, int]:
        """(justes, réponses du modèle, vrais avions) : de quoi calculer le F1."""
        return len(self.justes), len(self.justes) + len(self.fausses), self.n_avions


def corriger(verite_image: pd.DataFrame, detections: list[Detection]) -> Correction:
    """Compare les réponses aux vraies annotations, avec la même règle que nos scores officiels :
    une réponse est juste si sa boîte recouvre un vrai avion encore libre (IoU ≥ 0,5)
    avec le bon type.
    """
    boites = verite_image[["xmin", "ymin", "xmax", "ymax"]].to_numpy(float)
    types = [str(c) for c in verite_image["class_name"]]
    statuts, associes, manques = match_details(
        boites,
        [CLASS_TO_INDEX[c] for c in types],
        [d.box for d in detections],
        [CLASS_TO_INDEX[d.class_name] for d in detections],
        [d.score for d in detections],
    )
    justes, fausses = [], []
    for d, statut, t in zip(detections, statuts, associes, strict=True):
        if statut == "bon_type":
            justes.append(d)
        elif statut == "mauvais_type":
            fausses.append((d, f"vrai : {types[t]}"))
        else:  # aucun vrai avion libre sous cette boîte
            # doublon : elle recouvre un avion déjà trouvé par une autre boîte ; sinon, la vérité
            # n'a aucun avion ici (le fond, ou un avion que MAR20 n'a pas annoté, comme sur 3607)
            doublon = len(boites) > 0 and box_iou(np.array([d.box]), boites).max() >= 0.5
            fausses.append((d, "doublon" if doublon else "aucun avion annoté"))
    oublies = [(tuple(boites[t]), types[t]) for t in manques]
    return Correction(justes, fausses, oublies, len(boites))


def dessiner_reponses(image_rgb, detections: list[Detection]):
    """Les réponses du modèle, en bleu."""
    boites = [TaggedBox(d.box, BLEU, d.class_name) for d in detections]
    return draw_tagged_boxes(image_rgb, boites, font_size=15 if len(boites) > DENSE else 18)


def dessiner_correction(image_rgb, c: Correction):
    """Vert : juste ; rouge : fausse ; rouge pointillé : oublié. Les erreurs passent au-dessus."""
    dense = len(c.justes) + len(c.fausses) + len(c.oublies) > DENSE
    boites = [TaggedBox(d.box, VERT, "" if dense else f"✓ {d.class_name}") for d in c.justes]
    boites += [TaggedBox(b, ROUGE, f"✗ oublié : {vrai}", dashed=True) for b, vrai in c.oublies]
    boites += [TaggedBox(d.box, ROUGE, f"✗ {d.class_name} ({raison})") for d, raison in c.fausses]
    return draw_tagged_boxes(image_rgb, boites, font_size=15 if dense else 18)


def resume_md(detections: list[Detection], duree: float, seuil: float) -> str:
    """Ce que le modèle a répondu."""
    types = Counter(d.class_name for d in detections)
    liste = ", ".join(f"{n} × {c}" for c, n in types.most_common()) or "aucun"
    return (
        f"**{len(detections)} avion(s) trouvé(s)** : {liste}  \n"
        f"en {virgule(duree, 1)} s, seuil {virgule(seuil)}"
    )


def detail(compteur: Counter) -> str:
    """Counter({'A19 (vrai : A1)': 1, 'A1': 3}) -> « — A19 (vrai : A1), 3 × A1 »."""
    morceaux = [t if k == 1 else f"{k} × {t}" for t, k in compteur.most_common()]
    return " — " + ", ".join(morceaux) if morceaux else ""


def bilan_md(c: Correction) -> str:
    """Le bilan oui / non de l'image."""
    fausses = Counter(f"{d.class_name} ({raison})" for d, raison in c.fausses)
    oublies = Counter(vrai for _, vrai in c.oublies)
    lignes = [
        f"🟩 **Justes : {len(c.justes)}**",
        f"🟥 **Fausses : {len(c.fausses)}**{detail(fausses)}",
        f"⬚ **Oubliés (pointillés) : {len(c.oublies)}**{detail(oublies)}",
        f"→ **{len(c.justes)} avion(s) sur {c.n_avions} bien reconnu(s)**",
    ]
    return "  \n".join(lignes)


def f1_md(cumul: dict, correction: Correction | None = None, au_seuil_officiel=True) -> str:
    """Le F1 cumulé des images corrigées (en additionnant leurs avions), puis celui de l'image.

    cumul : {numéro d'image : (justes, réponses, vrais avions)}.
    """
    if cumul:
        justes = sum(j for j, _, _ in cumul.values())
        reponses = sum(r for _, r, _ in cumul.values())
        n_avions = sum(n for _, _, n in cumul.values())
        f1 = virgule(f1_score(justes, reponses, n_avions), 3)
        texte = f"**F1 cumulé : {f1}** sur {len(cumul)} image(s) corrigée(s)"
    else:
        texte = "**F1 cumulé : —** (corrige des images pour le calculer)"
    if correction is not None:
        texte += f" · cette image : {virgule(f1_score(*correction.comptes()))}"
        texte += "" if au_seuil_officiel else " (seuil modifié : non comptée)"
    return texte


def creer_demo() -> gr.Blocks:
    pipeline = AircraftPipeline()
    verite, ordre = charger_images_test()
    vignettes = [
        (cv2.resize(lire_image(i), (200, 200)), f"★ {i}" if i == EXEMPLE else i) for i in ordre
    ]

    def predire(image_id, seuil):
        """Les réponses du modèle (les plus sûres d'abord) et le temps de calcul."""
        pipeline.min_score = seuil
        debut = time.perf_counter()
        image_bgr = cv2.imread(str(RAW_IMAGES_DIR / f"{image_id}.jpg"))  # le modèle lit du BGR
        detections = sorted(pipeline.predict(image_bgr), key=lambda d: -d.score)
        return detections, time.perf_counter() - debut

    # Chaque fonction renvoie, dans l'ordre de `sorties` : l'image, le résumé du modèle, le bilan,
    # la ligne du F1, puis l'image choisie et les comptes du F1 cumulé (gardés entre deux clics).
    def afficher(image_id, cumul):
        """Nouvelle image : affichée sans boîte, résultats précédents effacés."""
        image = gr.Image(lire_image(image_id), label=f"Image {image_id}")
        return image, "", "", f1_md(cumul), image_id, cumul

    def detecter(image_id, seuil, cumul):
        detections, duree = predire(image_id, seuil)
        image = dessiner_reponses(lire_image(image_id), detections)
        return image, resume_md(detections, duree, seuil), "", f1_md(cumul), image_id, cumul

    def corriger_clic(image_id, seuil, cumul):
        # la détection est refaite à chaque fois (0,2 s) : la correction suit toujours le curseur
        detections, duree = predire(image_id, seuil)
        correction = corriger(verite[verite["image_id"] == image_id], detections)
        # le F1 cumulé n'utilise que le seuil officiel (round : le curseur peut renvoyer 0.5000001)
        au_seuil_officiel = round(seuil, 2) == DEFAULT_MIN_SCORE
        if au_seuil_officiel:
            cumul[image_id] = correction.comptes()  # une image ne compte qu'une fois
        image = dessiner_correction(lire_image(image_id), correction)
        resume = resume_md(detections, duree, seuil)
        ligne = f1_md(cumul, correction, au_seuil_officiel)
        return image, resume, bilan_md(correction), ligne, image_id, cumul

    def sur_vignette(cumul, evt: gr.SelectData):  # evt.index : la vignette cliquée
        return afficher(ordre[evt.index], cumul)

    def au_hasard(cumul):
        index = random.randrange(len(ordre))  # une des 131 vignettes
        return gr.Gallery(selected_index=index), *afficher(ordre[index], cumul)

    with gr.Blocks(title="Détection d'avions · MAR20") as demo:
        etat_image, etat_cumul = gr.State(EXEMPLE), gr.State({})
        gr.Markdown("# ✈️ Détection d'avions sur images satellites")
        ligne_f1 = gr.Markdown(f1_md({}))
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                gr.Markdown(f"### 1. Choisir une image\n{len(ordre)} images de test, jamais vues")
                galerie = gr.Gallery(
                    vignettes,
                    columns=2,
                    height=420,
                    allow_preview=False,
                    object_fit="cover",
                    show_label=False,
                    selected_index=0,
                )
                bouton_hasard = gr.Button("🎲 Image au hasard")
            with gr.Column(scale=3):
                image = gr.Image(
                    lire_image(EXEMPLE),
                    label=f"Image {EXEMPLE}",
                    interactive=False,
                    height="70vh",
                    format="png",
                )
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 2. Le modèle répond")
                bouton_detecter = gr.Button("🔍 Détecter les avions", variant="primary")
                resume = gr.Markdown()
                seuil = gr.Slider(
                    0.05,
                    0.95,
                    value=DEFAULT_MIN_SCORE,
                    step=0.05,
                    label="Seuil de confiance (officiel : 0,50)",
                    info="Après l'avoir changé, recliquer sur un bouton.",
                )
                gr.Markdown("### 3. On corrige avec la vérité")
                bouton_corriger = gr.Button("✅ Corriger avec la vérité", variant="primary")
                bilan = gr.Markdown()

        sorties = [image, resume, bilan, ligne_f1, etat_image, etat_cumul]
        galerie.select(sur_vignette, [etat_cumul], sorties)
        bouton_hasard.click(au_hasard, [etat_cumul], [galerie, *sorties])
        bouton_detecter.click(detecter, [etat_image, seuil, etat_cumul], sorties)
        bouton_corriger.click(corriger_clic, [etat_image, seuil, etat_cumul], sorties)
    return demo


if __name__ == "__main__":
    creer_demo().launch(inbrowser=True, theme=THEME)
