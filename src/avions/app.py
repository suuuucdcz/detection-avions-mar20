"""Démonstration : tester le modèle sur les images de TEST, jamais vues pendant l'entraînement.

Lancer depuis la racine du projet :  uv run python scripts/app.py

1. On choisit une des 131 images de test (aucune image extérieure, décision D17).
2. « Détecter les avions » : le modèle répond par des boîtes bleues, là où sa confiance finale
   (détecteur × classifieur) atteint le seuil, 0,5 par défaut comme pour nos scores officiels.
3. « Corriger avec la vérité » : c'est oui ou non. Chaque réponse est juste (vert) ou fausse
   (rouge, avec la raison), et chaque vrai avion sans réponse est oublié (rouge pointillé).
   En haut, le F1 cumulé des images corrigées : après les 131, il vaut exactement 0,954.
"""

import random
import time
from collections import Counter
from dataclasses import dataclass

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from avions.constants import CLASS_TO_INDEX, RAW_IMAGES_DIR, SPLITS_FILE
from avions.data.images import load_image_rgb
from avions.data.voc import load_annotations
from avions.evaluation import box_iou, match_details
from avions.pipeline import DEFAULT_MIN_SCORE, AircraftPipeline, Detection
from avions.visualization import TaggedBox, draw_tagged_boxes

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
