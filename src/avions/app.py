"""Interface web de démonstration (Gradio) : tester le modèle sur des images qu'il n'a jamais vues.

Lancer depuis la racine du projet :  uv run python scripts/app.py

Comme le demande le sujet, on n'ajoute aucune image extérieure : on choisit une image du dataset
parmi celles qui n'ont JAMAIS servi à l'entraînement, c'est-à-dire les jeux de validation et de
test (décision D17).

1. « Analyser » : le modèle trouve les avions et donne leur type.
   - avions SÛRS (confiance finale >= seuil) : boîtes épaisses ;
   - avions À VÉRIFIER : le détecteur voit un avion mais le type est incertain : boîtes fines « ? ».
2. « Vérifier » : on compare avec les vraies annotations, boîte par boîte :
   vert = bon type, orange = mauvais type, rouge = fausse alarme, violet = avion manqué.
"""

import random
import time
from collections import Counter

import cv2
import gradio as gr
import pandas as pd

from avions.constants import CLASS_TO_INDEX, RAW_IMAGES_DIR, SPLITS_FILE
from avions.data.voc import load_annotations
from avions.evaluation import match_details
from avions.pipeline import DEFAULT_MIN_SCORE, REVIEW_MIN_DETECTOR, AircraftPipeline, Detection
from avions.visualization import draw_boxes, draw_detections

JEUX = {"Validation (130 images)": "val", "Test (131 images)": "test"}
JEU_PAR_DEFAUT = "Validation (130 images)"
NOMS_DES_JEUX = {"val": "validation", "test": "test (évaluation finale)"}
COULEURS = {  # en RGB
    "bon_type": (0, 200, 0),
    "mauvais_type": (255, 140, 0),
    "fausse_alarme": (230, 0, 0),
    "manque": (170, 0, 255),
}
VERDICTS = {
    "bon_type": "bon type",
    "mauvais_type": "mauvais type",
    "fausse_alarme": "fausse alarme",
    "manque": "avion manqué",
}
LEGENDE = (
    "🟩 bon type · 🟧 mauvais type · 🟥 fausse alarme · 🟪 avion manqué "
    "· boîte fine = avion « à vérifier »"
)
EXEMPLES = (
    "**Images intéressantes (validation)** : **1453**, A1 et A19 qui se ressemblent ; "
    "**1320**, 5 types dont le rare A11 ; **1700**, 41 avions serrés ; **2350**, un A15 confondu."
)
COLONNES_ANALYSE = ["n°", "type", "confiance", "détecteur", "classifieur", "niveau"]
COLONNES_VERIF = ["verdict", "type prédit", "vrai type", "confiance", "niveau"]


def charger_images_non_vues() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Les vrais avions (annotations) et le résumé des images de validation et de test."""
    jeux = pd.read_csv(SPLITS_FILE, dtype={"image_id": str}).set_index("image_id")["split"]
    avions = load_annotations()
    avions = avions[avions["image_id"].map(jeux).isin(["val", "test"])]
    images = avions.groupby("image_id").agg(
        n_avions=("class_name", "size"),
        types=("class_name", lambda c: ", ".join(sorted(set(c), key=lambda t: int(t[1:])))),
    )
    images["jeu"] = images.index.map(jeux)
    return avions, images


def lire_image_bgr(image_id: str):
    return cv2.imread(str(RAW_IMAGES_DIR / f"{image_id}.jpg"))


def resume_analyse(surs: list[Detection], a_verifier: list[Detection], ms: float) -> str:
    if not surs and not a_verifier:
        return f"**Aucun avion détecté** ({ms:.0f} ms)."
    comptes = Counter(d.class_name for d in surs)
    types = ", ".join(f"{n} × {c}" for c, n in sorted(comptes.items(), key=lambda x: -x[1]))
    texte = f"**{len(surs)} avion(s) sûr(s)** en {ms:.0f} ms" + (f" : {types}." if surs else ".")
    if a_verifier:
        texte += f" **+ {len(a_verifier)} à vérifier** (boîtes fines « ? »)."
    return texte + " Clique sur **Vérifier** pour comparer avec les vraies annotations."


def verifier_detections(image_bgr, verite: pd.DataFrame, detections: list[Detection]):
    """Image de vérification (une couleur par verdict), texte de bilan et tableau détaillé."""
    vraies_boites = verite[["xmin", "ymin", "xmax", "ymax"]].to_numpy(float)
    vraies_classes = verite["class_name"].to_numpy()
    statuts, associes, manques = match_details(
        vraies_boites,
        [CLASS_TO_INDEX[c] for c in vraies_classes],
        [d.box for d in detections],
        [CLASS_TO_INDEX[d.class_name] for d in detections],
        [d.score for d in detections],
    )

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    lignes = []
    for d, statut, t in zip(detections, statuts, associes, strict=True):
        vrai = vraies_classes[t] if t is not None else "—"
        etiquette = {
            "bon_type": f"{d.class_name} ok",
            "mauvais_type": f"{d.class_name} (vrai {vrai})",
            "fausse_alarme": f"{d.class_name} fausse alarme",
        }[statut]
        image = draw_boxes(
            image,
            [tuple(round(v) for v in d.box)],
            [etiquette + (" ?" if d.to_review else "")],
            thickness=1 if d.to_review else 3,
            colors=[COULEURS[statut]],
        )
        niveau = "à vérifier" if d.to_review else "sûr"
        lignes.append([VERDICTS[statut], d.class_name, vrai, round(d.score, 2), niveau])
    for t in manques:
        image = draw_boxes(
            image,
            [tuple(int(v) for v in vraies_boites[t])],
            [f"manque {vraies_classes[t]}"],
            thickness=2,
            colors=[COULEURS["manque"]],
        )
        lignes.append([VERDICTS["manque"], "—", vraies_classes[t], None, "—"])

    def compte(niveau_a_verifier: bool) -> Counter:
        return Counter(
            s for d, s in zip(detections, statuts, strict=True) if d.to_review == niveau_a_verifier
        )

    surs, a_verifier = compte(False), compte(True)
    texte = (
        f"**Vérité : {len(verite)} avion(s).** "
        f"Avions sûrs : **{surs['bon_type']} bon type**, **{surs['mauvais_type']} mauvais type**, "
        f"**{surs['fausse_alarme']} fausse(s) alarme(s)**. "
    )
    if a_verifier:
        texte += (
            f"À vérifier : {a_verifier['bon_type']} bon type, {a_verifier['mauvais_type']} "
            f"mauvais type, {a_verifier['fausse_alarme']} fausse(s) alarme(s). "
        )
    texte += f"Avions manqués : **{len(manques)}**.\n\n{LEGENDE}"

    tableau = pd.DataFrame(lignes, columns=COLONNES_VERIF)
    ordre = {"mauvais type": 0, "fausse alarme": 1, "avion manqué": 2, "bon type": 3}
    tableau = tableau.sort_values("verdict", key=lambda v: v.map(ordre), kind="stable")
    return image, texte, tableau


def creer_demo(pipeline: AircraftPipeline | None = None) -> gr.Blocks:
    pipeline = pipeline or AircraftPipeline(review_min_detector=REVIEW_MIN_DETECTOR)
    verite, images = charger_images_non_vues()

    def choix_images(jeu: str) -> list[tuple[str, str]]:
        ids = sorted(images.index[images["jeu"] == JEUX[jeu]], key=int)
        return [
            (f"{i} : {images.at[i, 'n_avions']} avion(s), {images.at[i, 'types']}", i) for i in ids
        ]

    def changer_de_jeu(jeu):
        choix = choix_images(jeu)
        return gr.Dropdown(choices=choix, value=choix[0][1])

    def image_au_hasard(jeu):
        return gr.Dropdown(value=random.choice(choix_images(jeu))[1])

    def montrer(image_id):
        """Nouvelle image : on l'affiche sans boîtes et on efface les résultats précédents."""
        if not image_id:
            return None, None, "", "", None, None
        jeu = NOMS_DES_JEUX[images.at[image_id, "jeu"]]
        texte = f"Image **{image_id}** (jeu de {jeu}). Clique sur **Analyser**."
        image = cv2.cvtColor(lire_image_bgr(image_id), cv2.COLOR_BGR2RGB)
        return image, None, texte, "", pd.DataFrame(columns=COLONNES_ANALYSE), None

    def analyser(image_id, seuil, montrer_a_verifier):
        if not image_id:
            return None, "Choisis d'abord une image.", None, None, "", None
        image = lire_image_bgr(image_id)
        pipeline.min_score = seuil
        debut = time.perf_counter()
        detections = sorted(pipeline.predict(image), key=lambda d: (d.to_review, -d.score))
        ms = 1000 * (time.perf_counter() - debut)
        if not montrer_a_verifier:
            detections = [d for d in detections if not d.to_review]
        surs = [d for d in detections if not d.to_review]
        a_verifier = [d for d in detections if d.to_review]

        tableau = pd.DataFrame(
            [
                [
                    i + 1,
                    d.class_name,
                    d.score,
                    d.detector_score,
                    d.classifier_score,
                    "à vérifier" if d.to_review else "sûr",
                ]
                for i, d in enumerate(detections)
            ],
            columns=COLONNES_ANALYSE,
        ).round(2)
        annotee = draw_detections(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections)
        return annotee, resume_analyse(surs, a_verifier, ms), tableau, detections, "", None

    def verifier(image_id, detections):
        if not image_id or detections is None:
            return None, "Clique d'abord sur **Analyser**.", None
        image, texte, tableau = verifier_detections(
            lire_image_bgr(image_id), verite[verite["image_id"] == image_id], detections
        )
        return image, texte, tableau

    with gr.Blocks(title="Détection d'avions sur images satellites") as demo:
        gr.Markdown(
            "# Détection et classification d'avions sur images satellites\n"
            "Choisis une image que le modèle **n'a jamais vue pendant son entraînement** "
            "(validation ou test), clique sur **Analyser**, puis sur **Vérifier** pour voir où "
            "il a raison et où il se trompe. Le **détecteur** (YOLO11n) trouve les avions, puis "
            "le **classifieur** (ResNet-18) donne leur type (A1 … A20)."
        )
        detections_memoire = gr.State()
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                jeu = gr.Radio(
                    list(JEUX), value=JEU_PAR_DEFAUT, label="Images jamais vues à l'entraînement"
                )
                choix_initial = choix_images(JEU_PAR_DEFAUT)
                image_id = gr.Dropdown(choices=choix_initial, value="1453", label="Image")
                bouton_hasard = gr.Button("Image au hasard")
                gr.Markdown(EXEMPLES)
                seuil = gr.Slider(
                    0.05,
                    0.95,
                    value=DEFAULT_MIN_SCORE,
                    step=0.05,
                    label="Seuil de confiance des avions sûrs (relancer « Analyser » après un changement)",
                )
                montrer_a_verifier = gr.Checkbox(value=True, label="Montrer les avions à vérifier")
                bouton_analyser = gr.Button("Analyser", variant="primary")
                bouton_verifier = gr.Button("Vérifier", variant="secondary")
            with gr.Column(scale=3):
                with gr.Row():
                    image_modele = gr.Image(label="Prédictions du modèle", interactive=False)
                    image_verite = gr.Image(
                        label="Vérification (vraies annotations)", interactive=False
                    )
                texte_analyse = gr.Markdown()
                texte_verif = gr.Markdown()
        tableau_analyse = gr.Dataframe(
            label="Avions détectés par le modèle", headers=COLONNES_ANALYSE
        )
        tableau_verif = gr.Dataframe(label="Verdict boîte par boîte", headers=COLONNES_VERIF)

        jeu.change(changer_de_jeu, jeu, image_id)
        bouton_hasard.click(image_au_hasard, jeu, image_id)
        sorties_montrer = [
            image_modele,
            image_verite,
            texte_analyse,
            texte_verif,
            tableau_analyse,
            tableau_verif,
        ]
        image_id.change(montrer, image_id, sorties_montrer)
        demo.load(montrer, image_id, sorties_montrer)
        bouton_analyser.click(
            analyser,
            inputs=[image_id, seuil, montrer_a_verifier],
            outputs=[
                image_modele,
                texte_analyse,
                tableau_analyse,
                detections_memoire,
                texte_verif,
                image_verite,
            ],
        )
        bouton_verifier.click(
            verifier,
            inputs=[image_id, detections_memoire],
            outputs=[image_verite, texte_verif, tableau_verif],
        )
    return demo
