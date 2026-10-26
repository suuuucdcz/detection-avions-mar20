"""Interface web de démonstration (Gradio) : on dépose une image satellite, on voit les avions.

Lancer depuis la racine du projet :  uv run python scripts/app.py

Deux niveaux de résultat (décision D14) :
- avions SÛRS (confiance finale >= seuil) : le résultat automatique, celui qu'on évalue ;
  boîtes épaisses ;
- avions À VÉRIFIER : le détecteur est sûr que c'est un avion, mais la confiance finale reste
  basse (type incertain). Ils sont montrés à part, marqués « ? », pour qu'un humain tranche
  (humain dans la boucle) ; boîtes fines.
Si l'image vient du dataset, on compare à la vérité (avions trouvés, bien classés, fausses alarmes).
"""

import time
from collections import Counter
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from avions.constants import CLASS_TO_INDEX, RAW_ANNOTATIONS_DIR, RAW_IMAGES_DIR, SPLITS_FILE
from avions.data.voc import read_voc_file
from avions.evaluation import match_predictions
from avions.pipeline import DEFAULT_MIN_SCORE, REVIEW_MIN_DETECTOR, AircraftPipeline, Detection
from avions.visualization import draw_detections

# Images de VALIDATION choisies pour la démo (jamais vues à l'entraînement)
EXEMPLES = {
    "1453": "12 avions ; le réseau hésite entre A1 et A19",
    "1320": "5 types différents, dont le rare A11",
    "1700": "41 avions serrés",
    "2350": "un A15 que le réseau confond",
}
COLONNES = ["n°", "type", "confiance", "détecteur", "classifieur", "statut"]
NOMS_DES_JEUX = {
    "train": "d'**entraînement** (le modèle l'a déjà vue : résultat optimiste)",
    "val": "de **validation** (jamais vue à l'entraînement)",
    "test": "de **test** (à n'utiliser qu'une fois, à la fin du projet)",
}


def resume(surs: list[Detection], a_verifier: list[Detection], ms: float) -> str:
    if not surs and not a_verifier:
        return f"**Aucun avion détecté** ({ms:.0f} ms)."
    comptes = Counter(d.class_name for d in surs)
    types = ", ".join(f"{n} × {c}" for c, n in sorted(comptes.items(), key=lambda x: -x[1]))
    texte = f"**{len(surs)} avion(s) sûr(s)** en {ms:.0f} ms" + (f" : {types}." if surs else ".")
    if a_verifier:
        texte += (
            f" **+ {len(a_verifier)} à vérifier** (boîtes fines, « ? ») : le détecteur voit un "
            "avion, mais le type est incertain."
        )
    return texte


def apparies(vraies_boites, vraies_classes, detections, avec_type: bool) -> int:
    """Nombre d'avions réels recouverts (IoU >= 0,5) par une détection (du bon type si demandé)."""
    if not detections:
        return 0
    boites = np.array([d.box for d in detections], float)
    classes = np.array([CLASS_TO_INDEX[d.class_name] for d in detections], int)
    if not avec_type:
        vraies_classes, classes = np.zeros_like(vraies_classes), np.zeros_like(classes)
    return int(match_predictions(vraies_boites, vraies_classes, boites, classes)[:, 0].sum())


def comparer_a_la_verite(image_path: Path, surs: list[Detection], a_verifier: list[Detection]):
    """Si l'image vient du dataset : avions trouvés, bien classés, fausses alarmes."""
    annotation = RAW_ANNOTATIONS_DIR / f"{image_path.stem}.xml"
    if not annotation.exists():
        return ""
    verite = pd.DataFrame(read_voc_file(annotation))
    vraies_boites = verite[["xmin", "ymin", "xmax", "ymax"]].to_numpy(float)
    vraies_classes = verite["class_name"].map(CLASS_TO_INDEX).to_numpy()
    trouves = apparies(vraies_boites, vraies_classes, surs, avec_type=False)
    bien_classes = apparies(vraies_boites, vraies_classes, surs, avec_type=True)

    jeux = pd.read_csv(SPLITS_FILE, dtype={"image_id": str}).set_index("image_id")["split"]
    jeu = NOMS_DES_JEUX.get(jeux.get(image_path.stem, ""), "du dataset")
    n = len(verite)
    texte = (
        f"\n\n**Vérité terrain** (image {jeu}) : {n} avion(s). Avions sûrs : "
        f"trouvés **{trouves}/{n}**, bien classés **{bien_classes}/{n}**, "
        f"fausses alarmes **{len(surs) - trouves}**."
    )
    if a_verifier:
        reels = apparies(vraies_boites, vraies_classes, a_verifier, avec_type=False)
        texte += f" Parmi les {len(a_verifier)} « à vérifier », {reels} sont de vrais avions."
    return texte


def creer_demo(pipeline: AircraftPipeline | None = None) -> gr.Blocks:
    pipeline = pipeline or AircraftPipeline(review_min_detector=REVIEW_MIN_DETECTOR)

    def analyser(image_path, seuil_confiance, montrer_a_verifier):
        if image_path is None:
            return None, "Dépose d'abord une image satellite.", pd.DataFrame(columns=COLONNES)
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        pipeline.min_score = seuil_confiance

        debut = time.perf_counter()
        detections = sorted(pipeline.predict(image), key=lambda d: -d.score)
        ms = 1000 * (time.perf_counter() - debut)
        surs = [d for d in detections if not d.to_review]
        a_verifier = [d for d in detections if d.to_review] if montrer_a_verifier else []

        # Avions sûrs : boîtes épaisses ; avions à vérifier : boîtes fines marquées « ? »
        image_annotee = draw_detections(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), surs + a_verifier)

        lignes = [
            [d.class_name, d.score, d.detector_score, d.classifier_score, statut]
            for groupe, statut in ((surs, "sûr"), (a_verifier, "à vérifier"))
            for d in groupe
        ]
        tableau = pd.DataFrame(
            [[i + 1, *ligne] for i, ligne in enumerate(lignes)], columns=COLONNES
        )
        texte = resume(surs, a_verifier, ms) + comparer_a_la_verite(image_path, surs, a_verifier)
        return image_annotee, texte, tableau.round(2)

    with gr.Blocks(title="Détection d'avions sur images satellites") as demo:
        gr.Markdown(
            "# Détection et classification d'avions sur images satellites\n"
            "Dépose une image satellite (ou choisis un exemple en bas de page). "
            "Le **détecteur** (YOLO11n) trouve les avions, puis le **classifieur** (ResNet-18) "
            "donne leur type parmi A1 … A20. Confiance d'un avion = confiance du détecteur × "
            "confiance du classifieur."
        )
        with gr.Row():
            with gr.Column(scale=1):
                entree = gr.Image(type="filepath", label="Image satellite")
                seuil_confiance = gr.Slider(
                    0.05,
                    0.95,
                    value=DEFAULT_MIN_SCORE,
                    step=0.05,
                    label=(
                        "Seuil de confiance des avions sûrs (plus haut : moins de fausses "
                        "alarmes, mais plus d'avions manqués)"
                    ),
                )
                montrer_a_verifier = gr.Checkbox(
                    value=True, label="Montrer les avions à vérifier (type incertain)"
                )
                bouton = gr.Button("Analyser l'image", variant="primary")
            with gr.Column(scale=2):
                sortie_image = gr.Image(label="Avions détectés")
                sortie_texte = gr.Markdown()
        sortie_tableau = gr.Dataframe(label="Détail des avions détectés", headers=COLONNES)

        bouton.click(
            analyser,
            inputs=[entree, seuil_confiance, montrer_a_verifier],
            outputs=[sortie_image, sortie_texte, sortie_tableau],
        )
        gr.Examples(
            examples=[[str(RAW_IMAGES_DIR / f"{image_id}.jpg")] for image_id in EXEMPLES],
            inputs=[entree],
            example_labels=[f"{image_id} : {texte}" for image_id, texte in EXEMPLES.items()],
            label="Exemples (images de validation)",
        )
    return demo
