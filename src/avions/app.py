"""Interface web de démonstration (Gradio) : tester le modèle sur les images de TEST.

Lancer depuis la racine du projet :  uv run python scripts/app.py

Comme le demande le sujet, aucune image extérieure n'est ajoutée : on choisit une des 131 images
du jeu de test, qui n'ont servi ni à entraîner ni à régler le modèle (décision D17).

1. « Détecter les avions » : avions SÛRS (boîtes épaisses) et avions À VÉRIFIER (boîtes fines « ? »,
   le détecteur voit un avion mais le type est incertain).
2. « Comparer avec la vérité » : chaque boîte est jugée face aux vraies annotations :
   vert = bon type, orange = mauvais type, rouge = fausse alarme, violet = avion manqué.
"""

import json
import random
import time
from collections import Counter

import cv2
import gradio as gr
import pandas as pd

from avions.constants import CLASS_TO_INDEX, RAW_IMAGES_DIR, RUNS_DIR, SPLITS_FILE
from avions.data.voc import load_annotations
from avions.evaluation import match_details
from avions.pipeline import DEFAULT_MIN_SCORE, REVIEW_MIN_DETECTOR, AircraftPipeline, Detection
from avions.visualization import draw_boxes, draw_detections

# Images de test conseillées pour une démonstration (affichées en premier, avec une étoile)
IMAGES_CONSEILLEES = {
    "883": "11 avions de 2 types : sans faute",
    "3607": "50 avions serrés : le cas le plus difficile",
    "3803": "des A1 confondus avec des A19, très ressemblants",
    "1128": "le modèle hésite : tout part en « à vérifier »",
}
COULEURS = {  # en RGB, pour les boîtes de la comparaison
    "bon_type": (22, 163, 74),
    "mauvais_type": (234, 88, 12),
    "fausse_alarme": (220, 38, 38),
    "manque": (147, 51, 234),
}
VERDICTS = {
    "bon_type": "bon type",
    "mauvais_type": "mauvais type",
    "fausse_alarme": "fausse alarme",
    "manque": "avion manqué",
}
COLONNES_ANALYSE = ["n°", "type", "confiance", "détecteur", "classifieur", "niveau"]
COLONNES_VERIF = ["verdict", "type prédit", "vrai type", "confiance", "niveau"]

THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate", neutral_hue="slate")
CSS = """
.entete h1 { margin-bottom: 4px; }
.entete p { margin-top: 0; opacity: .8; }
.cartes { display: flex; gap: 12px; flex-wrap: wrap; margin: 6px 0; }
.carte { flex: 1; min-width: 120px; padding: 12px 16px; border-radius: 12px;
         background: var(--block-background-fill); border: 1px solid var(--border-color-primary); }
.carte .valeur { font-size: 26px; font-weight: 700; line-height: 1.2; }
.carte .libelle { font-size: 13px; opacity: .75; }
.indigo .valeur { color: #4f46e5; } .gris .valeur { color: #64748b; }
.vert .valeur { color: #16a34a; } .orange .valeur { color: #ea580c; }
.rouge .valeur { color: #dc2626; } .violet .valeur { color: #9333ea; }
.legende { font-size: 14px; opacity: .85; }
.pied { font-size: 13px; opacity: .65; text-align: center; }
"""


def cartes(elements: list[tuple]) -> str:
    """Petites cartes de chiffres clés : (valeur, libellé, couleur)."""
    contenu = "".join(
        f"<div class='carte {couleur}'><div class='valeur'>{valeur}</div>"
        f"<div class='libelle'>{libelle}</div></div>"
        for valeur, libelle, couleur in elements
    )
    return f"<div class='cartes'>{contenu}</div>"


def entete() -> str:
    """Titre et scores officiels sur le test (lus dans runs/comparaison_test/metrics.json)."""
    html = (
        "<div class='entete'><h1>✈️ Détection d'avions sur images satellites</h1>"
        "<p>Un <b>détecteur</b> (YOLO11n) trouve les avions, puis un <b>classifieur</b> "
        "(ResNet-18) reconnaît leur type parmi 20 (A1 … A20). Toutes les images proposées "
        "viennent du <b>jeu de test</b> : le modèle ne les a jamais vues.</p></div>"
    )
    fichier = RUNS_DIR / "comparaison_test" / "metrics.json"
    if fichier.exists():
        s = json.loads(fichier.read_text(encoding="utf-8"))["deux_etages"]["au_seuil"]
        n = int(s["bon_type"] + s["mauvais_type"] + s["manques"])
        html += cartes(
            [
                (f"{s['f1']:.3f}".replace(".", ","), "F1-score sur le test", "indigo"),
                (f"{s['precision']:.3f}".replace(".", ","), "précision", "indigo"),
                (f"{int(s['fausses_alarmes'])} / {n}", "fausses alarmes / avions", "rouge"),
                (f"{int(s['mauvais_type'])} / {n}", "mauvais types / avions", "orange"),
            ]
        )
    return html


def charger_images_test() -> tuple[pd.DataFrame, list[str]]:
    """Les vrais avions des images de test, et l'ordre d'affichage (images conseillées d'abord)."""
    jeux = pd.read_csv(SPLITS_FILE, dtype={"image_id": str}).set_index("image_id")["split"]
    avions = load_annotations()
    avions = avions[avions["image_id"].map(jeux) == "test"]
    autres = sorted(set(avions["image_id"]) - set(IMAGES_CONSEILLEES), key=int)
    return avions, [*IMAGES_CONSEILLEES, *autres]


def lire_image_bgr(image_id: str):
    return cv2.imread(str(RAW_IMAGES_DIR / f"{image_id}.jpg"))


def vignette(image_id: str):
    return cv2.cvtColor(cv2.resize(lire_image_bgr(image_id), (200, 200)), cv2.COLOR_BGR2RGB)


def verifier_detections(image_bgr, verite: pd.DataFrame, detections: list[Detection]):
    """Image de comparaison (une couleur par verdict), cartes de bilan et tableau détaillé."""
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

    surs = Counter(s for d, s in zip(detections, statuts, strict=True) if not d.to_review)
    a_verifier = Counter(s for d, s in zip(detections, statuts, strict=True) if d.to_review)
    bilan = cartes(
        [
            (len(verite), "avions dans la vérité", "gris"),
            (surs["bon_type"], "bon type (sûrs)", "vert"),
            (surs["mauvais_type"], "mauvais type (sûrs)", "orange"),
            (surs["fausse_alarme"], "fausses alarmes (sûrs)", "rouge"),
            (len(manques), "avions manqués", "violet"),
        ]
    )
    if sum(a_verifier.values()):
        bilan += (
            f"<p class='legende'>Parmi les {sum(a_verifier.values())} avions « à vérifier » : "
            f"{a_verifier['bon_type']} bon type, {a_verifier['mauvais_type']} mauvais type, "
            f"{a_verifier['fausse_alarme']} fausse(s) alarme(s).</p>"
        )
    bilan += (
        "<p class='legende'>🟩 bon type · 🟧 mauvais type · 🟥 fausse alarme · 🟪 avion manqué "
        "· boîte fine = avion « à vérifier »</p>"
    )

    tableau = pd.DataFrame(lignes, columns=COLONNES_VERIF)
    ordre = {"mauvais type": 0, "fausse alarme": 1, "avion manqué": 2, "bon type": 3}
    tableau = tableau.sort_values("verdict", key=lambda v: v.map(ordre), kind="stable")
    return image, bilan, tableau


def creer_demo(pipeline: AircraftPipeline | None = None) -> gr.Blocks:
    pipeline = pipeline or AircraftPipeline(review_min_detector=REVIEW_MIN_DETECTOR)
    verite, ordre = charger_images_test()
    galerie = [(vignette(i), f"★ {i}" if i in IMAGES_CONSEILLEES else i) for i in ordre]
    conseils = "\n".join(f"- **★ {i}** : {texte}" for i, texte in IMAGES_CONSEILLEES.items())

    def afficher(image_id):
        """Nouvelle image : on l'affiche sans boîtes et on efface les résultats précédents."""
        image = cv2.cvtColor(lire_image_bgr(image_id), cv2.COLOR_BGR2RGB)
        info = f"**Image {image_id}** sélectionnée. Clique sur **Détecter les avions**."
        vide_a = pd.DataFrame(columns=COLONNES_ANALYSE)
        vide_v = pd.DataFrame(columns=COLONNES_VERIF)
        return image_id, info, image, None, "", "", vide_a, vide_v, None

    def choisir(evt: gr.SelectData):
        return afficher(ordre[evt.index])

    def au_hasard():
        index = random.randrange(len(ordre))
        return gr.Gallery(selected_index=index), *afficher(ordre[index])

    def detecter(image_id, seuil, montrer_a_verifier):
        image = lire_image_bgr(image_id)
        pipeline.min_score = seuil
        debut = time.perf_counter()
        detections = sorted(pipeline.predict(image), key=lambda d: (d.to_review, -d.score))
        ms = 1000 * (time.perf_counter() - debut)
        if not montrer_a_verifier:
            detections = [d for d in detections if not d.to_review]
        surs = [d for d in detections if not d.to_review]
        types = Counter(d.class_name for d in surs)
        resume = cartes(
            [
                (len(surs), "avions sûrs", "indigo"),
                (len(detections) - len(surs), "à vérifier", "orange"),
                (f"{ms:.0f} ms", "temps de calcul", "gris"),
            ]
        )
        if types:
            liste = ", ".join(f"{n} × {c}" for c, n in types.most_common())
            resume += f"<p class='legende'>Types trouvés : {liste}. Clique sur <b>Comparer avec la vérité</b>.</p>"
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
        return annotee, resume, tableau, detections, None, ""

    def comparer(image_id, detections, seuil, montrer_a_verifier):
        sorties_detection = None
        if detections is None:  # on n'a pas encore détecté : on le fait d'abord
            sorties_detection = detecter(image_id, seuil, montrer_a_verifier)
            detections = sorties_detection[3]
        image, bilan, tableau = verifier_detections(
            lire_image_bgr(image_id), verite[verite["image_id"] == image_id], detections
        )
        if sorties_detection is None:
            return gr.skip(), gr.skip(), gr.skip(), detections, image, bilan, tableau
        annotee, resume, tableau_analyse, _, _, _ = sorties_detection
        return annotee, resume, tableau_analyse, detections, image, bilan, tableau

    with gr.Blocks(title="Détection d'avions sur images satellites") as demo:
        image_courante = gr.State(ordre[0])
        detections_memoire = gr.State()
        gr.HTML(entete())
        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=320):
                gr.Markdown("### 1. Choisis une image de test")
                choix = gr.Gallery(
                    value=galerie,
                    columns=4,
                    height=430,
                    allow_preview=False,
                    object_fit="cover",
                    show_label=False,
                    selected_index=0,
                )
                bouton_hasard = gr.Button("🎲 Image au hasard")
                info = gr.Markdown()
                gr.Markdown(f"**Images conseillées :**\n{conseils}")
                with gr.Accordion("Réglages", open=False):
                    seuil = gr.Slider(
                        0.05,
                        0.95,
                        value=DEFAULT_MIN_SCORE,
                        step=0.05,
                        label="Seuil de confiance des avions sûrs (0,5 : réglage choisi sur la validation)",
                    )
                    montrer_a_verifier = gr.Checkbox(
                        value=True, label="Montrer les avions à vérifier"
                    )
            with gr.Column(scale=5):
                gr.Markdown("### 2. Détecte les avions, puis 3. compare avec la vérité")
                with gr.Row():
                    bouton_detecter = gr.Button(
                        "🔍 Détecter les avions", variant="primary", size="lg"
                    )
                    bouton_comparer = gr.Button(
                        "✅ Comparer avec la vérité", variant="secondary", size="lg"
                    )
                with gr.Row():
                    image_modele = gr.Image(
                        label="Ce que voit le modèle", interactive=False, height=460
                    )
                    image_verite = gr.Image(
                        label="Comparaison avec la vérité", interactive=False, height=460
                    )
                with gr.Row():
                    resume_detection = gr.HTML()
                    bilan_verite = gr.HTML()
        with gr.Accordion("Détails boîte par boîte", open=False), gr.Row():
            tableau_analyse = gr.Dataframe(label="Avions détectés", headers=COLONNES_ANALYSE)
            tableau_verif = gr.Dataframe(label="Verdict de chaque boîte", headers=COLONNES_VERIF)
        gr.HTML(
            "<p class='pied'>Projet IPSA « Introduction à l'IA » · dataset MAR20 · "
            "détecteur YOLO11n + classifieur ResNet-18 · calcul sur le processeur</p>"
        )

        sorties_affichage = [
            image_courante,
            info,
            image_modele,
            image_verite,
            resume_detection,
            bilan_verite,
            tableau_analyse,
            tableau_verif,
            detections_memoire,
        ]
        choix.select(choisir, None, sorties_affichage)
        bouton_hasard.click(au_hasard, None, [choix, *sorties_affichage])
        demo.load(lambda: afficher(ordre[0]), None, sorties_affichage)
        bouton_detecter.click(
            detecter,
            inputs=[image_courante, seuil, montrer_a_verifier],
            outputs=[
                image_modele,
                resume_detection,
                tableau_analyse,
                detections_memoire,
                image_verite,
                bilan_verite,
            ],
        )
        bouton_comparer.click(
            comparer,
            inputs=[image_courante, detections_memoire, seuil, montrer_a_verifier],
            outputs=[
                image_modele,
                resume_detection,
                tableau_analyse,
                detections_memoire,
                image_verite,
                bilan_verite,
                tableau_verif,
            ],
        )
    return demo
