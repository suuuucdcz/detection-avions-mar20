"""Interface web de démonstration (Gradio) : tester le modèle sur les images de TEST.

Lancer depuis la racine du projet :  uv run python scripts/app.py

Comme le demande le sujet, aucune image extérieure n'est ajoutée : on choisit une des 131 images
du jeu de test, qui n'ont servi ni à entraîner ni à régler le modèle (décision D17).

L'écran se lit de gauche à droite, comme une phrase (décision D19) :
1. on CHOISIT une image (à gauche) ; elle s'affiche en grand (au centre) ;
2. « Détecter les avions » : le modèle répond (boîtes bleues), là où sa confiance finale atteint
   le seuil de 0,5, exactement comme pour nos scores officiels ;
3. « Corriger avec la vérité » : c'est oui ou non. Chaque réponse est JUSTE (vert) ou FAUSSE
   (rouge, la raison est écrite), et un vrai avion sans réponse est OUBLIÉ (rouge pointillé).
   Additionnés sur les 131 images, ces bilans redonnent exactement nos scores officiels.
"""

import json
import random
import time
from collections import Counter
from dataclasses import dataclass

import cv2
import gradio as gr
import pandas as pd

from avions.constants import CLASS_TO_INDEX, RAW_IMAGES_DIR, RUNS_DIR, SPLITS_FILE
from avions.data.voc import load_annotations
from avions.evaluation import match_details
from avions.pipeline import DEFAULT_MIN_SCORE, AircraftPipeline, Detection
from avions.visualization import TaggedBox, draw_tagged_boxes

# Exemples pour la démonstration, du plus simple au plus difficile (affichés en premier)
IMAGES_CONSEILLEES = {
    "883": "Sans faute",  # 11 avions, 11 justes
    "3803": "A1 ou A19 ?",  # 13 justes, 1 A19 au lieu de A1, 3 A1 oubliés
    "3256": "Bien vus, mal classés",  # 5 A13 pris pour des A15
    "3734": "Faux avions",  # 9 justes, 2 « pas un avion »
    "3607": "50 avions serrés",  # 48 justes, 2 mauvais types, 1 « pas un avion »
}
BLEU, VERT, ROUGE = (37, 99, 235), (22, 163, 74), (220, 38, 38)  # couleurs des boîtes (RGB)
DENSE = 30  # au-delà de 30 boîtes, les réponses justes n'ont pas d'étiquette (image lisible)
ETAPES = {"image": "Image seule", "reponse": "Réponses du modèle", "correction": "Correction"}
COLONNES_REPONSE = ["n°", "type", "confiance finale", "détecteur", "classifieur"]
COLONNES_CORRECTION = ["verdict", "réponse du modèle", "vérité", "confiance"]

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    # police livrée avec Gradio : elle fonctionne sans internet (le jour de l'oral)
    font=[gr.themes.LocalFont("IBM Plex Sans", weights=(400, 600, 700)), "system-ui", "sans-serif"],
).set(block_radius="12px", button_large_radius="10px", button_medium_radius="10px")

# Le design est prévu en clair : on force le thème clair, même si Windows est en mode sombre
JS = """() => {
  const u = new URL(location.href);
  if (u.searchParams.get('__theme') !== 'light') {
    u.searchParams.set('__theme', 'light');
    location.replace(u.href);
  }
}"""
# Après un choix par bouton, on fait défiler la galerie jusqu'à la vignette choisie
DEFILER = """() => setTimeout(() => document.querySelector('#galerie .thumbnail-item.selected')
  ?.scrollIntoView({block: 'nearest', behavior: 'smooth'}), 80)"""

CSS = """
#app { max-width: 1680px; margin: 0 auto; width: 100%; gap: 12px; }
.entete { display: flex; justify-content: space-between; align-items: center; gap: 16px;
          flex-wrap: wrap; padding: 4px 2px; }
.entete h1 { margin: 0; font: 700 22px/1.25 var(--font); color: #0f172a; }
.badge { font-size: 13px; padding: 4px 12px; border-radius: 999px; background: #e2e8f0;
         color: #334155; }

/* trois colonnes : choisir | regarder | répondre puis corriger */
.carte { background: #fff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 16px !important;
         box-shadow: 0 1px 2px rgba(15, 23, 42, .06); gap: 10px !important; }
#col-choix { flex: 0 0 300px !important; max-width: 300px; }
#col-droite { flex: 0 0 380px !important; max-width: 380px; gap: 12px !important; }
#carte-reponse { flex-grow: 0 !important; }  /* seule la carte 3 s'étire jusqu'en bas */
#carte-correction #details { margin-top: auto; }
.carte .html-container { padding: 0 !important; }  /* Gradio ajoute 10 px en haut et en bas */
@media (max-width: 1200px) {
  #col-choix, #col-droite { flex: 1 1 100% !important; max-width: none; }
}
@media (max-height: 820px) {  /* petits écrans (projecteur 1366 x 768) : tout reste visible */
  .carte { padding: 12px !important; gap: 6px !important; }
  #col-droite { gap: 8px !important; }
  #col-droite button.lg { padding-top: 8px !important; padding-bottom: 8px !important; }
  .ligne { min-height: 30px; }
  .ligne .nb { font-size: 24px; }
}

/* titres d'étape : une pastille numérotée, puis le titre */
.etape { display: flex; align-items: center; gap: 10px; margin: 0; color: #334155;
         font: 700 13px/1.3 var(--font); letter-spacing: .06em; text-transform: uppercase; }
.etape .num { flex: none; width: 24px; height: 24px; border-radius: 50%; display: grid;
              place-items: center; background: #2563eb; color: #fff; letter-spacing: 0; }
.portee { margin-left: auto; font: 600 10.5px/1 var(--font); letter-spacing: .06em; color: #64748b;
          background: #f1f5f9; padding: 4px 6px; border-radius: 6px; }
.sur-titre { margin: 4px 0 0; font: 700 11px/1 var(--font); letter-spacing: .06em;
             text-transform: uppercase; color: #64748b; }

/* exemples conseillés et galerie */
.conseils { gap: 4px !important; }
.conseil { justify-content: flex-start !important; text-align: left; font-weight: 500 !important;
           font-size: 14px !important; padding-top: 5px !important; padding-bottom: 5px !important;
           min-height: 0 !important; }
#galerie .grid-container { grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
                           grid-auto-rows: auto !important; gap: 6px !important; }
#galerie .thumbnail-item { height: auto; aspect-ratio: 1 / 1; border-radius: 8px; }
#galerie .thumbnail-item.selected { border-color: #2563eb;
                                    box-shadow: inset 0 0 0 3px #2563eb; }
#galerie .caption-label { opacity: 1; font-size: 11px; padding: 1px 6px; }

/* l'image au centre, sur un fond sombre */
.titre-scene { display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
               font: 600 16px/1.3 var(--font); color: #0f172a; }
.etoile { font-size: 13px; font-weight: 600; color: #1d4ed8; background: #eff6ff;
          border: 1px solid #bfdbfe; padding: 2px 8px; border-radius: 999px; }
.mode { margin-left: auto; font: 700 11px/1 var(--font); letter-spacing: .06em;
        text-transform: uppercase; padding: 6px 10px; border-radius: 999px; }
.mode.image { background: #e2e8f0; color: #334155; }
.mode.reponse { background: #eff6ff; color: #1d4ed8; }
.mode.correction { background: #0f172a; color: #fff; }
#scene, #scene .image-frame { background: #0f172a !important; border-radius: 10px; }
#scene .image-frame img { object-fit: contain !important; }

/* lignes de bilan : la pastille a la couleur et le trait des boîtes, elle sert de légende */
.ligne { display: grid; grid-template-columns: 26px 1fr auto; align-items: center; gap: 10px;
         padding: 2px 0; min-height: 38px; font-size: 15px; font-weight: 600; }
.ligne .nb { font-size: 28px; font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums; }
.ligne .pastille, .ligne .nb { color: inherit !important; }  /* sinon Gradio impose sa couleur */
.ligne.vide, .ligne.vide span { color: #94a3b8 !important; }
.ligne.zero .nb { color: #94a3b8 !important; }  /* « 0 faute » n'est pas une erreur : en gris */
.pastille { width: 22px; height: 14px; border: 3px solid currentColor; border-radius: 3px; }
.pastille.pointilles { border-style: dashed; }
.bleu { color: #1d4ed8; } .vert { color: #15803d; } .rouge { color: #b91c1c; }
.raison { margin: -4px 0 4px 36px; font-size: 13px; color: #475569; }
.resume { margin-top: 6px; padding-top: 10px; border-top: 1px solid #e2e8f0; font-size: 15px;
          font-weight: 600; color: #0f172a; }
.sans-faute { margin-top: 6px; background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0;
              border-radius: 10px; padding: 8px 12px; font-weight: 700; }
.note { font-size: 13px; color: #64748b; margin: 4px 0 0; }
.puce { display: inline-block; padding: 1px 8px; border-radius: 999px; background: #fef3c7;
        color: #92400e; font-size: 12px; font-weight: 600; }

/* onglet des scores sur tout le test */
.scores { display: grid; gap: 18px; max-width: 1100px; margin: 8px auto; }
.scores h2 { margin: 0; font: 700 20px/1.3 var(--font); color: #0f172a; display: flex;
             align-items: center; gap: 10px; }
.scores h3 { margin: 0 0 8px; font: 700 15px/1.3 var(--font); color: #0f172a; }
.tuiles, .indicateurs { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                        gap: 12px; }
.tuile { border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 16px; background: #fff; }
.tuile.pointilles { border-style: dashed; border-color: #fca5a5; }
.tuile .nb { font-size: 40px; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }
.tuile .libelle { font-size: 15px; font-weight: 600; color: #0f172a; }
.tuile .detail { font-size: 13px; color: #64748b; margin-top: 2px; }
.indicateur { border-left: 3px solid #2563eb; padding: 4px 12px; }
.indicateur .nb { font: 700 24px/1.2 var(--font); color: #1d4ed8; }
.indicateur .libelle { font-weight: 700; color: #0f172a; }
.indicateur .detail { font-size: 13px; color: #475569; }
.chaine { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: 14px; }
.chaine span { background: #f1f5f9; border-radius: 8px; padding: 6px 10px; }
.chaine b { color: #1d4ed8; }
.comparaison { border-collapse: collapse; width: 100%; font-size: 14px; border: none !important; }
.comparaison th, .comparaison td { padding: 8px 12px; border: none !important;
                                   border-bottom: 1px solid #e2e8f0 !important; text-align: right;
                                   background: none !important; }
.comparaison th { color: #64748b; font-weight: 600; }
.comparaison th:first-child, .comparaison td:first-child { text-align: left; }
.comparaison .nous { font-weight: 700; color: #0f172a; }
"""


def virgule(x: float, decimales: int = 2) -> str:
    """0.954 -> « 0,95 » : les nombres à la française."""
    return f"{x:.{decimales}f}".replace(".", ",")


def charger_images_test() -> tuple[pd.DataFrame, list[str]]:
    """Les vrais avions des images de test, et l'ordre d'affichage (images conseillées d'abord)."""
    jeux = pd.read_csv(SPLITS_FILE, dtype={"image_id": str}).set_index("image_id")["split"]
    avions = load_annotations()
    avions = avions[avions["image_id"].map(jeux) == "test"]
    autres = sorted(set(avions["image_id"]) - set(IMAGES_CONSEILLEES), key=int)
    return avions, [*IMAGES_CONSEILLEES, *autres]


def lire_image_bgr(image_id: str):
    return cv2.imread(str(RAW_IMAGES_DIR / f"{image_id}.jpg"))


def lire_image_rgb(image_id: str):
    return cv2.cvtColor(lire_image_bgr(image_id), cv2.COLOR_BGR2RGB)


def vignette(image_id: str):
    return cv2.resize(lire_image_rgb(image_id), (200, 200))


@dataclass
class Correction:
    """La correction OUI / NON d'une image : chaque réponse du modèle, puis les avions oubliés."""

    reponses: list[tuple[Detection, str]]  # (réponse du modèle, vrai type ; "" = pas un avion)
    oublies: list[tuple[tuple, str]]  # vrais avions sans aucune réponse : (boîte, vrai type)

    @property
    def justes(self) -> list[Detection]:
        return [d for d, vrai in self.reponses if d.class_name == vrai]

    @property
    def fausses(self) -> list[tuple[Detection, str]]:  # mauvais type, ou pas un avion
        return [(d, vrai) for d, vrai in self.reponses if d.class_name != vrai]

    @property
    def n_avions(self) -> int:
        """Vrais avions de l'image = justes + mauvais types + oubliés."""
        return sum(1 for _, vrai in self.reponses if vrai) + len(self.oublies)


def corriger(verite_image: pd.DataFrame, detections: list[Detection]) -> Correction:
    """Compare les réponses aux vraies annotations, avec la même règle que nos scores officiels."""
    boites = verite_image[["xmin", "ymin", "xmax", "ymax"]].to_numpy(float)
    types = [str(c) for c in verite_image["class_name"]]
    _, associes, manques = match_details(
        boites,
        [CLASS_TO_INDEX[c] for c in types],
        [d.box for d in detections],
        [CLASS_TO_INDEX[d.class_name] for d in detections],
        [d.score for d in detections],
    )
    reponses = [
        (d, types[t] if t is not None else "") for d, t in zip(detections, associes, strict=True)
    ]
    return Correction(reponses, [(tuple(boites[t]), types[t]) for t in manques])


def dessiner(image_rgb, etape: str, detections, correction: Correction | None):
    """L'image au centre : sans boîte, puis les réponses (bleu), puis la correction (vert / rouge)."""
    if etape == "reponse":
        dense = len(detections) > DENSE
        boites = [TaggedBox(d.box, BLEU, d.class_name) for d in detections]
        return draw_tagged_boxes(image_rgb, boites, font_size=13 if dense else 18)
    if etape == "correction":
        dense = len(correction.reponses) + len(correction.oublies) > DENSE
        justes = [
            TaggedBox(d.box, VERT, "" if dense else f"✓ {d.class_name}") for d in correction.justes
        ]
        oublies = [
            TaggedBox(boite, ROUGE, f"✗ oublié : {vrai}", dashed=True, hollow_label=True)
            for boite, vrai in correction.oublies
        ]
        fausses = [
            TaggedBox(
                d.box, ROUGE, f"✗ {d.class_name}, vrai : {vrai}" if vrai else "✗ pas un avion"
            )
            for d, vrai in correction.fausses
        ]
        # les erreurs sont dessinées en dernier : elles restent au-dessus des réponses justes
        return draw_tagged_boxes(
            image_rgb, justes + oublies + fausses, font_size=15 if dense else 18
        )
    return image_rgb


# ---------- petits morceaux de HTML ----------


def entete(n_images: int) -> str:
    return (
        "<div class='entete'><h1>✈️ Détection d'avions sur images satellites</h1>"
        f"<span class='badge'>{n_images} images de test · jamais vues par le modèle</span></div>"
    )


def titre_etape(numero: int, texte: str, portee: str = "") -> str:
    badge = f"<span class='portee'>{portee}</span>" if portee else ""
    return f"<div class='etape'><span class='num'>{numero}</span>{texte}{badge}</div>"


def titre_scene(image_id: str, etape: str) -> str:
    conseil = IMAGES_CONSEILLEES.get(image_id)
    etoile = f"<span class='etoile'>★ {conseil}</span>" if conseil else ""
    return (
        f"<div class='titre-scene'>Image {image_id} {etoile}"
        f"<span class='mode {etape}'>{ETAPES[etape]}</span></div>"
    )


def ligne(libelle: str, nombre: int | None, couleur: str, pointilles: bool = False) -> str:
    """Une ligne de bilan : pastille (même couleur et même trait que les boîtes), libellé, nombre."""
    classe = "vide" if nombre is None else f"{couleur} zero" if nombre == 0 else couleur
    pastille = "pastille pointilles" if pointilles else "pastille"
    valeur = "—" if nombre is None else nombre
    return (
        f"<div class='ligne {classe}'><span class='{pastille}'></span>"
        f"<span>{libelle}</span><span class='nb'>{valeur}</span></div>"
    )


def note(texte: str) -> str:
    return f"<p class='note'>{texte}</p>"


def raison(texte: str) -> str:
    return f"<p class='raison'>{texte}</p>"


def resume_html(detections, duree: float = 0.0, seuil: float = DEFAULT_MIN_SCORE) -> str:
    """Carte 2 : ce que le modèle a répondu."""
    if detections is None:
        return ligne("Avions trouvés", None, "bleu") + note(
            "Le modèle n'a pas encore regardé cette image."
        )
    n = len(detections)
    html = ligne("Avion trouvé" if n == 1 else "Avions trouvés", n, "bleu")
    if n:
        types = Counter(d.class_name for d in detections)
        html += raison(" · ".join(f"{k} × {c}" for c, k in types.most_common()))
    else:
        html += note(f"Aucun avion trouvé avec une confiance d'au moins {virgule(seuil)}.")
    officiel = "" if seuil == DEFAULT_MIN_SCORE else " <span class='puce'>officiel : 0,50</span>"
    return html + note(f"en {virgule(duree, 1)} s · seuil {virgule(seuil)}{officiel}")


def bilan_html(correction: Correction | None, seuil: float = DEFAULT_MIN_SCORE) -> str:
    """Carte 3 : justes, fausses (avec la raison), oubliés (avec le vrai type)."""
    if correction is None:
        return (
            ligne("Justes", None, "vert")
            + ligne("Fausses", None, "rouge")
            + ligne("Oubliés", None, "rouge", pointilles=True)
            + note("La vérité (les annotations du dataset) n'est pas encore affichée.")
        )
    justes, n = len(correction.justes), correction.n_avions
    fausses = Counter(
        f"{d.class_name} au lieu de {vrai}" if vrai else "pas un avion"
        for d, vrai in correction.fausses
    )
    oublies = Counter(vrai for _, vrai in correction.oublies)
    html = ligne("Justes", justes, "vert") + ligne("Fausses", len(correction.fausses), "rouge")
    if fausses:
        html += raison(" · ".join(t + (f" (× {k})" if k > 1 else "") for t, k in fausses.items()))
    html += ligne("Oubliés", len(correction.oublies), "rouge", pointilles=True)
    if oublies:
        html += raison("vrais types : " + ", ".join(f"{k} × {t}" for t, k in oublies.most_common()))
    avions = "avion" if n == 1 else "avions"
    if justes == n and not correction.fausses:
        html += f"<div class='sans-faute'>✓ Sans faute : {justes} {avions} sur {n}</div>"
    else:
        s = "s" if justes > 1 else ""
        html += f"<div class='resume'>{justes} avion{s} sur {n} bien reconnu{s}</div>"
    if seuil != DEFAULT_MIN_SCORE:
        html += note(
            f"Seuil modifié ({virgule(seuil)}) : ce bilan ne compte plus dans les scores officiels."
        )
    return html


def tableau(etape: str, detections, correction: Correction | None) -> pd.DataFrame:
    """Le détail boîte par boîte : les réponses du modèle, ou le verdict de chacune."""
    if etape == "correction":
        lignes = [
            ["✗ faux", d.class_name, vrai or "pas un avion", virgule(d.score)]
            for d, vrai in correction.fausses
        ]
        lignes += [["✗ oublié", "—", vrai, "—"] for _, vrai in correction.oublies]
        lignes += [
            ["✓ juste", d.class_name, d.class_name, virgule(d.score)] for d in correction.justes
        ]
        return pd.DataFrame(lignes, columns=COLONNES_CORRECTION)
    lignes = [
        [
            i + 1,
            d.class_name,
            *(virgule(s) for s in (d.score, d.detector_score, d.classifier_score)),
        ]
        for i, d in enumerate(detections or [])
    ]
    return pd.DataFrame(lignes, columns=COLONNES_REPONSE)


def scores_globaux_html() -> str:
    """Onglet 2 : le bilan de tout le jeu de test (runs/comparaison_test/metrics.json)."""
    fichier = RUNS_DIR / "comparaison_test" / "metrics.json"
    if not fichier.exists():
        return (
            "<div class='scores'><p>Scores pas encore calculés : lance <code>uv run python "
            "scripts/evaluate_systems.py --split test --je-confirme-le-test</code>.</p></div>"
        )
    m = json.loads(fichier.read_text(encoding="utf-8"))
    # « au_seuil » : la règle de l'interface (le modèle répond si sa confiance atteint 0,5)
    nous, temoin = m["deux_etages"]["au_seuil"], m["temoin_yolo20"]["au_seuil"]
    justes, mauvais, alarmes, oublies = (
        int(nous[k]) for k in ("bon_type", "mauvais_type", "fausses_alarmes", "manques")
    )
    n, reponses = justes + mauvais + oublies, justes + mauvais + alarmes

    def tuile(nb, libelle, detail, couleur, pointilles=False):
        bord = " pointilles" if pointilles else ""
        return (
            f"<div class='tuile{bord}'><div class='nb {couleur}'>{nb}</div>"
            f"<div class='libelle'>{libelle}</div><div class='detail'>{detail}</div></div>"
        )

    def indicateur(nb, libelle, detail):
        return (
            f"<div class='indicateur'><div class='nb'>{nb}</div>"
            f"<div class='libelle'>{libelle}</div><div class='detail'>{detail}</div></div>"
        )

    def comparaison(libelle, cle, fmt=str):
        return f"<tr><td>{libelle}</td><td class='nous'>{fmt(nous[cle])}</td><td>{fmt(temoin[cle])}</td></tr>"

    entier = lambda x: str(int(x))
    temps = (m["deux_etages"]["ms_par_image_cpu"], m["temoin_yolo20"]["ms_par_image_cpu"])
    return f"""<div class='scores'>
<h2>Bilan des 131 images de test <span class='portee'>Tout le test</span></h2>
<p class='note'>{n} vrais avions · le modèle répond si sa confiance atteint {virgule(nous["seuil"])}
(le réglage officiel, le même que dans « Tester une image »).</p>
<div class='tuiles'>
{tuile(justes, "justes", f"{virgule(100 * justes / n, 1)} % des avions", "vert")}
{
        tuile(
            mauvais + alarmes,
            "fausses",
            f"{mauvais} mauvais types + {alarmes} « pas un avion »",
            "rouge",
        )
    }
{tuile(oublies, "oubliés", f"{virgule(100 * oublies / n, 1)} % des avions", "rouge", True)}
</div>
<div class='indicateurs'>
{
        indicateur(
            virgule(nous["precision"], 3),
            "Précision",
            f"Quand le modèle répond, il a juste {round(100 * nous['precision'])} fois sur 100 "
            f"({justes} / {reponses}).",
        )
    }
{
        indicateur(
            virgule(nous["rappel"], 3),
            "Rappel",
            f"Il retrouve {virgule(100 * nous['rappel'], 1)} % des vrais avions, avec le bon type "
            f"({justes} / {n}).",
        )
    }
{
        indicateur(
            virgule(nous["f1"], 3),
            "F1-score",
            "Précision et rappel résumés en un seul nombre : notre score principal.",
        )
    }
{
        indicateur(
            virgule(m["deux_etages"]["mAP50"], 3),
            "mAP50",
            "La mesure standard en détection, calculée sur tous les seuils.",
        )
    }
</div>
<div><h3>Comment ça marche ?</h3><div class='chaine'>
<span>image</span> → <span><b>détecteur</b> YOLO11n : où sont les avions ?</span> →
<span><b>classifieur</b> ResNet-18 : quel type, de A1 à A20 ?</span> →
<span>confiance finale = détecteur × classifieur</span> → <span>réponse si ≥ 0,50</span>
</div></div>
<div><h3>Pourquoi deux étages ? Comparaison avec un seul YOLO à 20 classes</h3>
<table class='comparaison'>
<tr><th></th><th>Notre système</th><th>Témoin YOLO 20 classes</th></tr>
{comparaison("Justes", "bon_type", entier)}
{comparaison("Mauvais types", "mauvais_type", entier)}
{comparaison("« Pas un avion »", "fausses_alarmes", entier)}
{comparaison("Oubliés", "manques", entier)}
{comparaison("F1-score", "f1", lambda x: virgule(x, 3))}
<tr><td>Temps par image (processeur)</td><td class='nous'>{virgule(temps[0] / 1000)} s</td>
<td>{virgule(temps[1] / 1000)} s</td></tr>
</table>
<p class='note'>Le témoin oublie moins d'avions, mais se trompe de type bien plus souvent.</p></div>
<p class='note'>ⓘ Ces chiffres sont exactement la somme des corrections image par image de
l'onglet « Tester une image ».</p>
</div>"""


# ---------- l'interface ----------


def creer_demo(pipeline: AircraftPipeline | None = None) -> gr.Blocks:
    pipeline = pipeline or AircraftPipeline()  # sans niveau « à vérifier » : c'est oui ou non
    verite, ordre = charger_images_test()
    pipeline.predict(lire_image_bgr(ordre[0]))  # chauffe : le premier clic de l'oral est rapide
    galerie_vals = [(vignette(i), f"★ {i}" if i in IMAGES_CONSEILLEES else i) for i in ordre]

    def predire(image_id, seuil):
        pipeline.min_score = seuil
        debut = time.perf_counter()
        detections = sorted(pipeline.predict(lire_image_bgr(image_id)), key=lambda d: -d.score)
        return detections, time.perf_counter() - debut

    def rendu(image_id, etape, detections=None, duree=0.0, seuil=DEFAULT_MIN_SCORE):
        """Tout l'affichage, recalculé à partir de l'état : image, étape, réponses du modèle."""
        correction = None
        if etape == "correction":
            correction = corriger(verite[verite["image_id"] == image_id], detections)
        table = tableau(etape, detections, correction)
        suivant = {"image": 0, "reponse": 1, "correction": 2}[etape]  # le bouton bleu
        boutons = [gr.Button(variant="primary" if k == suivant else "secondary") for k in range(3)]
        return (
            titre_scene(image_id, etape),
            dessiner(lire_image_rgb(image_id), etape, detections, correction),
            resume_html(detections, duree, seuil),
            bilan_html(correction, seuil),
            table,
            gr.Accordion(label=f"Détail boîte par boîte ({len(table)})"),
            *boutons,
            image_id,
            etape,
            detections,
            duree,
        )

    with gr.Blocks(title="Détection d'avions · MAR20", fill_width=True) as demo:
        etat_image, etat_etape = gr.State(ordre[0]), gr.State("image")
        etat_detections, etat_duree = gr.State(None), gr.State(0.0)
        with gr.Column(elem_id="app"):
            gr.HTML(entete(len(ordre)))
            with gr.Tabs():
                with gr.Tab("Tester une image", id="tester"), gr.Row(equal_height=True):
                    # 1. CHOISIR
                    with gr.Column(
                        scale=0, min_width=300, elem_classes=["carte"], elem_id="col-choix"
                    ):
                        gr.HTML(titre_etape(1, "Choisir une image"))
                        gr.HTML("<p class='sur-titre'>Exemples conseillés</p>")
                        with gr.Column(elem_classes=["conseils"], min_width=0):
                            boutons_conseil = [
                                gr.Button(f"★ {i} · {texte}", size="sm", elem_classes=["conseil"])
                                for i, texte in IMAGES_CONSEILLEES.items()
                            ]
                        bouton_hasard = gr.Button("🎲 Image au hasard", size="md")
                        gr.HTML(
                            f"<p class='sur-titre'>Toutes les images de test ({len(ordre)})</p>"
                        )
                        galerie = gr.Gallery(
                            value=galerie_vals,
                            columns=3,
                            height="clamp(150px, calc(100vh - 540px), 520px)",
                            allow_preview=False,
                            object_fit="cover",
                            show_label=False,
                            container=False,
                            buttons=[],
                            interactive=False,
                            selected_index=0,
                            elem_id="galerie",
                        )
                    # L'IMAGE, en grand
                    with gr.Column(scale=1, min_width=480, elem_classes=["carte"]):
                        scene_titre = gr.HTML(titre_scene(ordre[0], "image"))
                        scene = gr.Image(
                            value=lire_image_rgb(ordre[0]),
                            interactive=False,
                            show_label=False,
                            container=False,
                            buttons=["fullscreen"],
                            format="png",  # pas de flou sur les traits et le texte
                            height="clamp(380px, calc(100vh - 236px), 920px)",
                            elem_id="scene",
                        )
                    # 2. LE MODÈLE RÉPOND, puis 3. ON CORRIGE
                    with gr.Column(scale=0, min_width=380, elem_id="col-droite"):
                        with gr.Column(elem_classes=["carte"], elem_id="carte-reponse"):
                            gr.HTML(titre_etape(2, "Le modèle répond"))
                            bouton_detecter = gr.Button(
                                "🔍 Détecter les avions", variant="primary", size="lg"
                            )
                            resume = gr.HTML(resume_html(None))
                            with gr.Accordion("⚙ Seuil de confiance", open=False):
                                seuil = gr.Slider(
                                    0.05,
                                    0.95,
                                    value=DEFAULT_MIN_SCORE,
                                    step=0.05,
                                    label="Le modèle répond s'il est sûr à au moins…",
                                    info="Confiance finale = détecteur × classifieur. "
                                    "0,50 : réglage officiel, choisi sur la validation.",
                                )
                        with gr.Column(elem_classes=["carte"], elem_id="carte-correction"):
                            gr.HTML(titre_etape(3, "On corrige", "Cette image"))
                            bouton_corriger = gr.Button("✅ Corriger avec la vérité", size="lg")
                            bilan = gr.HTML(bilan_html(None))
                            with gr.Accordion(
                                "Détail boîte par boîte (0)", open=False, elem_id="details"
                            ) as details:
                                table = gr.Dataframe(
                                    value=pd.DataFrame(columns=COLONNES_REPONSE),
                                    interactive=False,
                                    show_label=False,
                                    max_height=280,
                                    wrap=True,
                                )
                with gr.Tab("Scores sur tout le test", id="scores"):
                    gr.HTML(scores_globaux_html())

        sorties = [
            scene_titre,
            scene,
            resume,
            bilan,
            table,
            details,
            bouton_detecter,
            bouton_corriger,
            bouton_hasard,
            etat_image,
            etat_etape,
            etat_detections,
            etat_duree,
        ]

        def sur_vignette(seuil, evt: gr.SelectData):
            return rendu(ordre[evt.index], "image", seuil=seuil)

        def choisir(index, seuil):
            return gr.Gallery(selected_index=index), *rendu(ordre[index], "image", seuil=seuil)

        def au_hasard(image_id, seuil):
            index = random.choice([k for k, i in enumerate(ordre) if i != image_id])
            return choisir(index, seuil)

        def detecter(image_id, seuil):
            return rendu(image_id, "reponse", *predire(image_id, seuil), seuil)

        def corriger_clic(image_id, detections, duree, seuil):
            if detections is None:  # on n'a pas encore détecté : on le fait d'abord
                detections, duree = predire(image_id, seuil)
            return rendu(image_id, "correction", detections, duree, seuil)

        def changer_seuil(image_id, etape, seuil):
            if etape == "image":  # rien n'est affiché : rien à recalculer
                return [gr.skip()] * len(sorties)
            return rendu(image_id, etape, *predire(image_id, seuil), seuil)

        galerie.select(sur_vignette, [seuil], sorties, show_progress="hidden")
        for index, bouton in enumerate(boutons_conseil):
            bouton.click(
                lambda s, index=index: choisir(index, s),
                [seuil],
                [galerie, *sorties],
                show_progress="hidden",
            ).then(None, js=DEFILER)
        bouton_hasard.click(
            au_hasard, [etat_image, seuil], [galerie, *sorties], show_progress="hidden"
        ).then(None, js=DEFILER)
        bouton_detecter.click(detecter, [etat_image, seuil], sorties, show_progress="minimal")
        bouton_corriger.click(
            corriger_clic,
            [etat_image, etat_detections, etat_duree, seuil],
            sorties,
            show_progress="minimal",
        )
        seuil.release(
            changer_seuil, [etat_image, etat_etape, seuil], sorties, show_progress="minimal"
        )
    return demo
