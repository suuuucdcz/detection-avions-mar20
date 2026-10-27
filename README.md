# Détection et classification d'avions sur images satellites

Projet du module « Introduction à l'IA » (IPSA Toulouse). L'application doit **localiser les avions** dans une image satellite et **déterminer leur modèle** (20 types, codés A1 à A20).

## Architecture (en deux étages)

```
image satellite
     │
     ▼
┌──────────────────────┐   boîtes « avion »   ┌──────────────────────────┐
│ 1. DÉTECTEUR         │ ───────────────────► │ découpe d'une vignette   │
│ YOLO11n, 1 classe    │                      │ autour de chaque avion   │
│ « où sont les avions │                      └────────────┬─────────────┘
│   dans l'image ? »   │                                   ▼
└──────────────────────┘                      ┌──────────────────────────┐
                                              │ 2. CLASSIFIEUR           │
                                              │ ResNet-18 (transfer      │
                                              │ learning) « quel modèle  │
                                              │ d'avion ? » → A1…A20     │
                                              └────────────┬─────────────┘
                                                           ▼
                                     image annotée + fichier XML (Pascal VOC)
```

Un détecteur YOLO à 20 classes, qui fait tout en une seule passe, sert de **témoin** pour vérifier que l'approche en deux étages est vraiment meilleure.

Pour aller plus loin :
- [docs/architecture.md](docs/architecture.md) : l'architecture complète, étape par étape, avec les résultats ;
- [docs/journal_decisions.md](docs/journal_decisions.md) : la justification de chaque choix ;
- [docs/fiches_oral.md](docs/fiches_oral.md) : les 12 notions théoriques de l'oral, appliquées au projet.

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/). L'application et l'évaluation tournent sur le processeur. Les entraînements se font sur un GPU, par exemple sur Google Colab.

```bash
uv sync                      # crée .venv avec Python 3.11 et installe toutes les bibliothèques
uv run pytest                # lance les tests
```

### Ce qui n'est pas dans ce dépôt (trop lourd)

| Élément | Taille | Où le placer |
|---|---|---|
| Le dataset fourni par l'enseignant (sous-ensemble « train » de MAR20) | 425 Mo | `data/raw/images/*.jpg` et `data/raw/annotations/*.xml` |
| Les modèles entraînés (`detecteur_1classe.pt`, `classifieur_resnet18.pt`, etc.) | 139 Mo | `models/` |
| Les images du corpus de test annoté | 68 Mo | `reports/corpus_test_annote/images/` et `apercus/` |

Les modèles et le corpus complet sont disponibles sur demande et via le lien indiqué dans le rapport. Sans les modèles, on peut tout réentraîner (voir ci-dessous).

### Reproduire les résultats

```bash
uv run python scripts/make_splits.py       # découpage (déjà fourni : data/splits/splits.csv)
uv run python scripts/prepare_yolo.py      # dataset au format YOLO
uv run python scripts/make_crops.py        # vignettes pour les classifieurs
uv run python scripts/train_baseline.py    # forêt aléatoire (processeur)
# Détecteurs et réseaux : notebooks/colab_entrainement.ipynb et colab_classification.ipynb (GPU)
uv run python scripts/evaluate_systems.py  # comparaison sur la validation
uv run python scripts/evaluate_classifiers.py
```

## Les grandes étapes et où les trouver

Les étapes suivent la démarche du prof. Les choix et leurs raisons sont dans [docs/journal_decisions.md](docs/journal_decisions.md) (D1 à D10).

| Étape | Ce qu'on a fait | Code | Résultats |
|---|---|---|---|
| 1. Analyse des données | EDA : qualité, classes, tailles, doublons | `notebooks/01_eda.ipynb` | figures dans `reports/figures/eda/`, conclusions à la fin du notebook |
| 2. Préparation | découpage 80 / 10 / 10 par image ; conversion au format YOLO | `scripts/make_splits.py`, `scripts/prepare_yolo.py` | `data/splits/splits.csv` (`data/yolo/` est régénéré à chaque fois) |
| 3. Détection | YOLO11n à 1 classe « avion », entraîné sur Colab : rappel 0,997 en validation | `configs/detecteur_1classe.yaml`, `scripts/train_detector.py`, `notebooks/colab_entrainement.ipynb` | `models/detecteur_1classe.pt`, courbes dans `runs/detecteur_1classe/` |
| 3 bis. Témoin | YOLO11n à 20 classes, pour comparer : mAP@0.5 de 0,986 en validation | `configs/detecteur_20classes.yaml` | `models/detecteur_20classes.pt`, `runs/detecteur_20classes/` |
| 4. Classification | vignettes ; forêt aléatoire (macro-F1 de 0,614) ; **ResNet-18 pré-entraîné : macro-F1 de 0,996** ; sans pré-entraînement : 0,973 ; petit CNN : 0,690 | `scripts/make_crops.py`, `scripts/train_baseline.py`, `scripts/train_classifier.py`, `configs/classifieur_*.yaml`, `notebooks/colab_classification.ipynb` | `runs/baseline_rf/`, `runs/classifieur_*/`, `models/` |
| 5. Intégration | chaîne complète détecteur → vignettes → classifieur : mAP50 de 0,990 en validation, contre 0,968 pour le témoin ; avions « à vérifier » dans le corpus annoté (D14), interface Gradio (D17 à D19) | `src/avions/pipeline.py`, `scripts/predict.py`, `scripts/evaluate_systems.py` | `runs/comparaison_val/`, `reports/figures/predictions/`, `scripts/app.py` |
| 6. Déploiement | Docker *(séance 5)* | *(à venir)* | *(à venir)* |

## Résultats finaux (jeu de test : 131 images jamais vues, 788 avions)

| Système | mAP50 | Précision | Rappel | F1 | Fausses alarmes | Mauvais types |
|---|---|---|---|---|---|---|
| **Deux étages (le nôtre), seuil 0,5** | **0,957** | **0,984** | 0,925 | **0,954** | **3** | **9** |
| Témoin YOLO à 20 classes, seuil 0,5 | 0,940 | 0,913 | 0,910 | 0,912 | 19 | 49 |

La mAP50 est mesurée sur tous les seuils. Avec les avions « à vérifier », 785 avions sur 788 sont présentés.

**Corpus de test annoté par le modèle :** `reports/corpus_test_annote/`, avec les XML Pascal VOC, les aperçus et un `LISEZMOI.md`. Il est généré par `uv run python scripts/annotate_corpus.py`. Le détail et l'analyse sont dans [docs/journal_decisions.md](docs/journal_decisions.md), D15.

## Lancer la démo

```bash
uv run python scripts/app.py
```

La page s'ouvre dans le navigateur (http://127.0.0.1:7860). Comme le demande le sujet, aucune image extérieure n'est ajoutée : l'interface propose **les 131 images de test**, que le modèle n'a jamais vues pendant son entraînement. L'écran se lit de gauche à droite, en trois colonnes :
1. **Choisir une image** (à gauche) : un des cinq exemples conseillés (★), une vignette de la galerie, ou « 🎲 Image au hasard ». Elle s'affiche en grand au centre.
2. **🔍 Détecter les avions** : le modèle répond par des boîtes bleues, là où sa confiance finale atteint 0,5, comme pour nos scores officiels.
3. **✅ Corriger avec la vérité** : c'est **oui ou non**. Chaque réponse devient 🟩 **juste** ou 🟥 **fausse** (la raison est écrite : « A19, vrai : A1 », « pas un avion »), et chaque vrai avion sans réponse est **oublié** (rouge pointillé). Le bilan de l'image sert de légende ; le détail est dans « Détail boîte par boîte ».

Additionnés sur les 131 images, ces bilans redonnent exactement les scores officiels, présentés dans le second onglet, « Scores sur tout le test ». Le seuil de confiance se règle dans « ⚙ Seuil de confiance ». Pour l'oral sur un projecteur 1366 × 768, passer le navigateur en plein écran (F11).

Pour une seule image en ligne de commande : `uv run python scripts/predict.py data/raw/images/1453.jpg`.

## Google Drive et Colab

La carte graphique du PC plante pendant les longs entraînements (voir D10). Les calculs lourds se font donc sur **Google Colab**.

Le dossier **`Mon Drive/projet_avions`** est une **copie exacte de ce projet**, avec les mêmes dossiers, grâce à l'application « Google Drive pour ordinateur ».

1. Sur le PC, `uv run python scripts/sync_drive.py` synchronise le projet et le Drive. Pour chaque fichier, la version la plus récente gagne.
2. Dans Colab, ouvrir depuis le Drive le notebook de l'étape voulue, choisir un GPU, puis exécuter les cellules. Les résultats sont écrits dans `runs/` et `models/` sur le Drive.
   - `notebooks/colab_entrainement.ipynb` : les détecteurs ;
   - `notebooks/colab_classification.ipynb` : les classifieurs.
3. Sur le PC, relancer `uv run python scripts/sync_drive.py` pour ramener ces résultats dans le projet.

## Organisation du dépôt

| Dossier | Contenu |
|---|---|
| `src/avions/` | le code du projet, sous forme de paquet Python réutilisable |
| `notebooks/` | exploration et analyse (EDA, analyse des erreurs) |
| `scripts/` | les commandes à lancer (préparer les données, entraîner, annoter) |
| `configs/` | les paramètres (hyperparamètres, seuils) |
| `tests/` | les tests automatiques (`uv run pytest`) |
| `data/` | les données (ignorées par Git, sauf les fichiers de split) |
| `models/` | les poids entraînés (ignorés par Git) |
| `reports/` | figures et métriques pour le rapport et l'oral |
| `docs/` | plan du projet, architecture, journal des décisions, fiches de révision |

## Avancement

- [x] Séance 1 : installation de l'environnement
- [x] Séance 1 : analyse exploratoire des données (`notebooks/01_eda.ipynb`) et découpage train / validation / test (`uv run python scripts/make_splits.py`)
- [x] Séance 2 : détection des avions (le détecteur et le témoin, entraînés sur Colab)
- [x] Séance 3 : vignettes, classification, première démo de bout en bout et comparaison avec le témoin
- [x] Séance 4 : interface graphique, comparaisons, évaluation finale sur le test (mAP50 de 0,957, F1 de 0,954)
- [ ] Séance 5 : déploiement (Docker), annotation de notre jeu de test par le modèle final, rapport, répétitions

Le détail est dans [docs/plan_projet.md](docs/plan_projet.md).

## Licence

Copyright © 2026 mathis derory. Ce projet est distribué sous licence **GNU AGPL-3.0** (fichier [LICENSE](LICENSE)).

Ce choix est imposé par [Ultralytics](https://github.com/ultralytics/ultralytics), la bibliothèque des détecteurs YOLO, elle-même sous licence AGPL-3.0 : un projet qui l'utilise doit être publié, code source compris, sous la même licence.

Avant de publier des notebooks passés par Colab, lancer `uv run python scripts/nettoyer_notebooks.py`. Ce script retire les informations personnelles que Colab y enregistre (nom et identifiant du compte Google).
