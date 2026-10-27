# Détection et classification d'avions sur images satellites

Projet du module « Introduction à l'IA » (IPSA Toulouse). L'application **localise les avions** dans une image satellite et **détermine leur type**, parmi 20 codés A1 à A20.

## Le rendu

| Demandé par le sujet | Où le trouver |
|---|---|
| Le rapport : architecture, lien Git, lien du corpus | [docs/rapport.md](docs/rapport.md) (et [docs/rapport.pdf](docs/rapport.pdf), joint à l'e-mail) |
| Le code du projet | ce dépôt : [`src/avions/`](src/avions/) et [`scripts/`](scripts/) |
| Le corpus de test annoté par le modèle | [reports/corpus_test_annote/](reports/corpus_test_annote/) : 131 XML, aperçus, tableau, notice |
| Une démonstration qui fonctionne | `uv run python scripts/app.py` (voir [Lancer la démo](#lancer-la-démo)) |

Pour comprendre et présenter le projet, tout est dans le dossier [oral/](oral/) (voir [oral/LISEZ-MOI.md](oral/LISEZ-MOI.md)) :
- [oral/guide_projet.html](oral/guide_projet.html) : le projet étape par étape, avec les figures, les 12 notions de l'oral et le code fichier par fichier. GitHub n'affiche que son code : l'ouvrir dans un navigateur après avoir cloné ou téléchargé le dépôt (Code > Download ZIP) ;
- [docs/architecture.md](docs/architecture.md) : l'architecture en détail ;
- [docs/journal_decisions.md](docs/journal_decisions.md) : chaque choix et sa justification (D1 à D21) ;
- [oral/fiches_oral.md](oral/fiches_oral.md) : les fiches de révision des 12 notions ;
- `oral/lancer_demo.bat` et `oral/ouvrir_le_code.bat` : un double-clic pour lancer la démo, ou pour ouvrir le code à montrer dans VS Code.

## L'architecture, en deux étages

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
                                              │ learning) « quel type    │
                                              │ d'avion ? » → A1…A20     │
                                              └────────────┬─────────────┘
                                                           ▼
                        confiance finale = détecteur × classifieur, réponse si ≥ 0,5
```

Un détecteur YOLO à 20 classes, qui fait tout en une seule passe, sert de **témoin** pour vérifier que l'approche en deux étages est meilleure.

## Les résultats (jeu de test : 131 images jamais vues, 788 avions)

| Système | mAP50 | Précision | Rappel | F1 | Fausses alarmes | Mauvais types |
|---|---|---|---|---|---|---|
| **Deux étages (le nôtre)** | **0,957** | **0,984** | 0,925 | **0,954** | **3** | **9** |
| Témoin YOLO à 20 classes | 0,940 | 0,913 | 0,910 | 0,912 | 19 | 49 |

La mAP50 est mesurée sur tous les seuils de confiance ; les autres colonnes le sont au seuil de 0,5 de l'application.

Sur nos 3 fausses alarmes, 2 sont des doublons (deux boîtes sur un même avion), et la 3e est un vrai avion que le dataset n'a pas annoté. Le détail et l'analyse des erreurs sont dans le [rapport](docs/rapport.md).

## Lancer la démo

**1. Installer** [uv](https://docs.astral.sh/uv/), puis, à la racine du projet :

```bash
uv sync
```

Cette commande crée l'environnement (Python 3.11) avec toutes les bibliothèques.

**2. Placer le dataset fourni par l'enseignant** (le sous-ensemble « train » de MAR20, 425 Mo, pas dans Git). Copier le contenu du dossier `student-dataset/` de l'archive, de sorte d'obtenir :
- les images dans `data/raw/images/*.jpg` ;
- les annotations dans `data/raw/annotations/*.xml`.

Les deux modèles finaux sont déjà dans le dépôt (`models/`).

**3. Lancer l'interface :**

```bash
uv run python scripts/app.py
```

La page s'ouvre dans le navigateur. Elle propose les **131 images de test**, que le modèle n'a jamais vues, et s'ouvre sur l'exemple ★ 3803.
- **🔍 Détecter les avions** : le modèle répond par des boîtes bleues, là où sa confiance finale atteint le seuil (0,5 par défaut).
- **✅ Corriger avec la vérité** : chaque réponse est 🟩 **juste** ou 🟥 **fausse**, avec la raison (« vrai : A1 », « doublon » ou « aucun avion annoté »), et chaque vrai avion sans réponse est **oublié** (rouge pointillé).
- En haut, le **F1 cumulé** additionne les avions des images corrigées. Après les 131 images, il vaut exactement 0,954.
- Le curseur change le seuil ; il faut ensuite recliquer sur un bouton. Une image corrigée à un autre seuil que 0,5 ne compte pas dans le F1 cumulé.

Pour une seule image, en ligne de commande : `uv run python scripts/predict.py data/raw/images/1453.jpg`.

Pour vérifier que tout fonctionne : `uv run pytest` (31 tests).

## L'organisation du dépôt

| Dossier | Contenu |
|---|---|
| `src/avions/` | le code du projet, sous forme de paquet Python : lecture des données, découpage, vignettes, classifieurs, système en deux étages (`pipeline.py`), évaluation, interface (`app.py`) |
| `scripts/` | les commandes à lancer : préparer les données, entraîner, évaluer, annoter le corpus, lancer l'interface |
| `configs/` | les réglages de chaque entraînement |
| `notebooks/` | l'analyse des données (`01_eda.ipynb`) et les notebooks d'entraînement sur Colab |
| `tests/` | les tests automatiques |
| `data/splits/` | le découpage entraînement / validation / test (le dataset lui-même n'est pas dans Git) |
| `models/` | les deux modèles finaux : le détecteur et le classifieur |
| `runs/` | les traces des entraînements et des évaluations : courbes, métriques, matrices de confusion |
| `reports/` | le corpus de test annoté et les figures de l'analyse des données |
| `docs/` | le rapport, l'architecture, le journal des décisions, le plan |
| `oral/` | la préparation de l'oral : le guide (et `guide_images/`, des copies allégées des figures de l'EDA), les fiches, les raccourcis de la démo et du code |

## Les étapes du projet

Elles suivent la démarche du prof. Il nous a demandé en séance de faire la détection avant la classification.

| Étape | Ce qu'on a fait | Code | Résultats |
|---|---|---|---|
| 1. Analyse des données | EDA : qualité, types, tailles, quasi-doublons, corrélations | `notebooks/01_eda.ipynb` | `reports/figures/eda/` |
| 2. Préparation | découpage 80 / 10 / 10 par image, format YOLO, vignettes | `scripts/make_splits.py`, `scripts/prepare_yolo.py`, `scripts/make_crops.py` | `data/splits/splits.csv` |
| 3. Détection | YOLO11n à 1 classe « avion » (et le témoin à 20 classes), entraînés sur Colab | `configs/detecteur_*.yaml`, `scripts/train_detector.py`, `notebooks/colab_entrainement.ipynb` | `runs/detecteur_*/` |
| 4. Classification | forêt aléatoire, petit CNN, ResNet-18 avec et sans pré-entraînement | `scripts/train_baseline.py`, `scripts/train_classifier.py`, `scripts/evaluate_classifiers.py`, `configs/classifieur_*.yaml`, `notebooks/colab_classification.ipynb` | `runs/baseline_rf/`, `runs/classifieur_*/`, `runs/classifieurs_val/`, `runs/classifieurs_test/` |
| 5. Intégration | le système en deux étages, l'évaluation finale, le corpus annoté, l'interface | `src/avions/pipeline.py`, `scripts/evaluate_systems.py`, `scripts/annotate_corpus.py`, `scripts/app.py` | `runs/comparaison_*/`, `reports/corpus_test_annote/` |
| 6. Déploiement | conteneur Docker | *(à venir)* | |

## Reproduire les résultats

```bash
uv run python scripts/make_splits.py       # découpage (déjà fourni : data/splits/splits.csv)
uv run python scripts/prepare_yolo.py      # dataset au format YOLO
uv run python scripts/make_crops.py        # vignettes pour les classifieurs
uv run python scripts/train_baseline.py    # forêt aléatoire (sur le processeur)
# détecteurs et réseaux : notebooks/colab_entrainement.ipynb et colab_classification.ipynb (sur GPU)
uv run python scripts/evaluate_systems.py      # les deux systèmes, sur la validation
uv run python scripts/evaluate_classifiers.py  # les classifieurs, sur la validation
# évaluation finale sur le test, faite avec les réglages figés (runs/comparaison_test/, runs/classifieurs_test/)
uv run python scripts/evaluate_systems.py --split test --je-confirme-le-test
uv run python scripts/evaluate_classifiers.py --split test --je-confirme-le-test
uv run python scripts/annotate_corpus.py       # corpus de test annoté
```

Seuls les deux modèles de la démo sont dans Git. Les évaluations ont aussi besoin du témoin (`models/detecteur_20classes.pt`) et des modèles de comparaison (`baseline_rf.joblib`, `classifieur_smallcnn.pt`, `classifieur_resnet18_sans_pretrain.pt`, 90 Mo) : ils se réentraînent avec les commandes ci-dessus, ou sont disponibles sur demande.

La carte graphique du PC plantait pendant les longs entraînements (décision D10). Ils ont donc été faits sur **Google Colab**. Le dossier `Mon Drive/projet_avions` reprend la même arborescence que ce projet, et `uv run python scripts/sync_drive.py` synchronise les deux dans les deux sens (le plus récent gagne).

Avant de publier des notebooks passés par Colab, lancer `uv run python scripts/nettoyer_notebooks.py`. Ce script retire les informations personnelles que Colab y enregistre.

## Licence

Copyright © 2026 mathis derory. Ce projet est distribué sous licence **GNU AGPL-3.0** (fichier [LICENSE](LICENSE)).

Ce choix est imposé par [Ultralytics](https://github.com/ultralytics/ultralytics), la bibliothèque des détecteurs YOLO, elle-même sous licence AGPL-3.0 : un projet qui l'utilise doit être publié, code source compris, sous la même licence.
