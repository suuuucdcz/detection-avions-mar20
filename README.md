# Détection et classification d'avions sur images satellites

Projet du module « Introduction à l'IA » (IPSA Toulouse). L'application **localise les avions** dans une image satellite et **détermine leur type**, parmi 20 codés A1 à A20.

## Le contenu du dépôt

| Quoi | Où |
|---|---|
| Le code, un notebook par étape, avec ses résultats | [etapes/](etapes/) |
| L'interface de démonstration | [app.py](app.py) (voir [Lancer la démo](#lancer-la-démo)) |
| Le corpus de test annoté par le modèle | [reports/corpus_test_annote/](reports/corpus_test_annote/) : 131 XML, aperçus, tableau, notice |

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

Sur nos 3 fausses alarmes, 2 sont des doublons (deux boîtes sur un même avion), et la 3e est un vrai avion que le dataset n'a pas annoté. Le choix du seuil et le résultat final sont dans le notebook [5_integration.ipynb](etapes/5_integration.ipynb).

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
uv run python app.py
```

La page s'ouvre dans le navigateur. Elle propose les **131 images de test**, que le modèle n'a jamais vues, et s'ouvre sur l'exemple ★ 3803.
- **🔍 Détecter les avions** : le modèle répond par des boîtes bleues, là où sa confiance finale atteint le seuil (0,5 par défaut).
- **✅ Corriger avec la vérité** : chaque réponse est 🟩 **juste** ou 🟥 **fausse**, avec la raison (« vrai : A1 », « doublon » ou « aucun avion annoté »), et chaque vrai avion sans réponse est **oublié** (rouge pointillé).
- En haut, le **F1 cumulé** additionne les avions des images corrigées. Après les 131 images, il vaut exactement 0,954.
- Le curseur change le seuil ; il faut ensuite recliquer sur un bouton. Une image corrigée à un autre seuil que 0,5 ne compte pas dans le F1 cumulé.

## L'organisation du dépôt

| Dossier ou fichier | Contenu |
|---|---|
| `etapes/` | le projet en 5 notebooks, un par étape du sujet : tout le code utilisé, dans l'ordre, déjà exécuté avec ses résultats |
| `app.py` | l'interface de la démo, en un seul fichier (la même chaîne en deux étages que le notebook 5) |
| `models/` | les deux modèles finaux : le détecteur et le classifieur |
| `data/splits/` | le découpage entraînement / validation / test (le dataset lui-même n'est pas dans Git) |
| `runs/` | les courbes d'entraînement des deux modèles finaux, et les scores des évaluations (y compris ceux des modèles écartés) |
| `reports/corpus_test_annote/` | le corpus de test annoté par le modèle |

## Les étapes du projet

Elles suivent la démarche du prof. Il nous a demandé en séance de faire la détection avant la classification.

| Étape | Ce qu'on a fait | Notebook | Résultats |
|---|---|---|---|
| 1. Analyse des données | EDA : qualité, types, tailles, quasi-doublons, corrélations | `etapes/1_analyse_des_donnees.ipynb` | les figures, dans le notebook |
| 2. Préparation | découpage 80 / 10 / 10 par image, format YOLO, vignettes | `etapes/2_preparation_des_donnees.ipynb` | `data/splits/splits.csv` |
| 3. Détection | YOLO11n à 1 classe « avion », entraîné sur Colab | `etapes/3_detection.ipynb` | `runs/detecteur_1classe/` |
| 4. Classification | ResNet-18 pré-entraîné, comparé à une forêt aléatoire, un petit CNN et un ResNet-18 sans pré-entraînement | `etapes/4_classification.ipynb` | `runs/classifieur_resnet18/`, `runs/classifieurs_val/`, `runs/classifieurs_test/` |
| 5. Intégration | le système en deux étages, le choix du seuil, l'évaluation finale (avec le témoin à 20 classes), le corpus annoté, l'interface | `etapes/5_integration.ipynb` et `app.py` | `runs/comparaison_val/`, `runs/comparaison_test/`, `reports/corpus_test_annote/` |
| 6. Déploiement | conteneur Docker | *(à venir)* | |

## Reproduire les résultats

Ouvrir les notebooks d'`etapes/` dans l'ordre (VS Code ou Jupyter, avec l'environnement du projet) et les exécuter. Tout tourne sur le processeur : les notebooks rechargent les modèles entraînés de `models/` et les mesurent à nouveau sur la validation. Le test, mesuré une seule fois à la fin, n'est jamais recalculé : son résultat est relu dans `runs/comparaison_test/`.

La carte graphique du PC plantait pendant les longs entraînements. Ils ont donc été faits sur **Google Colab**. Pour les relancer, la partie « Pour relancer sur Google Colab », en tête des notebooks 2, 3 et 4, explique la marche à suivre ; il suffit ensuite de mettre `ENTRAINER = True` dans les notebooks 3 et 4.

## Licence

Copyright © 2026 mathis derory. Ce projet est distribué sous licence **GNU AGPL-3.0** (fichier [LICENSE](LICENSE)).

Ce choix est imposé par [Ultralytics](https://github.com/ultralytics/ultralytics), la bibliothèque des détecteurs YOLO, elle-même sous licence AGPL-3.0 : un projet qui l'utilise doit être publié, code source compris, sous la même licence.
