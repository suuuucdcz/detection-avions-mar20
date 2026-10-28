# Le projet, étape par étape

Un notebook par étape demandée par le prof. Chaque notebook contient **tout son code**, dans l'ordre, avec les explications et les résultats déjà affichés. On peut le lire sans ouvrir aucun autre fichier du projet.

| Notebook | Étape | Ce qu'il fait |
|---|---|---|
| `1_analyse_des_donnees.ipynb` | Analyse des données | qualité des données, types d'avions, tailles, quasi-doublons, corrélations |
| `2_preparation_des_donnees.ipynb` | Préparation des données | découpage train / validation / test, format YOLO, vignettes |
| `3_detection.ipynb` | Détection | YOLO11n à une seule classe « avion » |
| `4_classification.ipynb` | Classification | ResNet-18 pré-entraîné, sur les vignettes |
| `5_integration.ipynb` | Intégration | le système en deux étages, le choix du seuil, le résultat final, l'interface |

Le prof a demandé en séance de faire **la détection avant la classification** : c'est l'ordre suivi ici. La 6e étape, le déploiement avec Docker, n'est pas encore faite.

## Pour les relancer

- Ouvrir le notebook depuis ce dossier, dans VS Code ou Jupyter, avec l'environnement du projet (`.venv`).
- Le dataset du prof doit être dans `data/raw/images/` et `data/raw/annotations/`.
- Les lancer dans l'ordre : l'étape 2 produit les fichiers dont les étapes 3 et 4 ont besoin.
- Tout tourne sur le processeur. Les cellules d'entraînement sont désactivées (`ENTRAINER = False`) : les entraînements ont été faits sur Google Colab, car la carte graphique du PC plantait. Les notebooks utilisent donc les modèles entraînés du dossier `models/`.
- Pour réentraîner : la partie « Pour relancer sur Google Colab », en tête des notebooks 2, 3 et 4, explique comment faire sur Colab, puis mettre `ENTRAINER = True`.
- Le test n'est jamais recalculé : il a été mesuré une seule fois, à la fin, et son résultat est relu dans `runs/`.

## Et le reste du dépôt ?

`app.py`, à la racine, contient l'interface complète de la démo : `uv run python app.py`. Il reprend la même chaîne en deux étages que le notebook 5.
