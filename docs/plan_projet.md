# Plan du projet

Ce plan suit **dans l'ordre** les étapes demandées par le prof :

1. analyse des données ;
2. préparation des données ;
3. **détection** ;
4. **classification** ;
5. intégration ;
6. déploiement, vraiment à la fin.

Le PDF du sujet place la classification avant la détection, mais le prof a précisé en séance qu'il fallait faire **la détection d'abord**.

Il est réparti sur les 5 séances de 4 h, et a deux niveaux :
- **le cœur**, à faire absolument ;
- **les bonus**, seulement s'il reste du temps.

Une démo qui marche vaut mieux que dix idées à moitié finies.

## Ce qu'il faut savoir sur nos données

- **1 331 images** satellites, presque toutes en 800×800 px, avec **7 870 avions** annotés par une boîte (format Pascal VOC).
- **20 classes**, A1 à A20, très déséquilibrées. A6 n'a que 107 avions dans 29 images, alors que A16 en a 785.
- Une image contient de 1 à 50 avions, souvent plusieurs du même type (c'est une base aérienne).
- Les avions mesurent environ 100 px de côté (de 28 à 566 px).
- **Le prof ne fournit pas de corpus de test.** C'est à nous de mettre de côté une partie des images, notre jeu de **test** de 131 images et 788 avions, et de ne l'utiliser **qu'une seule fois, à la fin**. C'est aussi ce jeu que le modèle final annotera pour le rapport : c'est le « corpus de test annoté ».
- Ce dataset est exactement la partie « entraînement » du dataset public MAR20. **On ne télécharge jamais le reste de MAR20** : ce serait ajouter des données, ce que la FAQ interdit.

## Les choix techniques en bref

| Brique | Choix | Pourquoi |
|---|---|---|
| Langage, environnement | Python 3.11, uv | conseillé par le prof ; uv fige les versions pour que tout le monde ait le même environnement |
| Bibliothèque principale | PyTorch | utilise la RTX 4060 sous Windows, contrairement à TensorFlow (plus de GPU sous Windows depuis la 2.10) |
| Détecteur | YOLO11n (Ultralytics), 1 classe « avion » | léger (2,6 M de paramètres), rapide, pré-entraîné ; c'est le choix frugal recommandé par le cours |
| Classifieur | ResNet-18 pré-entraîné sur ImageNet | le transfer learning donne de bons résultats avec peu de données ; l'architecture s'explique sur un schéma |
| Modèles « maison » | forêt aléatoire, petit CNN | témoins chiffrés ; ils illustrent les notions de l'oral |
| Témoin | YOLO11n à 20 classes | vérifie avec des chiffres que l'approche en deux étages vaut le coup |
| Interface | Gradio | une démo web en quelques lignes |
| Déploiement | Docker (version CPU) | demandé par le prof |

## Points de vigilance (les erreurs classiques)

1. **Fuite de données.** Le découpage train / validation / test se fait **par image**, jamais par avion. Sinon, deux avions de la même image (même sol, même lumière) se retrouvent l'un dans le train et l'autre dans le test, et les scores sont artificiellement bons.
2. **Ordre des classes.** On utilise partout la constante `CLASSES` de `src/avions/constants.py`, en ordre numérique. Un tri alphabétique donnerait A1, A10, A11… et décalerait les étiquettes sans aucune erreur visible.
3. **Ne jamais évaluer un modèle sur ses données d'entraînement.** C'est un conseil du prof. Le modèle les a déjà vues : il aurait un score excellent même s'il les avait simplement apprises par cœur, c'est-à-dire en overfitting. C'est comme donner à l'examen exactement les exercices corrigés en TD. On sépare donc les données en trois :
   - le **train**, pour apprendre ;
   - la **validation**, pour régler et comparer les modèles ;
   - le **test**, utilisé **une seule fois**, à la fin, pour le score final.
4. **Augmentation adaptée à la vue du ciel.** On utilise les rotations, les retournements et de légers changements de luminosité et de contraste. On évite les déformations de perspective, qui n'existent pas sur une image satellite.
5. **Déséquilibre des classes.** On pondère la fonction de perte et on juge les modèles sur la **macro-F1**, la moyenne des F1 de chaque classe. L'accuracy seule cacherait les classes rares.

## Plan des séances

### Séance 1 : analyse des données (étape 1)
- [x] Environnement : Python 3.11, bibliothèques, Git.
- [x] **EDA** dans `notebooks/01_eda.ipynb`, en suivant la méthode du cours :
  - inspection initiale : volumétrie, format ;
  - qualité : doublons, boîtes incohérentes ;
  - analyse univariée : avions par classe, tailles des boîtes ;
  - analyse bivariée : taille × classe, classes qui apparaissent ensemble ;
  - quelques avions de chaque classe à l'écran.
- [x] **Découpage** train / validation / test (80 / 10 / 10 %) par image, en gardant les proportions de classes et les quasi-doublons ensemble, enregistré dans `data/splits/splits.csv` (décisions D7 et D8).
- [x] Poids pré-entraînés : le prof confirme qu'ils sont autorisés, et le choix est libre.

**Livrable :** le notebook d'EDA avec ses graphiques et le fichier de split.

### Séance 2 : préparation pour la détection et détection (étapes 2 et 3)
- [x] Convertir les annotations au format attendu par YOLO (`uv run python scripts/prepare_yolo.py`).
- [x] Entraîner **YOLO11n à 1 classe** « avion » **sur Google Colab** (`notebooks/colab_entrainement.ipynb`) : 30 min, arrêt automatique à l'epoch 67. Le premier essai sur le PC, coupé par deux écrans bleus, a été abandonné (voir D9 et D10).
- [x] Évaluer le détecteur **sur la validation** : précision 0,987, rappel 0,997, mAP@0.5 0,993.
- [x] Regarder les avions manqués et les fausses alarmes (D20, D21).
- [x] Entraîner **YOLO11n à 20 classes**, le témoin end-to-end : mAP@0.5 de 0,986 en validation.

**Livrable :** des images satellites sur lesquelles tous les avions sont encadrés. C'est une première démo visuelle.

### Séance 3 : vignettes, classification, puis on branche les deux (étapes 2, 4 et 5)
- [x] Générer les **vignettes** : un carré autour de chaque avion, avec de la marge (`scripts/make_crops.py`, D11).
- [x] Baseline : **forêt aléatoire** sur des caractéristiques simples, HOG et taille : macro-F1 de 0,614 en validation (D12).
- [x] **ResNet-18** en transfer learning sur les vignettes, entraîné sur Colab (`notebooks/colab_classification.ipynb`) : macro-F1 de **0,996** en validation, soit 3 erreurs sur 787 (D12).
- [x] Évaluation **sur la validation** : macro-F1 et matrice de confusion. On l'évalue aussi sur les vignettes découpées **par notre détecteur**, qui sont moins bien cadrées que celles des annotations.
- [x] Script `predict` : image → détection → vignettes → classification → image annotée (`scripts/predict.py`).
- [x] Comparaison avec le témoin sur la validation (`scripts/evaluate_systems.py`, D13) : mAP50 de 0,990 pour nos deux étages contre 0,968 pour le témoin.
- [x] S'il reste du temps (sinon en séance 4) :
  - **petit CNN maison**, entraîné de zéro, avec ses courbes d'apprentissage pour observer l'overfitting ;
  - **ResNet-18 sans pré-entraînement**, pour mesurer ce qu'apporte le transfer learning.

**Livrable :** la première démo de bout en bout. Une image en entrée donne une image annotée en sortie, avec le type de chaque avion (MVP).

### Séance 4 : intégration complète et comparaisons (étape 5)
- [x] Interface **Gradio** : on choisit une des 131 images de test (jamais vues), le modèle répond, puis on corrige chaque réponse en oui / non face à la vérité (`scripts/app.py`, D17, D18, D19).
- [x] Comparaisons restantes : petit CNN, ResNet-18 sans pré-entraînement, notre système contre le témoin.
- [x] Analyse des erreurs : classes confondues, avions manqués. Choix des seuils de confiance.
- [x] Évaluation **unique** sur le jeu de test (D15) : mAP50 de 0,957 et F1 de 0,954 pour notre système, contre 0,940 et 0,912 pour le témoin.
- [x] README à jour, code propre et testé.

**Livrable :** la version 1.0. On n'ajoute plus de fonctionnalités après.

### Séance 5 : déploiement et livraison (étape 6)
- [ ] **Docker** : image CPU qui lance l'interface ou annote un dossier d'images. Docker Desktop est à installer chez soi avant cette séance (droits administrateur, WSL2).
- [x] Annoter **notre jeu de test** (131 images jamais vues) avec le modèle final (`scripts/annotate_corpus.py`, `reports/corpus_test_annote/`). C'est le « corpus de test annoté » demandé dans le rapport.
- [x] Corpus annoté versionné dans Git ; le rapport donne le lien GitHub de `reports/corpus_test_annote/`.
- [x] Rapport (`docs/rapport.md` et `.pdf`) : architecture, choix, résultats. Il s'écrit en grande partie au fil de l'eau grâce au [journal des décisions](journal_decisions.md).
- [ ] Slides : des schémas plutôt que du code, dans l'ordre des 6 étapes.
- [ ] Vidéo de secours de la démo.
- [ ] Deux répétitions chronométrées (10 min), avec un temps de parole équilibré.
- [ ] Réviser les fiches d'oral ([oral/fiches_oral.md](../oral/fiches_oral.md)).

## Métriques

| Ce qu'on évalue | Métriques |
|---|---|
| Classifieur (vignettes) | accuracy, **macro-F1**, précision / rappel / F1 par classe, matrice de confusion |
| Détecteur (avion ou pas) | précision, **rappel** (un avion manqué est grave), mAP@0.5 |
| Pipeline complet | mAP@0.5 sur les 20 classes : notre système contre le témoin |
| Frugalité (cours) | nombre de paramètres, temps de traitement d'une image (CPU et GPU) |

Ordre de grandeur publié sur ce même découpage de MAR20 : 86 à 90 % de mAP@0.5 avec des YOLO de taille moyenne (Adli et al., 2024).

## Bonus (seulement si le cœur est fini)

- **Explicabilité** (Grad-CAM) : une carte montre la zone de la vignette regardée par le réseau. C'est la réponse au besoin de « modèles explicables » vu en cours.
- **Humain dans la boucle** : sous un certain niveau de confiance, l'avion est marqué « à vérifier » au lieu d'être classé au hasard.
- **Pistes d'amélioration du score.** Chacune est mesurée sur la validation et n'est gardée que si elle aide. L'analyse des erreurs dit laquelle essayer en premier.
  - vignettes légèrement décalées à l'entraînement, pour imiter les boîtes du détecteur ;
  - vote par image : la plupart des images ne contiennent qu'un seul type d'avion ;
  - TTA : moyenne des prédictions sur 4 rotations de la vignette ;
  - backbone plus gros (ResNet-50, YOLO11s) ;
  - ensemble de classifieurs ;
  - réentraînement final sur train + validation ;
  - comparaison avec Faster R-CNN ou YOLO26.
- Export ONNX et mesure de la consommation d'énergie (CodeCarbon).

## Répartition dans le binôme (proposition)

- **Étudiant 1 :** EDA, découpage, détection, branchement des deux étages.
- **Étudiant 2 :** vignettes, classifieurs, métriques, interface, Docker.

Règle d'or : chacun doit pouvoir réexpliquer la partie de l'autre. Le prof interroge les deux.
