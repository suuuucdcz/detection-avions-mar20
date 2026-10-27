# Journal des décisions

Chaque choix technique important est noté ici avec les options envisagées et la raison du choix. Ce journal sert à rédiger le rapport et à répondre à « Pourquoi avez-vous choisi X ? » à l'oral.

Format : **contexte → options → choix → pourquoi → preuve chiffrée** (à compléter quand on a les résultats).

---

## D1 : architecture en deux étages, avec un témoin (26/10/2026)

**Contexte :** il faut trouver les avions dans l'image (**détection**) et dire de quel modèle il s'agit (**classification**).

**Options**
- A. Deux étages : un détecteur à 1 classe « avion », puis un classifieur à 20 classes sur une vignette découpée autour de chaque avion.
- B. Un seul détecteur à 20 classes (end-to-end) qui fait tout d'un coup.

**Choix :** A comme système principal, et B entraîné comme **témoin** pour comparer.

**Pourquoi**
- C'est la démarche conseillée par le prof : vignettes, puis classification, puis détection.
- Le détecteur apprend une seule notion (« avion ») à partir des 7 870 exemples. Il n'est donc pas gêné par le déséquilibre entre classes.
- Le classifieur voit chaque avion en gros plan, et on peut le faire tourner librement pendant l'entraînement (augmentation).
- On peut évaluer chaque étage séparément. Une erreur vient-elle d'un avion manqué ou d'un avion mal reconnu ?

**Preuve chiffrée :** mAP@0.5 du système A contre celui du système B, sur le même jeu de test. *(à compléter en séance 4)*

Le témoin B (YOLO11n à 20 classes, entraîné sur Colab) obtient déjà, **en validation**, une précision de 0,978, un rappel de 0,966, une **mAP@0.5 de 0,986** et une mAP@0.5:0.95 de 0,817 (meilleure epoch : 69 sur 80). C'est un témoin très fort : notre architecture en deux étages devra faire au moins aussi bien pour justifier sa complexité.

**Résultat en validation (D13), avec le même code d'évaluation et une seule classe par avion :** mAP50 de **0,990 pour nos deux étages** contre **0,968 pour le témoin**, et F1 de 0,988 contre 0,967. En contrepartie, le témoin est près de 4 fois plus rapide. La confirmation se fera sur le test, à la fin.

---

## D2 : PyTorch + Ultralytics plutôt que TensorFlow / Keras (26/10/2026)

**Contexte :** le prof recommande TensorFlow, Keras, PyTorch, scikit-learn et OpenCV. Nos entraînements se font sur une RTX 4060 (8 Go) sous Windows.

**Options :** TensorFlow / Keras, ou PyTorch.

**Choix :** PyTorch, avec Ultralytics pour les détecteurs YOLO. On garde scikit-learn pour les modèles classiques et les métriques, et OpenCV pour les images.

**Pourquoi**
- TensorFlow n'utilise plus la carte graphique sous Windows depuis la version 2.10. Il faudrait passer par WSL2, ou entraîner 10 à 50 fois plus lentement.
- PyTorch utilise CUDA directement sous Windows.
- Ultralytics fournit des détecteurs YOLO pré-entraînés, faciles à réentraîner sur nos données.

**Conséquence :** Ultralytics est sous licence AGPL-3.0. Notre dépôt Git doit donc être public et publié sous la même licence, ce qui convient à un projet d'école.

---

## D3 : Python 3.11 et uv (26/10/2026)

**Choix :** Python 3.11, comme le conseille le prof. L'environnement est géré par **uv**.

**Pourquoi**
- uv installe la bonne version de Python et toutes les bibliothèques en une commande (`uv sync`).
- Il fige les versions exactes dans `uv.lock` : le binôme, Docker et le prof obtiennent exactement le même environnement (reproductibilité).
- Pour que torch utilise la carte graphique, uv télécharge la version CUDA 13.2 de PyTorch sous Windows, et la version CPU sous Linux (Docker). La configuration est dans `pyproject.toml`.

---

## D4 : organisation des données (26/10/2026)

**Choix :** le dataset est rangé dans `data/raw/`, qui est ignoré par Git. Le sujet `main.pdf` est rangé dans `docs/`, ignoré lui aussi : c'est un document du prof, à ne pas publier.

**Pourquoi**
- L'ancien chemin contenait un espace (`student-dataset 2`), source fréquente de bugs.
- Les données sont trop lourdes pour Git (425 Mo). Seuls les fichiers de split, qui sont petits et essentiels à la reproductibilité, seront versionnés.

---

## D5 : transfer learning plutôt qu'un modèle créé de zéro (26/10/2026)

**Contexte :** faut-il créer notre propre réseau ou partir d'un modèle pré-entraîné ?
- La FAQ interdit d'ajouter des données, mais autorise le transfer learning.
- **Le prof a confirmé que les poids pré-entraînés sont autorisés** et que le choix des poids nous revient.

**Choix**
- Les modèles finaux partent de poids pré-entraînés : YOLO11n sur COCO, ResNet-18 sur ImageNet.
- On crée aussi nos propres modèles (forêt aléatoire, petit CNN) comme **témoins**.

**Pourquoi**
- Avec environ 1 000 images d'entraînement, un réseau partant de zéro risque l'overfitting : il apprend par cœur au lieu de généraliser.
- Un réseau pré-entraîné sait déjà « voir » les bords, les formes et les textures. On ne lui réapprend que la fin.
- C'est aussi le choix frugal recommandé dans le cours : réutiliser plutôt que tout réentraîner.

**Pourquoi ces poids-là**
- **ImageNet** pour le classifieur : 1,2 million de photos de 1 000 catégories. C'est la référence pour apprendre à un réseau à « voir » en général.
- **COCO** pour le détecteur : 80 catégories d'objets, avec leurs boîtes. Le réseau sait donc déjà localiser des objets, et COCO contient même une classe « airplane », même si ce sont surtout des photos prises du sol.

**Preuve chiffrée :** macro-F1 de ResNet-18 pré-entraîné contre ResNet-18 sans pré-entraînement, contre le petit CNN maison et contre la forêt aléatoire. *(à compléter en séance 2 ou 4)*

---

## D6 : ordre de construction, détection puis classification (26/10/2026)

**Contexte :** le PDF du sujet liste la classification avant la détection. En séance, le prof a précisé qu'il fallait faire **la détection d'abord**, puis la classification. Le déploiement vient vraiment à la fin.

**Choix :** on suit l'ordre donné par le prof.
- Séance 1 : analyse des données et découpage.
- Séance 2 : détection.
- Séance 3 : vignettes, classification, puis on branche les deux.
- Séance 5 : Docker.

**Pourquoi** *(nos hypothèses, à confirmer en demandant au prof)*
- **C'est l'ordre dans lequel l'application travaille.** Sur une nouvelle image, il n'y a pas d'annotation : il faut d'abord trouver les avions avant de pouvoir les découper et les classer.
- **Le classifieur doit être testé dans les conditions réelles.** En production, il ne recevra pas des vignettes parfaites découpées à partir des annotations, mais celles produites par le détecteur, qui sont parfois décalées, trop grandes ou trop petites. Avec le détecteur déjà prêt, on peut évaluer le classifieur sur ces vignettes réelles et l'entraîner à y résister.
- **C'est la brique la plus risquée.** Elle est la plus longue à entraîner et la plus difficile à évaluer (boîtes, IoU, mAP). La faire tôt laisse le temps de corriger.
- **Démo visuelle rapide.** Dès la séance 2, on peut montrer des images où les avions sont encadrés.

**Conséquence :** la première démo de bout en bout, avec le type de chaque avion, arrive en fin de séance 3.

---

## D7 : découpage 80 / 10 / 10, par image et stratifié (26/10/2026)

**Contexte :** on ne doit jamais évaluer un modèle sur ses données d'entraînement. On sépare donc les 1 331 images en trois jeux :
- **train** : pour apprendre ;
- **validation** : pour comparer les modèles et régler les paramètres ;
- **test** : utilisé une seule fois, à la fin.

**Options :** 70 / 15 / 15 ou 80 / 10 / 10.

**Choix :** **80 / 10 / 10**.

**Pourquoi**
- Avec seulement 1 331 images, chaque image d'entraînement compte : 80 / 10 / 10 donne 15 % d'images d'entraînement en plus.
- *(Correction du 26/10/2026 : le prof ne fournit **pas** de corpus de test. Notre jeu de test est donc la **seule** évaluation finale, et c'est lui que le modèle annotera pour le rapport. Avec 788 avions, il suffit pour les scores globaux. Les scores des classes rares restent imprécis, on les présentera avec leurs effectifs.)*
- La contrepartie : moins d'exemples pour évaluer. On l'a compensée en vérifiant que **chaque classe a au moins 3 images en validation et en test**.

**Comment**
- Découpage **par image** : tous les avions d'une image vont dans le même jeu. Les avions d'une même image partagent le décor, la lumière et souvent le type ; les séparer serait une fuite de données.
- **Stratifié** : chaque classe a environ 10 % de ses avions en validation et 10 % en test.
- Outil : `StratifiedGroupKFold` de scikit-learn, avec 10 paquets (1 pour le test, 1 pour la validation, 8 pour le train).
- On a essayé 50 graines aléatoires et gardé la meilleure.
- Code : `src/avions/data/split.py`, lancé par `scripts/make_splits.py`.

**Résultat :**

| Jeu | Images | Avions |
|---|---|---|
| train | 1 070 (80 %) | 6 295 |
| validation | 130 (10 %) | 787 |
| test | 131 (10 %) | 788 |

Pour chaque classe, la part en validation ou en test s'écarte d'au plus 1,2 point de 10 %.

**Limite connue :** les classes rares n'ont que 3 ou 4 images en validation et en test (A6, A9, A15). Leurs scores seront donc peu précis.

---

## D8 : les quasi-doublons restent dans le même jeu (26/10/2026)

**Contexte :** l'EDA (partie 2.4) a montré 3 paires d'images quasi identiques :
- 2286 / 2787 : identiques ;
- 2050 / 2967 : même scène ;
- 593 / 602 : même base, à une autre date.

**Choix :** on calcule une empreinte visuelle (dHash, 256 bits) pour chaque image. Les images à **80 bits ou moins** les unes des autres forment un **groupe**, et un groupe n'est jamais coupé entre deux jeux. On obtient 15 groupes, de 3 images au plus.

**Pourquoi ce seuil :** les vraies paires étaient à 46, 72 et 73 bits, et on garde une marge. Regrouper par erreur deux images différentes ne coûte presque rien. Séparer deux quasi-doublons, en revanche, fausserait l'évaluation.

**Limite :** deux vues très décalées d'une même base ne sont pas repérées par cette empreinte.

**Vérification :** des tests automatiques (`tests/test_split.py`) contrôlent que les 3 paires sont dans le même jeu, qu'aucun groupe n'est coupé et que chaque classe est présente dans chaque jeu.

---

## D9 : réglages du détecteur YOLO11n (26/10/2026)

**Contexte :** l'étage 1 doit trouver tous les avions. Un avion manqué ne pourra jamais être classé, donc le **rappel** (la part des avions trouvés) est la priorité.

**Choix** (fichier `configs/detecteur_1classe.yaml`)

| Réglage | Valeur | Pourquoi |
|---|---|---|
| Modèle | YOLO11n pré-entraîné sur COCO | léger et rapide (frugalité) ; transfer learning |
| `single_cls` | true | une seule classe « avion », car le type sera donné par le classifieur |
| `imgsz` | 800 | résolution native : à 640 px, les plus petits avions passeraient de 38 à 30 px (EDA) |
| `fliplr`, `flipud` | 0,5 | vu du ciel, il n'y a ni haut ni bas (EDA : aucune orientation privilégiée) |
| `degrees` | 0 | une rotation quelconque agrandirait les boîtes horizontales |
| `patience` | 15 | early stopping : on arrête quand la validation ne progresse plus, ce qui limite l'overfitting |

**Données :** les annotations VOC sont converties au format YOLO par `scripts/prepare_yolo.py`. On a un fichier texte par image, avec une ligne `classe x_centre y_centre largeur hauteur` par avion, en valeurs entre 0 et 1. Les 20 classes sont gardées dans les fichiers, et c'est `single_cls` qui les fusionne. Le même dataset sert donc aussi au témoin à 20 classes (`configs/detecteur_20classes.yaml`, mêmes réglages).

**Preuve chiffrée** (entraînement propre sur Colab, GPU T4, validation de 130 images et 787 avions)

| Précision | Rappel | mAP@0.5 | mAP@0.5:0.95 | Meilleure epoch | Durée |
|---|---|---|---|---|---|
| 0,987 | **0,997** | 0,993 | 0,807 | 52 | 30 min |

L'early stopping a arrêté l'entraînement à l'epoch 67 : il n'y avait plus de progrès depuis 15 epochs. L'objectif de rappel ≥ 0,95 est largement atteint.

**Premier essai sur le PC, abandonné** (validation, 130 images, 787 avions, meilleure epoch : 74)

| Précision | Rappel | mAP@0.5 | mAP@0.5:0.95 | Temps par image (CPU) |
|---|---|---|---|---|
| 0,987 | **0,997** | 0,994 | 0,819 | 88 ms |

L'objectif de rappel ≥ 0,95 est largement atteint : pas besoin de YOLO11s. Trouver « un avion » est la partie facile ; la difficulté du projet est dans la classification des 20 types.

**Incident (voir D10) :** l'entraînement a provoqué deux écrans bleus (code 0x113, pilote graphique NVIDIA), aux epochs 22 et 74. Il a été repris une fois avec `--resume` et `batch=8`. Les 6 dernières epochs ont été abandonnées : le modèle ne progressait presque plus.

**Décision : on repart de zéro.** Un entraînement coupé deux fois, repris avec un autre batch et arrêté avant la fin n'est pas un résultat propre ni facile à défendre. Ses poids et ses courbes ont été mis à la corbeille. Les deux détecteurs (1 classe et témoin 20 classes) sont réentraînés **entièrement sur Colab**, dans les mêmes conditions et avec les réglages de `configs/`. Les chiffres ci-dessus montrent seulement l'ordre de grandeur attendu.

**Plan B si le rappel en validation est inférieur à 0,95 :** passer à YOLO11s, en changeant `model: yolo11s.pt`.

---

## D10 : entraînements sur Google Colab, le reste sur le PC (26/10/2026)

**Contexte :** la carte graphique du PC (RTX 4060 Laptop) a fait **deux écrans bleus identiques** (code 0x113, erreur fatale du pilote graphique NVIDIA) pendant l'entraînement du détecteur, même en réduisant la charge (`batch=8`) et avec le PC branché.

**Options**
- Continuer sur la carte locale : risque de nouveaux plantages, avec perte de travail.
- Tout faire sur le processeur : trop lent pour entraîner (plusieurs heures par modèle).
- Entraîner sur **Google Colab** : GPU T4 gratuit chez Google.

**Choix :**
- **Colab pour les entraînements uniquement.**
- **Le PC pour tout le reste** : code, EDA, tests, et utilisation des modèles (inférence) sur le processeur. Le détecteur y prend 88 ms par image, ce qui suffit largement pour la démo.

**Comment**
- `scripts/prepare_colab.py` crée deux fichiers à déposer sur Google Drive (dossier `projet_avions`) :
  - `code_projet.zip` : le code, les configs et le découpage ;
  - `donnees_avions.zip` : les images et les annotations.
- `notebooks/colab_entrainement.ipynb` décompresse le tout, installe Ultralytics dans la **même version** que sur le PC, puis lance **les mêmes scripts** avec les **mêmes configs**. Les résultats sont donc comparables.
- Les sorties sont écrites directement sur le Drive. Si Colab se déconnecte, l'entraînement reprend avec `--resume`.
- Le fonctionnement a été vérifié sur le PC en simulant Colab sur le processeur : décompression dans un dossier vide, préparation, mini-entraînement.

**Limite :** Colab impose sa propre version de Python, pas forcément la 3.11. Ce n'est pas un problème : seules les bibliothèques d'entraînement y tournent, dans les mêmes versions.

**Mise à jour : envoi automatique avec Google Drive pour ordinateur.** Le connecteur Google Drive de Claude et l'extension Chrome ne pouvaient pas envoyer le fichier de données de 440 Mo. On utilise donc l'application « Google Drive pour ordinateur » : le Drive apparaît sur le PC comme un dossier (`G:\Mon Drive`).
- `scripts/prepare_colab.py` y copie directement le code, les données et le notebook, dans `Mon Drive/projet_avions`.
- `scripts/recuperer_modeles.py` rapatrie les modèles entraînés par Colab dans `models/` et les courbes dans `runs/`.
- On n'a plus à faire de dépôt ni de téléchargement à la main.

**Mise à jour 2 : le Drive devient une copie exacte du projet.** Plutôt que des fichiers zip, le dossier `Mon Drive/projet_avions` reprend **la même arborescence que le projet sur le PC**. Chaque étape (données, code, figures, modèles, documents) se retrouve au même endroit des deux côtés.
- `scripts/sync_drive.py` remplace `prepare_colab.py` et `recuperer_modeles.py`. Pour chaque fichier, la version la plus récente est copiée de l'autre côté, et les suppressions ne sont jamais recopiées.
- Ne sont pas synchronisés : `.venv`, `.git`, les caches, `data/yolo` (régénéré) et les dossiers `weights/` des entraînements.
- Colab copie le projet depuis le Drive sur son disque, plus rapide. Il écrit ses résultats directement dans `runs/` et `models/` sur le Drive.
- Le fonctionnement a été vérifié en simulant les cellules Colab sur le processeur, à partir du Drive.

---

## D11 : les vignettes (26/10/2026)

**Contexte :** le classifieur ne regarde pas l'image entière, mais une **vignette** : un carré découpé autour d'un avion.

**Choix** (`src/avions/data/crops.py`, `scripts/make_crops.py`)
- **Un carré de 2 × le plus grand côté de la boîte**, centré sur l'avion. On garde un peu de contexte, et assez de marge pour faire tourner la vignette sans faire apparaître de coins vides.
- **Une bordure grise neutre** complète le carré quand il dépasse de l'image (avions au bord).
- **Taille de 256 × 256 pixels** : l'avion y mesure environ 128 pixels, à peu près sa taille d'origine (médiane de 111 px selon l'EDA).
- **Une seule fonction `make_crop`**, pour l'entraînement (boîtes annotées) comme pour l'application (boîtes du détecteur). Le classifieur voit donc le même genre d'images dans les deux cas.
- On obtient 7 870 vignettes (6 295 train, 787 validation, 788 test), régénérées en moins d'une minute. Elles ne sont ni dans Git ni sur le Drive.

---

## D12 : les classifieurs comparés (26/10/2026)

**Contexte :** l'étage 2 doit dire quel type d'avion (A1 à A20) se trouve dans chaque vignette. Tous les modèles sont évalués sur la **validation**, avec la **macro-F1** comme métrique principale.

**Les modèles**

| Modèle | Idée | Où |
|---|---|---|
| Forêt aléatoire | ML classique : 300 arbres de décision sur des caractéristiques faites à la main (HOG = forme des contours, plus la taille de la boîte) | `scripts/train_baseline.py` |
| **ResNet-18 pré-entraîné** | **classifieur principal**, transfer learning depuis ImageNet | `configs/classifieur_resnet18.yaml` |
| ResNet-18 sans pré-entraînement | mesure l'apport du transfer learning | `configs/classifieur_resnet18_sans_pretrain.yaml` |
| Petit CNN maison | témoin simple, pour observer l'overfitting | `configs/classifieur_smallcnn.yaml` |

**La recette des réseaux** (`scripts/train_classifier.py`)
- Data augmentation : rotation à n'importe quel angle, retournements, cadrage un peu décalé qui imite les boîtes du détecteur, luminosité et contraste ±20 %.
- Perte pondérée en 1/√effectif, pour donner plus de poids aux classes rares.
- 3 premières epochs avec le backbone gelé : seule la nouvelle couche finale apprend. Ensuite, tout le réseau apprend.
- Early stopping sur la macro-F1 de validation.

**Résultats en validation**

| Modèle | Accuracy | Macro-F1 |
|---|---|---|
| Forêt aléatoire | 0,693 | 0,614 |
| **ResNet-18 pré-entraîné** | **0,996** | **0,996** |
| ResNet-18 sans pré-entraînement | 0,973 | 0,973 |
| Petit CNN maison | 0,751 | 0,690 |

Les réseaux ont été entraînés sur Colab. Le ResNet-18 pré-entraîné atteint sa meilleure epoch à la 9e et s'arrête à la 14e (2 min). Sans pré-entraînement, le même réseau atteint sa meilleure epoch à la 29e et s'arrête à la 37e (5 min). Le petit CNN atteint sa meilleure epoch à la 20e et s'arrête à la 28e (4 min).

**Ce que montrent ces résultats**
- **Transfer learning** : à réseau identique, partir des poids ImageNet **divise le taux d'erreur par 7** (0,4 % au lieu de 2,7 %). Le réseau atteint aussi son meilleur score 3 fois plus vite (epoch 9 au lieu de 29).
- **Deep learning contre ML classique** : le CNN qui apprend lui-même ses caractéristiques (0,996) écrase la forêt aléatoire sur HOG (0,614).
- **Overfitting et underfitting**, les deux cas sont illustrés :
  - la forêt aléatoire fait de l'**overfitting** : 100 % sur l'entraînement, 69 % en validation ;
  - le petit CNN fait plutôt de l'**underfitting** : sa loss reste élevée aussi bien en entraînement (1,50) qu'en validation (1,51). Il est trop simple pour bien distinguer les 20 types ;
  - le ResNet-18 pré-entraîné ne montre aucun écart : loss de 0,73 en entraînement comme en validation.

**Les 3 erreurs du ResNet-18** (sur 787 vignettes)

| Vignette | Vraie classe | Prédiction | Confiance |
|---|---|---|---|
| image 1453, avion 7 | A1 | A19 | 0,56 |
| image 1453, avion 10 | A1 | A19 | 0,49 |
| image 2350, avion 0 | A15 | A13 | 0,55 |

Ce sont des chasseurs qui se ressemblent (déjà remarqué dans l'EDA). Le réseau hésite justement sur ces cas : confiance d'environ 0,5, contre 0,89 en médiane quand il a raison. Deux pistes en découlent :
- marquer « à vérifier » les prédictions peu sûres (humain dans la boucle) ;
- le vote par image, puisque 2 des 3 erreurs sont dans la même image.

**Limite à dire à l'oral :** la validation et le test viennent des mêmes bases aériennes que l'entraînement, ce qui donne des scores probablement optimistes. Sur des bases jamais vues, ils seraient sans doute plus bas. À titre de comparaison, les publications sur le test officiel de MAR20 donnent environ 0,86 à 0,90 de mAP@0.5. On ne peut pas le vérifier sans données externes, que la FAQ interdit.

**Observation : un overfitting typique.** La forêt aléatoire obtient 100 % sur ses propres données d'entraînement, mais 69 % en validation : ses arbres ont appris les vignettes par cœur. Elle reconnaît mal les classes rares (F1 de 0 pour A11, 0,27 pour A6).

---

## D13 : comparaison du système complet et du témoin (26/10/2026)

**Contexte :** notre architecture en deux étages (D1) doit prouver qu'elle fait au moins aussi bien que le témoin, un YOLO à 20 classes qui fait tout en un seul passage.

**Le système complet** (`src/avions/pipeline.py`)

```
image → détecteur 1 classe → boîtes → make_crop (comme à l'entraînement) → ResNet-18 → type
```

La confiance finale d'un avion vaut la confiance du détecteur × celle du classifieur. Tout tourne sur le processeur. La démo `scripts/predict.py` produit une image annotée.

**Méthode de comparaison** (`src/avions/evaluation.py`, `scripts/evaluate_systems.py`)
- Les deux systèmes passent par **le même code d'évaluation**, sur la **validation** (130 images, 787 avions).
- On calcule la mAP50 et la mAP50-95, puis la précision, le rappel et la F1 au meilleur seuil.
- **Une seule classe par avion**, comme dans l'application réelle.
- **Vérification du code :** sur le détecteur à 1 classe, il redonne la mAP50 calculée par YOLO (0,9933 contre 0,9936).
- **Pourquoi le témoin perd des points ici :** pendant son entraînement, YOLO affichait une mAP50 de 0,986. Sa propre validation autorise **plusieurs classes par boîte** : quand il hésite entre A1 et A19, il propose les deux et obtient le point. Avec une seule classe par avion, il obtient 0,968, quel que soit le seuil de confiance. C'est cette règle-là qu'on applique aux deux systèmes.

**Résultats en validation**

| Système | mAP50 | mAP50-95 | Précision | Rappel | F1 | Temps par image (CPU) |
|---|---|---|---|---|---|---|
| **Deux étages (le nôtre)** | **0,990** | 0,805 | **0,984** | **0,993** | **0,988** | 212 ms |
| Témoin YOLO à 20 classes | 0,968 | **0,806** | 0,969 | 0,965 | 0,967 | **72 ms** |

**Écarts par classe (AP50)**
- Notre système fait nettement mieux sur **A11** (0,995 contre 0,785), ainsi que sur A18, A19 et A4.
- Le témoin fait mieux sur **A15** (0,995 contre 0,945).
- Détail complet : `runs/comparaison_val/ap50_par_classe.csv`.

**Conclusion**
- **Notre architecture en deux étages reconnaît mieux les types d'avions** : +2,2 points de mAP50 et +2,1 points de F1. Le gain vient surtout des classes rares ou difficiles (A11), ce qui est cohérent avec D1 : le classifieur voit chaque avion en gros plan.
- **Les boîtes sont aussi justes d'un côté que de l'autre** (mAP50-95 de 0,805 contre 0,806).
- **Le témoin est près de 4 fois plus rapide** (72 ms contre environ 210 ms par image) : c'est le **compromis précision / frugalité**.
- L'écart reste modéré, et il est mesuré sur seulement 130 images. On le confirmera **une seule fois** sur le test, à la fin du projet.

---

## D14 : seuil de l'application, faux positifs et avions « à vérifier » (26/10/2026)

**Contexte :** on nous a conseillé de faire très attention aux **faux positifs**. Avec le premier réglage de l'application (seuil de 0,25 sur la seule confiance du détecteur), la validation comptait **54 fausses alarmes** pour 787 avions.

**Vérifications préalables**
- Aucune image n'est commune aux jeux train, validation et test.
- Aucun groupe de quasi-doublons n'est coupé entre deux jeux.
- YOLO et les classifieurs n'ont appris que sur le train.
- Tous les scores sont mesurés sur la validation, et le test n'a jamais été utilisé.
- Nuance : la validation a servi à choisir la meilleure epoch et le seuil. Ses scores sont donc légèrement optimistes, d'où le test gardé intact pour la fin.

**Analyse sur la validation** (décompte au seuil, une classe par avion)

| Règle | Bon type | Mauvais type | Fausses alarmes | Manqués | F1 |
|---|---|---|---|---|---|
| Détecteur ≥ 0,25 (ancien réglage) | 781 | 6 | 54 | 0 | — |
| Confiance finale ≥ 0,25 | 784 | 3 | 16 | 0 | 0,986 |
| Confiance finale ≥ 0,4 | 779 | 2 | 5 | 6 | 0,990 |
| **Confiance finale ≥ 0,5 (choisi)** | **767** | **1** | **4** | 19 | **0,984** |
| Confiance finale ≥ 0,6 | 748 | 0 | 3 | 39 | 0,973 |

Ce tableau a été recalculé avec le détecteur propre (D16), sauf la première ligne, mesurée avant l'incident.
| Témoin YOLO à 20 classes ≥ 0,5 | 769 | 15 | 19 | 3 | 0,967 |

**Choix**
- On filtre sur la **confiance finale**, celle du détecteur × celle du classifieur, et non sur celle du détecteur seul. Le classifieur sert ainsi de **second filtre** : une boîte sur le fond ou sur un bout d'avion ne ressemble à aucun type. Résultat, mesuré avant l'incident D16 : 54 → 14 fausses alarmes, sans perdre un seul avion bien classé.
- **Seuil de 0,5** : choisi parce qu'il avait la meilleure F1 (à égalité avec 0,4) avec le moins de faux positifs. Recalculé avec le détecteur propre (D16), 0,4 donne une F1 un peu meilleure (0,990 contre 0,984). Mais 0,5 reste le seuil **figé avant le test**, qui donne 4 fausses alarmes et 1 mauvais type pour 787 avions. Le témoin, au même seuil, fait 19 fausses alarmes et 15 mauvais types.
- **Deux niveaux dans l'application** (humain dans la boucle) :
  - les avions **sûrs** ont une confiance finale ≥ 0,5. C'est le résultat automatique, celui qu'on évalue ;
  - les avions **à vérifier** sont ceux où le détecteur est sûr de voir un avion (≥ 0,6) mais où la confiance finale reste sous 0,5, le plus souvent parce que le type est incertain. Ils sont montrés à part, marqués « ? », au lieu d'être cachés.
  - Exemple de l'image 1453 : 8 avions sûrs (0 fausse alarme), plus 4 à vérifier qui sont tous de vrais avions (des A1 et A19 qui se ressemblent). Les 12 avions sont donc présentés.
- Le décompte des faux positifs fait désormais partie de `scripts/evaluate_systems.py`, et il sera donc aussi produit pour l'évaluation finale sur le test.
- Temps de calcul au réglage de l'application, sur le processeur : environ 200 ms par image pour nos deux étages, 72 ms pour le témoin.

---

## D15 : évaluation finale sur le jeu de test (26/10/2026)

**Engagement pris AVANT de regarder le test.** Tous les choix ont été faits sur la validation et sont figés :
- détecteur `models/detecteur_1classe.pt` et classifieur `models/classifieur_resnet18.pt` ;
- seuil de confiance finale de 0,5 et seuil « à vérifier » de 0,6 (D14) ;
- témoin `models/detecteur_20classes.pt`, et comparaison des classifieurs avec les modèles existants.

Le jeu de test (131 images, 788 avions, jamais vu par aucun modèle) est évalué **une seule fois**. Aucun réglage ne sera modifié après coup pour améliorer ce score.

### Résultats sur le test (131 images, 788 avions, jamais vus)

**Système complet** (`runs/comparaison_test/`)

| Système | mAP50 | mAP50-95 | Précision | Rappel | F1 (meilleur seuil) |
|---|---|---|---|---|---|
| **Deux étages (le nôtre)** | **0,957** | **0,770** | **0,967** | **0,949** | **0,955** |
| Témoin YOLO à 20 classes | 0,940 | 0,758 | 0,935 | 0,920 | 0,921 |

**Au seuil de l'application (0,5)**

| Système | Bon type | Mauvais type | Fausses alarmes | Manqués | Précision | Rappel | F1 |
|---|---|---|---|---|---|---|---|
| **Deux étages** | 729 | **9** | **3** | 50 | **0,984** | 0,925 | **0,954** |
| Témoin | 717 | 49 | 19 | 22 | 0,913 | 0,910 | 0,912 |

**Avions « à vérifier »** (mesure descriptive, aucun réglage modifié) : 61 boîtes, dont 47 vrais avions et 14 fausses alarmes. Au total, **785 avions sur 788** sont présentés, sûrs ou à vérifier. Seulement 3 sont complètement manqués.

**Classifieurs seuls**, sur les vignettes de test (`runs/classifieurs_test/`)

| Modèle | Accuracy | Macro-F1 | Erreurs (sur 788) |
|---|---|---|---|
| Forêt aléatoire | 0,670 | 0,625 | 260 |
| **ResNet-18 pré-entraîné** | **0,962** | **0,958** | **30** |
| ResNet-18 sans pré-entraînement | 0,944 | 0,949 | 44 |
| Petit CNN maison | 0,637 | 0,567 | 286 |

**Analyse**
1. **Notre architecture en deux étages reste meilleure que le témoin sur des images jamais vues.**
   - mAP50 de 0,957 contre 0,940 ;
   - F1 au seuil de l'application de 0,954 contre 0,912 ;
   - **5 fois moins de mauvais types** (9 contre 49) et **plus de 6 fois moins de fausses alarmes** (3 contre 19).
   - En contrepartie, au seuil de 0,5, elle « manque » plus d'avions (50 contre 22). Mais 47 de ces 50 avions apparaissent quand même dans l'application, dans la catégorie « à vérifier ».
2. **Les scores baissent par rapport à la validation, comme prévu** : mAP50 de 0,990 → 0,957, macro-F1 du classifieur de 0,996 → 0,958. Deux raisons :
   - la validation avait servi à choisir les epochs et le seuil ;
   - elle ne compte que 130 images.

   C'est exactement pour ça qu'on avait gardé le test intact. **Ces chiffres du test sont les vrais scores du projet.**
3. **L'apport du transfer learning est plus modeste que ne le laissait croire la validation** : 30 erreurs contre 44 sur le test, soit environ 30 % d'erreurs en moins, au lieu d'un rapport de 1 à 7 en validation. Il reste le meilleur choix : meilleur score, et apprentissage 3 fois plus rapide.
4. **Overfitting et underfitting confirmés** : la forêt aléatoire (0,625) et le petit CNN (0,567) restent loin derrière.
5. **Classes les plus difficiles pour le classifieur** :
   - A1, rappel de 0,87, confondu avec A19 ;
   - A11, rappel de 0,80 ;
   - A15, précision de 0,77 : d'autres avions sont pris pour des A15 ;
   - A5 et A13.

   Le témoin, lui, réussit mieux sur A11 et A15 (AP50 de 0,97 et 0,99, contre 0,80 et 0,88 pour nous), mais nettement moins bien sur A16 (0,72 contre 0,94).

**Conformément à l'engagement pris, aucun réglage n'a été modifié après cette évaluation.**

### Corpus de test annoté par le modèle (livrable du rapport)

`scripts/annotate_corpus.py` annote les 131 images de test avec le modèle final. Le résultat est dans `reports/corpus_test_annote/` :
- `annotations/` : un XML Pascal VOC par image, au même format que le dataset, avec en plus `<score>` et `<a_verifier>` ;
- `images/` : les images d'origine ;
- `apercus/` : les boîtes dessinées (épaisses : sûr ; fines « ? » : à vérifier) ;
- `detections.csv` : toutes les détections ;
- `LISEZMOI.md` : description, réglages et scores.

On trouve **741 avions sûrs et 61 à vérifier**.

**Vérification :** en relisant les XML eux-mêmes et en les comparant à la vérité, on retrouve exactement les scores de l'évaluation finale (729 bon type, 9 mauvais, 3 fausses alarmes, 50 manqués).

Les images (68 Mo) sont sur le Drive, d'où le lien partagé dans le rapport, mais pas dans Git.

---

## D16 : incident, le détecteur avait été écrasé (26/10/2026)

**Ce qui s'est passé**
- Sur Colab, la cellule 7 bis (« reprendre un entraînement interrompu ») a été exécutée alors que l'entraînement du détecteur était **déjà terminé**.
- Il n'y avait rien à reprendre. YOLO a donc lancé **un nouvel entraînement avec ses réglages par défaut** : 100 epochs sur `coco8.yaml`, 8 images de démonstration **extérieures au projet**, en partant de notre détecteur.
- Notre script a ensuite copié ce modèle sur `models/detecteur_1classe.pt`, et la synchronisation l'a ramené sur le PC.

**Comment on l'a découvert :** au moment du premier commit, en vérifiant un dossier inattendu, `runs/train`. Le fichier `models/detecteur_1classe.pt` indiquait `data: coco8.yaml` au lieu de notre `dataset.yaml`.

**Conséquences**
- Les mesures faites entre 21 h 13 et la découverte ont utilisé ce détecteur altéré : comparaison en validation (D13), seuils (D14), **première évaluation sur le test** (D15) et corpus annoté.
- Ses performances étaient presque identiques, car 8 images changent très peu un modèle déjà entraîné.
- Mais ce n'était **pas le modèle figé** et il avait vu des données externes, ce que la FAQ interdit.

**Correction**
1. **Restauration du détecteur propre** : les poids d'origine (`runs/detecteur_1classe/weights/best.pt` sur le Drive, entraîné sur notre `dataset.yaml`) ont été recopiés dans `models/`. Le dossier `runs/train` a été mis à la corbeille.
2. **Correction de la cause** dans `scripts/train_detector.py` :
   - `--resume` **refuse** désormais un entraînement terminé ou absent ;
   - le script refuse aussi de copier dans `models/` un entraînement dont le nom ne correspond pas.

   La cellule 7 bis du notebook porte un avertissement.
3. **Mesures refaites avec le détecteur propre, sans changer aucun réglage** :
   - validation (D13, D14) ;
   - test (D15), avec les mêmes choix figés : détecteur propre, classifieur ResNet-18, seuil de 0,5 ;
   - corpus annoté.

   Les chiffres de D13 à D15 ont été mis à jour. Les classifieurs ne dépendent pas du détecteur : leurs résultats sur le test ne changent pas.

**Résultats corrigés sur le test :** mAP50 de 0,957 au lieu de 0,956. Au seuil de 0,5, on obtient 729 bon type, 9 mauvais, **3 fausses alarmes**, 50 manqués, et une F1 de 0,954 au lieu de 0,948. Le témoin est inchangé, avec une F1 de 0,912.

**Ce qu'on n'a PAS fait :** recalculé sur la validation avec le détecteur propre, le seuil de 0,4 donne une F1 un peu meilleure (0,990 contre 0,984). Mais changer le seuil après avoir vu le test briserait notre engagement. **Le seuil reste à 0,5.** Le seuil de 0,4 est noté comme amélioration possible pour une version future, qu'il faudrait évaluer sur de nouvelles images.

**Leçon pour l'oral :** une erreur de manipulation peut fausser un résultat sans que les chiffres paraissent anormaux. Ce sont les vérifications systématiques qui l'ont révélée : relire ce qui est enregistré, et vérifier l'origine de chaque modèle.

---

## D17 : une interface pour tester le modèle « en direct » sur des images jamais vues (26/10/2026)

**Contexte :** le sujet demande de travailler uniquement avec les images fournies par le prof, sans en ajouter. La première interface permettait de déposer n'importe quelle photo.

**Choix** (`src/avions/app.py`)
- **Plus de dépôt de photo.** On choisit une image parmi celles **que le modèle n'a jamais vues pendant son entraînement** : les 130 images de validation et les 131 images de test, par une liste déroulante ou au hasard. La liste est construite à partir du fichier de découpage. Un test automatique (`tests/test_app.py`) vérifie qu'aucune image d'entraînement n'est proposée.
- **Bouton « Analyser »** : les prédictions du modèle, avec les avions sûrs et les avions à vérifier.
- **Bouton « Vérifier »** : on compare aux vraies annotations, **boîte par boîte**, avec la même règle que pour nos scores (`match_details` dans `src/avions/evaluation.py`) :
  - vert : bon type ;
  - orange : mauvais type (par exemple « A19 (vrai A1) ») ;
  - rouge : fausse alarme ;
  - violet : avion manqué.

  Un bilan chiffré et un tableau des erreurs accompagnent l'image.

**Intérêt pour l'oral :** la démonstration montre honnêtement où le modèle se trompe, sur des images qu'il n'a jamais vues. Sur l'image 1453, on obtient 0 fausse alarme et 0 avion manqué, et les seules erreurs sont des A1 pris pour des A19, deux types qui se ressemblent beaucoup. C'est l'occasion d'expliquer la matrice de confusion, la précision et le rappel.

Les images de test ont servi une seule fois à l'évaluation finale (D15). Les regarder dans l'interface ne modifie aucun réglage.

## D18 : une interface plus simple et plus claire (26/10/2026)

**Contexte :** pour la démonstration à l'oral, l'interface de D17 était trop chargée (deux jeux d'images, une liste déroulante, beaucoup de texte).

**Choix** (`src/avions/app.py`)
- **Seulement les 131 images de test.** Les images de validation ont servi à régler le seuil : les montrer n'apporte rien de plus, et un seul jeu est plus simple à expliquer (« des images que le modèle n'a jamais vues, et qui n'ont servi à aucun réglage »). Le test automatique vérifie toujours qu'aucune autre image n'est proposée.
- **Une galerie de vignettes** au lieu d'une liste déroulante, avec **quatre images conseillées** en tête (★) : 883 (sans faute), 3607 (50 avions serrés), 3803 (confusions A1 / A19) et 1128 (le modèle hésite).
- **Deux gros boutons** : « 🔍 Détecter les avions » puis « ✅ Comparer avec la vérité ». Le second lance la détection tout seul si on l'a oubliée.
- **Des cartes chiffrées** à la place des longs textes : en haut, les scores officiels sur le test (F1, précision, fausses alarmes, mauvais types) ; sous chaque image, le bilan de l'image.
- Les réglages (seuil, avions à vérifier) et les tableaux détaillés sont rangés dans des panneaux repliables : ils restent accessibles sans encombrer l'écran.

## D19 : la vérité en oui / non, et un écran qui se lit de gauche à droite (27/10/2026)

**Contexte :**
- Le niveau « à vérifier » troublait la démonstration. Puisqu'on travaille sur le dataset annoté du prof, la vérité est connue, donc la correction doit être **oui ou non**, sans nuance.
- La disposition de D18 donnait une impression d'éparpillement : les scores globaux étaient mélangés aux résultats de l'image, la colonne de gauche empilait six éléments, et la légende et les tableaux étaient loin de l'image.

**Choix** (`src/avions/app.py`, `src/avions/visualization.py`)
- **Plus de niveau « à vérifier » dans l'interface.** Le modèle répond seulement là où sa confiance finale atteint 0,5. C'est exactement la règle de nos scores officiels : additionnés sur les 131 images, les bilans par image redonnent 729 justes, 9 mauvais types, 3 « pas un avion » et 50 oubliés (vérifié image par image).
- **Correction binaire** : chaque réponse est **juste** (vert) ou **fausse** (rouge, avec la raison : « A19, vrai : A1 » ou « pas un avion ») ; chaque vrai avion sans réponse est **oublié** (rouge pointillé).
- **Trois colonnes, lues de gauche à droite** :
  1. choisir (exemples, hasard, galerie) ;
  2. l'image en grand, sur fond sombre ;
  3. « le modèle répond » puis « on corrige », chaque bouton au-dessus de son résultat.

  Le bilan de l'image utilise les mêmes pastilles que les boîtes : il sert de légende.
- **Les scores globaux ont leur propre onglet**, « Scores sur tout le test » : justes, fausses, oubliés, précision, rappel, F1, mAP50, et la comparaison avec le témoin. L'onglet de démonstration ne montre que l'image en cours.
- **Cinq exemples conseillés**, un par situation :
  - 883 : sans faute ;
  - 3803 : A1 ou A19 ? ;
  - 3256 : 5 A13 pris pour des A15 ;
  - 3734 : 2 « pas un avion » ;
  - 3607 : 50 avions serrés.
- **Dessin lisible** : étiquettes en pastille (Pillow), qui gèrent les accents, ✓ et ✗. Au-delà de 30 boîtes, les réponses justes n'ont pas d'étiquette.
- **Le F1 en direct**, en haut à droite.
  - En grand, le **F1 cumulé**. On additionne les avions de toutes les images corrigées depuis l'ouverture de la page, au seuil officiel, et chaque image ne compte qu'une fois. On calcule ensuite F1 = 2 × justes / (réponses + vrais avions). Après les 131 images, on retombe exactement sur 0,954.
  - En petit, le F1 de l'image en cours. Il n'est qu'indicatif : avec peu d'avions, une seule erreur le fait beaucoup varier.
  - Les couleurs : vert à partir de 0,9, orange à partir de 0,6, rouge en dessous.
  - Sur le test, 106 images sur 131 ont un F1 de 1.
  - La **moyenne** des F1 par image (0,937) n'est pas le F1 officiel (0,954). Le F1 officiel additionne d'abord les avions de toutes les images : une image de 50 avions y pèse plus qu'une image de 2.
- Testé à 1536 × 864 et 1366 × 768 : tout tient sur un écran, sans faire défiler.

**Méthode :** trois propositions de disposition indépendantes, chacune avec son angle (l'oral, un utilisateur débutant, le rendu visuel), puis un jury qui les a notées et fusionnées.

Le niveau « à vérifier » reste dans le pipeline et dans le corpus annoté (D14, D15).

## D20 : on garde YOLO11n, car les avions oubliés viennent du classifieur (27/10/2026)

**Question :** un détecteur plus gros (YOLO11s ou YOLO11m) améliorerait-il le F1 sur les images difficiles ?

**Analyse** des 50 avions oubliés du test, au seuil officiel. On a regardé, pour chacun, si le détecteur avait une boîte dessus (avec une IoU d'au moins 0,5), en descendant sa confiance jusqu'à 0,05.

| Ce qui s'est passé | Avions |
|---|---|
| Le détecteur l'a trouvé (confiance d'au moins 0,5), mais le classifieur hésitait sur le type | 47 |
| Le détecteur l'a trouvé, avec une confiance inférieure à 0,5 | 2 |
| Le détecteur ne l'a pas vu du tout | 1 |

- Sur ces avions, la confiance du détecteur est de 0,83 en médiane, celle du classifieur de 0,46. Le produit (0,83 × 0,46 ≈ 0,38) passe sous 0,5.
- Pour 29 des 47 avions, le classifieur avait pourtant deviné le bon type, mais sans être assez sûr.
- Les types les plus touchés sont A16 (19 oubliés), A1 (13) et A5 (7).

**Décision :** on garde YOLO11n. Un détecteur plus gros ne récupérerait que 1 à 3 avions sur 788, demanderait un nouvel entraînement et rendrait l'application 2 à 3 fois plus lente.

**Pistes d'amélioration, toutes côté classifieur** (non appliquées) :
- un seuil de 0,4, qui était meilleur sur la validation ;
- la moyenne des réponses du classifieur sur plusieurs rotations de la vignette ;
- un classifieur plus gros.

Ces réglages devraient être choisis sur la validation, jamais sur le test : aucun seuil n'a été recalculé sur le test pour cette analyse.
