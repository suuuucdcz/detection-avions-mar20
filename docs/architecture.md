# Architecture du système

Ce document décrit le fonctionnement de notre application de détection et de classification d'avions sur images satellites. Il sert de base à la partie « architecture » du rapport et de l'oral. Les justifications détaillées de chaque choix se trouvent dans [journal_decisions.md](journal_decisions.md).

## 1. Vue d'ensemble : deux étages spécialisés

```
                      image satellite (environ 800 × 800 pixels)
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │  ÉTAGE 1 : DÉTECTEUR, YOLO11n à 1 classe « avion »    │
        │  « Y a-t-il des avions ? Où ? »                       │
        │  → une boîte par avion + une confiance                │
        └───────────────────────────────────────────────────────┘
                                    │
                                    ▼
        découpe d'une VIGNETTE autour de chaque boîte
        (un carré de 2 × la taille de l'avion, en 256 × 256 pixels)
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │  ÉTAGE 2 : CLASSIFIEUR, ResNet-18                     │
        │  « Quel type d'avion ? » (A1 … A20)                   │
        │  → un type + une confiance                            │
        └───────────────────────────────────────────────────────┘
                                    │
                                    ▼
        CONFIANCE FINALE = confiance du détecteur × confiance du classifieur
                                    │
              ┌─────────────────────┼──────────────────────────┐
              ▼                     ▼                          ▼
       ≥ 0,5 : avion SÛR    détecteur ≥ 0,6, mais      sinon : écarté
       (résultat            confiance finale < 0,5 :
       automatique)         avion À VÉRIFIER (un
                            humain tranche)
```

**Pourquoi deux étages plutôt qu'un seul réseau ?**
- **Le détecteur** n'a qu'une question à résoudre, « avion ou pas », et il dispose des 7 870 exemples pour l'apprendre. Le déséquilibre entre les 20 types ne le gêne pas.
- **Le classifieur** voit chaque avion **en gros plan**, et peut apprendre avec des rotations dans tous les sens.
- **On peut évaluer chaque étage séparément**, et savoir si une erreur vient d'un avion manqué ou d'un avion mal reconnu.
- **C'est la démarche conseillée par le prof** : vignettes, classification, détection, intégration.
- **Le classifieur sert aussi de second filtre** contre les fausses alarmes : une boîte sur un toit ne ressemble à aucun type d'avion, donc sa confiance finale reste basse.

Pour le vérifier, on a entraîné un **témoin** : un YOLO11n à 20 classes qui fait tout d'un coup. Notre système fait mieux (voir la partie 4).

## 2. Les données et leur préparation

| Étape | Ce qu'on fait | Pourquoi |
|---|---|---|
| Analyse (EDA) | 1 331 images, 7 870 avions, 20 types très déséquilibrés (de 107 à 785 avions) | connaître ses données avant d'entraîner (« Foncer, c'est échouer ») |
| Découpage | train / validation / test = 1 070 / 130 / 131 images, **par image**, en gardant les proportions de classes et les quasi-doublons ensemble | éviter toute fuite de données entre les jeux |
| Format YOLO | annotations converties en fichiers texte (classe, centre, largeur, hauteur, entre 0 et 1) | format attendu par le détecteur |
| Vignettes | un carré autour de chaque avion : 6 295 en train, 787 en validation, 788 en test | entrée du classifieur ; **la même fonction** sert à l'entraînement et dans l'application |

## 3. Les modèles et leur entraînement

| Modèle | Rôle | Réglages clés |
|---|---|---|
| **YOLO11n** (pré-entraîné sur COCO) | étage 1 : trouver les avions | 1 classe, images en 800 px, retournements, early stopping (arrêt à l'epoch 67) |
| **ResNet-18** (pré-entraîné sur ImageNet) | étage 2 : reconnaître le type | vignettes en 224 px, rotations, luminosité, pondération des classes rares, 3 epochs avec le backbone gelé |
| Forêt aléatoire (HOG + taille) | témoin de ML classique | 300 arbres |
| ResNet-18 sans pré-entraînement, petit CNN | comparaisons | mesurent l'apport du transfer learning, et montrent l'underfitting |
| YOLO11n à 20 classes | témoin « tout en un » | mêmes réglages que le détecteur |

- **Entraînement** : sur Google Colab (GPU T4), car la carte graphique du PC était instable.
- **Utilisation** (l'application) : sur le processeur, en environ 0,2 seconde par image.

## 4. Les résultats (jeu de test : 131 images, 788 avions, jamais vus)

| Système (seuil de 0,5) | F1 | Précision | Rappel | Fausses alarmes | Mauvais types |
|---|---|---|---|---|---|
| **Notre système en deux étages** | **0,954** | **0,984** | 0,925 | **3** | **9** |
| Témoin YOLO à 20 classes | 0,912 | 0,913 | 0,910 | 19 | 49 |

Avec les avions « à vérifier », **785 avions sur 788** sont présentés à l'utilisateur.

**Classifieurs seuls**, macro-F1 sur les vignettes de test :

| Modèle | Macro-F1 |
|---|---|
| Forêt aléatoire | 0,625 |
| Petit CNN | 0,567 |
| ResNet-18 sans pré-entraînement | 0,949 |
| **ResNet-18 pré-entraîné** | **0,958** |

**Limites :**
- les images de test viennent des mêmes bases aériennes que l'entraînement : les scores sont sans doute un peu optimistes pour des bases jamais vues ;
- les confusions restantes concernent surtout **A1 et A19**, deux chasseurs très ressemblants.

## 5. L'application

L'interface web (Gradio, `uv run python scripts/app.py`) propose les **131 images de test**. L'écran se lit de gauche à droite :
1. on **choisit** une image : un exemple conseillé (★), une vignette, ou au hasard ;
2. **« Détecter les avions »** : le modèle **répond** (boîtes bleues), là où sa confiance finale atteint 0,5 ;
3. **« Corriger avec la vérité »** : c'est **oui ou non**. Chaque réponse est **juste** (vert) ou **fausse** (rouge : mauvais type ou « pas un avion »), et chaque vrai avion sans réponse est **oublié** (rouge pointillé).

C'est exactement la règle de nos scores officiels : additionnés sur les 131 images, les bilans redonnent 729 justes, 12 fausses et 50 oubliés. Le second onglet, « Scores sur tout le test », les présente. Le niveau « à vérifier » reste dans le corpus annoté, mais pas dans l'interface (décision D19).

Le **corpus de test annoté** (fichiers XML au format Pascal VOC, plus des aperçus) est produit par `scripts/annotate_corpus.py`.

## 6. Organisation du code

```
src/avions/
├── data/          lecture des annotations, découpage, vignettes, format YOLO
├── classif/       modèles, augmentations, jeu de vignettes, métriques
├── pipeline.py    le système complet en deux étages
├── evaluation.py  mAP, précision, rappel, F1, verdict boîte par boîte
├── export.py      export au format Pascal VOC
└── app.py         l'interface web
scripts/           les commandes (préparer, entraîner, évaluer, annoter, lancer l'interface)
configs/           les réglages de chaque entraînement
tests/             30 tests automatiques
```
