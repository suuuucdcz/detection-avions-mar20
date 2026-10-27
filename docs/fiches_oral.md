# Fiches de révision pour l'oral

À l'oral, chacun tire au sort des notions de cette liste. Il faut savoir **les définir, les expliquer et dire à quoi elles servent**.

Le meilleur réflexe est de répondre en trois temps :
1. **la définition**, en une phrase ;
2. **l'exemple tiré de notre projet**, avec nos chiffres ;
3. **à quoi ça sert**.

Un exemple concret impressionne beaucoup plus qu'une définition récitée.

Tous les chiffres viennent de notre projet. Le détail se trouve dans [journal_decisions.md](journal_decisions.md).

---

## 1. IA, Machine Learning et Deep Learning

**En une phrase :** l'IA est le domaine général. Le Machine Learning (ML) en est une partie, où la machine **apprend à partir d'exemples**. Le Deep Learning (DL) est une partie du ML qui utilise des **réseaux de neurones profonds** (beaucoup de couches).

```
IA  ⊃  Machine Learning  ⊃  Deep Learning
```

**Explication :**
- **IA** : tout programme qui imite un comportement « intelligent ». Il peut même suivre des règles écrites à la main. Le cours la définit comme la science des agents rationnels, qui perçoivent leur environnement et agissent pour atteindre un objectif.
- **ML** : au lieu d'écrire les règles, on montre des exemples, et l'algorithme trouve lui-même la règle.
- **DL** : un ML avec des réseaux de neurones à nombreuses couches, qui **trouvent eux-mêmes quoi regarder** dans les données brutes (les pixels).

**Dans notre projet :**
- **ML classique** : la forêt aléatoire. Elle ne regarde pas les pixels : on lui donne des caractéristiques **calculées à la main**, les HOG (la forme des contours) et la taille de l'avion. Elle obtient une macro-F1 de **0,625** sur le test.
- **Deep Learning** : le ResNet-18 apprend lui-même ce qui distingue les avions, directement à partir des pixels. Il obtient **0,958**.

**À retenir :** le DL est plus puissant sur les images, mais il demande plus de données et de calcul (d'où le GPU de Colab).

**Question possible :** « Pourquoi le Deep Learning marche mieux ici ? »
→ Parce que distinguer 20 avions demande des caractéristiques très fines. Un CNN les apprend tout seul, alors que des caractéristiques choisies à la main (HOG) en ratent beaucoup.

---

## 2. Apprentissage supervisé et non supervisé

**En une phrase :** en **supervisé**, les exemples sont fournis **avec la bonne réponse** (l'étiquette). En **non supervisé**, il n'y a pas de réponse, et l'algorithme cherche seul des structures dans les données (des groupes, par exemple).

**Dans notre projet :** **tout est supervisé**. Chaque image vient avec ses annotations : la boîte et le type de chaque avion. Le modèle apprend en comparant ses prédictions à ces bonnes réponses.

**Exemple de non supervisé :** regrouper les avions par taille **sans regarder leur type**, par exemple avec l'algorithme k-means. Ce serait du non supervisé. On ne l'a pas utilisé pour entraîner nos modèles.

**À quoi ça sert :**
- le supervisé sert à **prédire** une réponse connue à l'avance : le type d'avion ;
- le non supervisé sert à **explorer** : trouver des groupes, des anomalies.

**Question possible :** « Votre projet est supervisé ou non ? Pourquoi ? »
→ Supervisé, car on dispose des annotations, c'est-à-dire les boîtes et les types, faites par des humains au format Pascal VOC.

---

## 3. Régression et classification

**En une phrase :** la **classification** prédit une **catégorie** (« A1 ou A19 ? »). La **régression** prédit un **nombre** (une position, une taille, un prix).

**Dans notre projet, on fait les deux :**
- **classification** : le ResNet-18 choisit le type parmi 20 classes ;
- **régression** : le détecteur YOLO prédit **4 nombres** pour chaque avion, c'est-à-dire la position et la taille de la boîte.

  YOLO fait aussi une petite classification : « avion ou pas avion ».

**Question possible :** « Donnez un exemple de régression dans votre projet. »
→ Le détecteur prédit les coordonnées de chaque boîte (x, y, largeur, hauteur) : ce sont des nombres réels.

---

## 4. Jeux d'entraînement, de validation et de test

**En une phrase :**
- on **apprend** sur l'entraînement ;
- on **règle et compare** sur la validation ;
- on mesure le **vrai score une seule fois** sur le test.

**Dans notre projet :** on a découpé les 1 331 images en 80 / 10 / 10, soit **1 070 / 130 / 131 images**.
- **Par image**, et pas par avion. Sinon, deux avions d'une même image (même sol, même lumière) se retrouveraient dans deux jeux différents : c'est une **fuite de données**.
- Les **quasi-doublons** (deux photos de la même scène) restent dans le même jeu.
- La **validation** a servi à choisir la meilleure epoch, à comparer les modèles et à choisir le seuil de 0,5.
- Le **test** n'a servi **qu'une fois**, à la fin.

**L'exemple qui marque :** notre système obtient une F1 de **0,984 sur la validation**, mais de **0,954 sur le test**. La validation est un peu optimiste, parce qu'on a fait nos choix dessus. **Le vrai score, c'est celui du test.**

**Analogie :** l'entraînement, ce sont les exercices corrigés ; la validation, les examens blancs ; le test, l'examen final.

**Question possible :** « Pourquoi ne pas évaluer sur les données d'entraînement ? »
→ Le modèle les a déjà vues. Il pourrait les avoir apprises par cœur (overfitting) et paraître excellent sans savoir généraliser. Exemple : notre forêt aléatoire obtient 100 % sur l'entraînement, mais 69 % en validation.

---

## 5. Sur-apprentissage (overfitting) et sous-apprentissage (underfitting)

**En une phrase :**
- **overfitting** : le modèle **apprend par cœur** les exemples d'entraînement, et il est mauvais sur de nouvelles données ;
- **underfitting** : le modèle est **trop simple** pour apprendre, et il est mauvais partout.

**Dans notre projet, on a les deux cas, plus le bon :**

| Modèle | Entraînement | Validation | Diagnostic |
|---|---|---|---|
| Forêt aléatoire | accuracy de 100 % | accuracy de 69 % | **overfitting** : énorme écart, elle a appris par cœur |
| Petit CNN maison | loss élevée (1,50) | loss élevée (1,51) | **underfitting** : trop simple, mauvais partout |
| ResNet-18 pré-entraîné | loss de 0,73 | loss de 0,73 | **bon équilibre**, macro-F1 de 0,958 sur le test |

La loss n'est pas un pourcentage d'erreur : avec le label smoothing utilisé (0,1), elle ne descend pas sous 0,67 environ. En validation, l'accuracy est de 99,6 % pour le ResNet-18 et de 75,1 % pour le petit CNN.

**Comment on le combat :**
- l'**early stopping** : on arrête l'entraînement quand la validation ne progresse plus. Notre détecteur s'est arrêté seul à l'epoch 67 sur 80 ;
- la data augmentation ;
- le transfer learning ;
- la pénalisation des poids trop grands (*weight decay*).

**Question possible :** « Comment détecter l'overfitting ? »
→ En comparant les scores sur l'entraînement et sur la validation. Un grand écart signale de l'overfitting. Les courbes d'apprentissage le montrent : la loss d'entraînement continue de baisser alors que celle de validation remonte.

---

## 6. La descente de gradient

**En une phrase :** c'est la méthode qui **ajuste les poids** du réseau petit à petit, dans la direction qui **diminue l'erreur**.

**Comment ça marche, à chaque étape :**
1. le réseau fait une prédiction sur un paquet d'images (un **batch**) ;
2. on mesure son **erreur**, la *loss* ;
3. on calcule le **gradient** de la loss par rapport à chaque poids, par **rétropropagation** : il indique dans quel sens l'erreur augmente le plus vite ;
4. on déplace chaque poids dans le sens **opposé** : w ← w − η × gradient. Le coefficient η s'appelle le **learning rate**. Comme on le fait batch par batch, on parle de descente de gradient stochastique.

On recommence des milliers de fois. Un passage complet sur toutes les images s'appelle une **epoch**.

**Analogie :** descendre une montagne dans le brouillard. On ne voit pas le bas, mais on sent la pente sous ses pieds, et on fait un pas dans la direction qui descend.
- Un pas trop grand, et on saute par-dessus la vallée.
- Un pas trop petit, et on met des heures.

**Dans notre projet :**
- le ResNet-18 apprend par paquets de **64 vignettes**, avec un learning rate de **0,0003** (optimiseur AdamW, une version améliorée de la descente de gradient) ;
- sa loss baisse epoch après epoch, comme on le voit dans `runs/classifieur_resnet18/results.png`.

**Question possible :** « Que se passe-t-il si le learning rate est trop grand ? »
→ Les pas sont trop grands : l'erreur oscille ou explose au lieu de diminuer.

---

## 7. Arbres de décision et forêts aléatoires

**En une phrase :**
- un **arbre de décision** classe un exemple en posant une suite de **questions simples**, du genre « la largeur est-elle > 120 pixels ? » ;
- une **forêt aléatoire** fait **voter** des centaines d'arbres. Chacun est entraîné sur un tirage au hasard, avec remise, des exemples (bootstrap), et à chaque question il ne choisit que parmi une partie des caractéristiques tirée au hasard.

**Pourquoi une forêt plutôt qu'un seul arbre ?** Un arbre seul colle trop à ses données d'entraînement (overfitting). En faisant voter beaucoup d'arbres un peu différents, les erreurs se compensent.

**Dans notre projet :**
- c'est notre **modèle de référence** (baseline) : **300 arbres** ;
- ils travaillent sur des caractéristiques calculées à la main, les HOG et la taille de la boîte ;
- l'option `class_weight` compense les classes rares ;
- résultat sur le test : macro-F1 de **0,625**, contre **0,958** pour le ResNet-18 ;
- c'est aussi notre exemple d'overfitting : 100 % en entraînement.

**Question possible :** « Pourquoi garder une forêt aléatoire si elle est moins bonne ? »
→ Elle sert de **point de comparaison**. Elle montre ce qu'apporte le Deep Learning : 0,625 → 0,958.

---

## 8. Matrice de confusion, précision, rappel et F1-score

**La matrice de confusion** est un tableau : en lignes, la **vraie classe** ; en colonnes, la **classe prédite**. La diagonale correspond aux bonnes réponses. Le reste montre **quelles classes sont confondues entre elles**. Notre matrice 20 × 20 montre que **A1 est confondu avec A19**.

**Les mesures**, avec VP = vrais positifs, FP = faux positifs, FN = faux négatifs :
- **précision** = VP / (VP + FP) : « quand le modèle dit *avion*, a-t-il raison ? » ;
- **rappel** = VP / (VP + FN) : « parmi les vrais avions, combien en trouve-t-il ? » ;
- **F1** = 2 × précision × rappel / (précision + rappel) : un **compromis** entre les deux, élevé seulement si les deux le sont.

**Faux positif et faux négatif :**
- un **faux positif** est une réponse fausse : une boîte sans avion, ou le mauvais type ;
- un **faux négatif** est un vrai avion sans bonne réponse : oublié, ou mal classé.

Un mauvais type compte donc dans les deux. Au test : 12 faux positifs (3 fausses alarmes + 9 mauvais types) et 59 faux négatifs (50 oubliés + 9 mauvais types).

**Dans notre projet** (test, seuil de 0,5) :
- **précision de 0,984**, **rappel de 0,925**, **F1 de 0,954** ;
- **3 fausses alarmes** pour 788 avions, contre 19 pour le témoin.

**Le compromis précision / rappel**, mesuré sur la validation :
- au seuil 0,25 : 16 fausses alarmes, 0 avion manqué ;
- au seuil 0,5 : 4 fausses alarmes, 19 avions manqués.

Monter le seuil donne moins de fausses alarmes, mais plus d'avions manqués.

**Macro-F1 :** on calcule la F1 **de chaque classe**, puis on fait la moyenne. Chaque classe compte autant. C'est indispensable quand les classes sont déséquilibrées : A6 n'a que 107 avions, contre 785 pour A16. L'accuracy seule cacherait les erreurs sur les classes rares.

**Question possible :** « Pourquoi la F1 plutôt que l'accuracy ? »
→ Avec des classes déséquilibrées, un modèle peut avoir une bonne accuracy en ratant toutes les classes rares. La F1, et surtout la macro-F1, tient compte à la fois des fausses alarmes et des oublis, classe par classe.

---

## 9. Le perceptron et les réseaux de neurones (MLP)

**Le perceptron**, ou neurone formel (vu en cours), calcule une **somme pondérée** de ses entrées, ajoute un biais, puis applique une **fonction seuil** : il répond 0 ou 1. C'est un classifieur **linéaire** : il ne sait séparer que des classes séparables par une droite (un hyperplan), et il échoue sur le XOR. Le MLP lève cette limite.

```
entrées x1, x2, … → (w1·x1 + w2·x2 + … + b) → activation → sortie
```

Les **poids** (w) sont ce que l'apprentissage ajuste, par descente de gradient.

**Le MLP** (*multi-layer perceptron*) empile **plusieurs couches** de ces neurones, chaque neurone étant relié à tous ceux de la couche suivante. Grâce aux fonctions d'activation (comme ReLU), il peut apprendre des relations **non linéaires**.

**Dans notre projet :**
- la **dernière couche** de notre ResNet-18 est une couche de ce type. Elle prend les **512 nombres** qui résument la vignette, et produit **20 scores**, un par type d'avion. Une fonction *softmax* les transforme en probabilités ;
- le petit CNN maison se termine aussi par une couche de ce type (256 → 20).

Nous n'avons pas entraîné de MLP seul : sur des images, un CNN est bien plus adapté (voir la notion suivante).

**Question possible :** « Pourquoi ne pas utiliser un MLP directement sur les pixels ? »
→ Une image de 224 × 224 × 3 contient 150 528 pixels. Un MLP aurait un nombre énorme de poids et ignorerait la position des pixels les uns par rapport aux autres. Un CNN partage ses filtres sur toute l'image, et il est donc bien plus efficace.

---

## 10. Les réseaux de neurones convolutifs (CNN)

**En une phrase :** un CNN est un réseau spécialisé pour les images. Il fait glisser de **petits filtres** (3 × 3 pixels, par exemple) sur l'image pour détecter des **motifs locaux** : bords, coins, formes.

**Comment ça marche :**
1. **les premières couches** détectent des motifs simples : des bords, des lignes ;
2. **les couches suivantes** combinent ces motifs en formes plus complexes : une aile, un fuselage, des moteurs ;
3. le **pooling** (ou des convolutions avec un pas de 2) réduit la taille des cartes de caractéristiques au fil des couches, et le réseau voit de plus en plus « large » ;
4. **à la fin**, une couche de type perceptron décide de la classe.

Avantage : le même filtre sert partout dans l'image. Il y a beaucoup moins de poids qu'un MLP, et un avion est reconnu quelle que soit sa position.

**Dans notre projet, nos trois réseaux sont des CNN :**
- **YOLO11n**, le détecteur ;
- **ResNet-18**, le classifieur, avec 18 couches. Ses « connexions résiduelles » (des raccourcis entre couches) permettent d'entraîner des réseaux profonds ;
- **le petit CNN maison**, avec 4 blocs de convolution.

**Question possible :** « Qu'est-ce qu'un backbone ? »
→ C'est la partie du réseau qui **extrait les caractéristiques** de l'image (les couches de convolution). La « tête » transforme ensuite ces caractéristiques en réponse.

---

## 11. Le transfer learning

**En une phrase :** au lieu de partir de zéro, on **réutilise un réseau déjà entraîné** sur une autre grande tâche, et on l'adapte à la nôtre.

**Pourquoi ça marche :** un réseau entraîné sur des millions d'images sait déjà « voir » : bords, textures, formes. On garde ce savoir (le **backbone**) et on ne réapprend que la fin.

**Dans notre projet :**
- le **ResNet-18** a été pré-entraîné sur **ImageNet**, soit 1,2 million de photos de 1 000 catégories. On a remplacé sa dernière couche (1 000 sorties) par une couche à **20 sorties** ;
- pendant les **3 premières epochs**, le backbone est **gelé** et seule la nouvelle couche apprend. Ensuite, tout le réseau s'ajuste ;
- le **YOLO11n** a été pré-entraîné sur **COCO** ;
- dès la **1re epoch**, notre détecteur trouvait déjà 88 % des avions.

**La preuve chiffrée**, avec le même ResNet-18 sur le test :

| | Macro-F1 | Erreurs sur 788 |
|---|---|---|
| Avec pré-entraînement | **0,958** | **30** |
| Sans pré-entraînement | 0,949 | 44 |

Environ 30 % d'erreurs en moins, et le meilleur score est atteint 3 fois plus vite (epoch 9 contre 29).

**Règle du prof :** ajouter des **images** extérieures est interdit, mais le transfer learning est autorisé. On réutilise des **poids**, pas des images.

**Question possible :** « Pourquoi le transfer learning est-il utile avec peu de données ? »
→ Avec seulement 6 295 vignettes d'entraînement, un réseau partant de zéro apprend moins bien. Partir d'un réseau qui sait déjà voir donne de meilleurs résultats, plus vite, et avec moins de calcul : c'est aussi plus **frugal**, comme le recommande le cours.

---

## 12. La data augmentation

**En une phrase :** on crée des **variantes artificielles** des images d'entraînement (tournées, retournées, plus claires…). Le modèle voit plus de cas différents et généralise mieux.

**Dans notre projet** (choix justifiés par l'EDA) :
- **rotations de n'importe quel angle** et **retournements** : vu du ciel, un avion peut pointer dans toutes les directions ;
- **luminosité et contraste** ±20 % : les images vont de très sombres à très claires ;
- **cadrage légèrement décalé** : il imite les boîtes imparfaites du détecteur ;
- pour YOLO : retournements et « mosaïque », qui colle 4 images ensemble.

**Ce qu'on a évité :** les déformations de **perspective**. Une image satellite est prise du dessus, sans perspective : une telle déformation montrerait au modèle des images qui n'existent pas en vrai.

**Question possible :** « La data augmentation ajoute-t-elle des données ? Est-ce autorisé ? »
→ Elle ne crée pas de nouvelles informations extérieures : ce sont des **variantes de nos propres images**. C'est donc compatible avec la règle « pas de données externes ». Elle limite l'overfitting.

---

## Glossaire des mots du projet

| Mot | Définition simple |
|---|---|
| **Epoch** | un passage complet sur toutes les images d'entraînement |
| **Batch** | un paquet d'images traitées ensemble avant de corriger les poids (64 vignettes chez nous) |
| **Loss** | l'erreur que l'on cherche à diminuer pendant l'entraînement |
| **Early stopping** | arrêt automatique quand la validation ne progresse plus |
| **Boîte englobante** | le rectangle autour d'un avion (xmin, ymin, xmax, ymax) |
| **IoU** | à quel point deux boîtes se recouvrent, de 0 à 1. Une détection compte comme juste si son IoU avec le vrai avion est ≥ 0,5 |
| **mAP50** | une note globale de détection (de 0 à 1) qui combine précision et rappel sur tous les seuils |
| **NMS** | supprime les boîtes en double sur un même avion, en ne gardant que la plus sûre |
| **Seuil de confiance** | la confiance minimale pour garder une détection (0,5 chez nous) |
| **Vignette** | le carré découpé autour d'un avion, donné au classifieur |
| **Backbone** | la partie d'un réseau qui extrait les caractéristiques de l'image |
| **Fuite de données** | quand des informations du test se retrouvent dans l'entraînement, ce qui fausse le score |
| **Témoin** | le modèle de comparaison (ici, YOLO à 20 classes) |
| **Avion « à vérifier »** | une détection incertaine, montrée à part pour qu'un humain tranche (l'« humain dans la boucle » du cours). On le garde dans le corpus annoté, mais pas dans l'interface, où la correction est en oui / non |
