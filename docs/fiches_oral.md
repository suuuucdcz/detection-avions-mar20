# Fiches de révision pour l'oral

À l'oral, chacun tire au sort des notions de cette liste et doit savoir **les expliquer, les définir et dire à quoi elles servent**.

Pour chaque notion :
- **Où dans le projet** te dit où on la rencontre. Avoir un exemple tiré de notre projet impressionne beaucoup plus qu'une définition apprise par cœur.
- **Ma définition** est à écrire **avec tes propres mots**, au fur et à mesure qu'on avance. Je relirai.

---

## 1. IA / Machine Learning / Deep Learning

- **Où dans le projet :** la forêt aléatoire sur des caractéristiques HOG calculées « à la main », c'est du ML classique. Le CNN qui apprend lui-même ses caractéristiques, c'est du DL.
- **Ma définition :**

## 2. Apprentissage supervisé vs non supervisé

- **Où dans le projet :** nos classifieurs et détecteurs sont supervisés, car on leur donne les annotations. Regrouper les tailles d'avions sans regarder les classes pendant l'EDA (k-means) serait du non supervisé.
- **Ma définition :**

## 3. Régression vs classification

- **Où dans le projet :** dire « A1 ou A2 ? », c'est une classification. Le détecteur prédit aussi 4 nombres réels pour chaque boîte (position, taille), c'est une régression.
- **Ma définition :**

## 4. Jeux d'entraînement, de validation et de test

- **Où dans le projet :** notre découpage 80 / 10 / 10 **par image**, fait en séance 1 (1 070 / 130 / 131 images). Pourquoi par image et pas par avion ? (fuite de données)
  - Exemple frappant : la mAP50 de notre système vaut 0,990 sur la validation, qui a servi à choisir les epochs et le seuil, mais **0,957 sur le test**, jamais vu. Le test donne le vrai score.
- **Ma définition :**

## 5. Sur-apprentissage (overfitting) et sous-apprentissage (underfitting)

- **Où dans le projet :**
  - **overfitting** : la forêt aléatoire, avec 100 % sur l'entraînement mais 69 % en validation ;
  - **underfitting** : le petit CNN, dont la loss reste élevée en entraînement (1,50) comme en validation (1,51) ;
  - **le bon équilibre** : le ResNet-18 pré-entraîné, à 0,73 des deux côtés ;
  - l'**early stopping** arrête l'entraînement quand la validation ne progresse plus.
- **Ma définition :**

## 6. La descente de gradient

- **Où dans le projet :** la boucle d'entraînement du ResNet-18 : prédiction → calcul de l'erreur (loss) → gradient → mise à jour des poids.
- **Ma définition :**

## 7. Arbres de décision et forêts aléatoires

- **Où dans le projet :** la forêt aléatoire, notre baseline de classification en séance 2.
- **Ma définition :**

## 8. Matrice de confusion, précision, rappel, F1-score

- **Où dans le projet :**
  - la matrice de confusion 20 × 20 du classifieur montre quelles classes sont confondues : A1 avec A19, A15 avec A13 ;
  - le **compromis précision / rappel** du seuil de l'application, sur la validation : au seuil 0,25, il y a 16 fausses alarmes (faux positifs) et 0 avion manqué ; au seuil 0,5, 4 fausses alarmes et 19 avions manqués (faux négatifs) ;
  - au seuil choisi de 0,5, la précision vaut 0,994, le rappel 0,975 et la F1 0,984 en validation ; sur le test, 0,984, 0,925 et 0,954.
- **Ma définition :**

## 9. Le perceptron et les réseaux de neurones (MLP)

- **Où dans le projet :** le MLP de scikit-learn, en comparaison de la forêt aléatoire. Le neurone formel a été vu en cours.
- **Ma définition :**

## 10. Les réseaux de neurones convolutifs (CNN)

- **Où dans le projet :** notre petit CNN maison, ResNet-18 et YOLO sont tous des CNN.
- **Ma définition :**

## 11. Le transfer learning

- **Où dans le projet :** ResNet-18 pré-entraîné sur ImageNet et YOLO11n pré-entraîné sur COCO. La preuve chiffrée, en validation :
  - avec pré-entraînement : macro-F1 de 0,996 ;
  - sans pré-entraînement : 0,973.

  Sur le **test** (le vrai score), on compte 30 erreurs contre 44 sur 788 vignettes (macro-F1 de 0,958 contre 0,949), soit environ 30 % d'erreurs en moins. Le meilleur score est aussi atteint 3 fois plus vite.
  - À dire à l'oral : la validation (0,996 contre 0,973) exagérait l'écart. C'est pour ça qu'on garde un jeu de test.
- **Ma définition :**

## 12. La data augmentation

- **Où dans le projet :** les rotations et retournements des vignettes, adaptés à la vue du ciel. Pourquoi pas de déformation de perspective ?
- **Ma définition :**
