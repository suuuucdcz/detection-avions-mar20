# Corpus annoté par le modèle

131 images satellites annotées automatiquement par notre système en deux étages :
un détecteur YOLO11n (« où sont les avions ? »), puis un classifieur ResNet-18
(« quel type, de A1 à A20 ? »).

## Réglages

- **Avion sûr** : confiance finale (détecteur × classifieur) ≥ 0.5.
- **Avion à vérifier** : le détecteur est sûr de voir un avion (≥ 0.6),
  mais la confiance finale est plus basse, le plus souvent parce que le type est incertain.
  Il revient alors à un humain de trancher.

## Contenu

| Dossier ou fichier | Contenu |
|---|---|
| `annotations/` | un fichier XML par image, au format Pascal VOC (même format que le dataset) |
| `images/` | les images d'origine |
| `apercus/` | les images avec les boîtes : épaisses pour les avions sûrs, fines avec « ? » pour les avions à vérifier |
| `detections.csv` | toutes les détections : image, type, confiances, boîte, statut |

Dans les fichiers XML, chaque `<object>` contient :
- `<name>` : le type d'avion ;
- `<bndbox>` : la boîte, en pixels ;
- `<score>` : la confiance finale ;
- `<a_verifier>` : 1 si l'avion est à vérifier, 0 sinon.

## Ce que le modèle a trouvé

- **741 avions sûrs** et **61 avions à vérifier**.
- Types les plus fréquents parmi les avions sûrs : A2 (73), A14 (66), A16 (58), A19 (52), A9 (49).

## Scores sur ce corpus (comparaison avec la vérité terrain)

Ces images forment le **jeu de test** : elles n'ont jamais servi ni à entraîner, ni à
régler le modèle. Elles ont été évaluées une seule fois, à la fin du projet.

| Mesure | Valeur |
|---|---|
| Avions réels | 788 |
| Avions sûrs du bon type | 729 |
| Avions sûrs du mauvais type | 9 |
| Fausses alarmes (faux positifs) | 3 |
| Avions non détectés comme sûrs | 50 |
| Précision / rappel / F1 au seuil 0.5 | 0.984 / 0.925 / 0.954 |
| mAP50 (tous les seuils) | 0.957 |
