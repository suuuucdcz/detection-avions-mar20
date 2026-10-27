# Le dossier de l'oral

Tout ce qu'il faut pour préparer et passer l'oral, au même endroit.

## Ce qu'il y a dans ce dossier

| Fichier | À quoi il sert |
|---|---|
| `guide_projet.html` | **Le document pour tout apprendre.** Double-clic : il s'ouvre dans le navigateur. Il contient six parties : A, le projet en bref ; B, ce qu'on a fait étape par étape, avec les figures ; C, les 12 notions théoriques ; D, la préparation de l'oral ; E, le code fichier par fichier ; F, le glossaire. |
| `fiches_oral.md` | Les 12 notions en version courte, pour réviser vite. |
| `lancer_demo.bat` | Double-clic : lance l'interface de démonstration. Elle s'ouvre dans le navigateur. |
| `ouvrir_le_code.bat` | Double-clic : ouvre le projet dans VS Code, avec les fichiers des 10 passages à savoir montrer (partie E du guide). |
| `guide_images/` | Les figures du guide. Laisser ce dossier à côté du guide. |

**Ne sors pas ce dossier du projet** : les deux `.bat` lancent les commandes depuis `projet_ia`, le dossier parent. Pour emporter le projet, copie tout `projet_ia`, avec `data/raw/`.

## Ailleurs dans le projet

- `docs/rapport.pdf` : **le rapport à envoyer au prof**, au moins 24 h avant l'oral ;
- `docs/architecture.md` et `docs/journal_decisions.md` : l'architecture, et la justification de chaque choix (D1 à D21) ;
- le code : `src/avions/` (les fonctions) et `scripts/` (les commandes).

## Dans quel ordre réviser

1. **Le projet** : parties A et B du guide. Tu dois pouvoir raconter les 6 étapes dans l'ordre.
2. **Les 12 notions** : partie C du guide, puis les fiches pour les revoir vite. Elles sont tirées au sort pendant les questions individuelles.
3. **Le code** : partie E du guide, avec les fichiers ouverts dans VS Code (`ouvrir_le_code.bat`). Pour chaque fichier, sache dire ce qu'il fait en une phrase.
4. **L'oral** : partie D du guide, et répéter la démo jusqu'à la faire sans réfléchir.

## La démo, en 2 minutes

1. L'interface s'ouvre sur l'image ★ 3803. « Détecter », puis « Corriger » : 13 justes, 1 A19 qui était un A1, 3 A1 oubliés. C'est le lien avec l'analyse des données : A1 et A19 se ressemblent et partagent les mêmes bases.
2. Fais défiler la galerie presque jusqu'en bas et clique sur l'image 3734, puis « Corriger » : 9 avions sur 9, mais 2 doublons. Deux avions ont chacun reçu deux boîtes, et la deuxième boîte de chacun compte comme une erreur.
3. « Image au hasard », puis « Corriger », deux ou trois fois : le F1 cumulé, en haut, se rapproche de notre score officiel. Il ne compte que les images corrigées au seuil de 0,5.
4. Reviens sur ★ 3803 (première vignette), baisse le seuil à 0,3, puis « Corriger » : 17 réponses au lieu de 14, mais 4 fausses au lieu de 1 (les 3 A1 oubliés reviennent, classés A19). Plus de réponses, mais plus d'erreurs : c'est le compromis entre précision et rappel. À ce seuil, l'image n'entre pas dans le F1 cumulé ; remets ensuite le curseur à 0,5.
5. Conclus sur le score officiel : F1 de 0,954 au test, contre 0,912 pour le témoin.

## Le jour de l'oral

- [ ] Le rapport (`docs/rapport.pdf`, avec le nom du binôme) est envoyé à mail@julien-breton.fr **au moins 24 h avant**.
- [ ] Le dataset du prof est bien dans `data/raw/images/` et `data/raw/annotations/` : sans lui, la démo n'a pas d'images.
- [ ] La veille, tu as lancé `lancer_demo.bat` et testé les images 3803 et 3734.
- [ ] Le PC est chargé et tu as le chargeur. Sur le projecteur, passe le navigateur en plein écran (F11).
- [ ] Le code est ouvert dans VS Code (`ouvrir_le_code.bat`), prêt à être montré.
- [ ] Les diapositives montrent des schémas plutôt que du code. Tu as chronométré 10 minutes, et la parole est partagée avec ton binôme.
