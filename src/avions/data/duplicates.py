"""Repérage des images quasi identiques (quasi-doublons).

Deux images quasi identiques (même scène, ou même base à une autre date) doivent tomber
dans le même jeu (train, validation ou test). Sinon le modèle serait évalué sur une image
qu'il a pratiquement déjà vue : c'est une fuite de données. Voir l'EDA, partie 2.4.
"""

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from avions.constants import RAW_IMAGES_DIR

# Distance maximale (en bits sur 256) pour considérer deux images comme quasi identiques.
# Pendant l'EDA, les vraies paires vues à l'œil étaient à 46, 72 et 73 bits. On garde une marge :
# regrouper par erreur deux images différentes ne coûte presque rien, alors que séparer deux
# quasi-doublons crée une fuite de données.
NEAR_DUPLICATE_MAX_DISTANCE = 80


def difference_hash(gray: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """Empreinte visuelle (dHash) : image réduite, chaque pixel comparé à son voisin de droite."""
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def near_duplicate_groups(
    image_ids: Iterable[str],
    images_dir: Path = RAW_IMAGES_DIR,
    max_distance: int = NEAR_DUPLICATE_MAX_DISTANCE,
) -> pd.Series:
    """Donne à chaque image un numéro de groupe : les images quasi identiques ont le même numéro.

    On construit un graphe où deux images sont reliées si leurs empreintes sont proches.
    Un groupe = un ensemble d'images reliées entre elles (une « composante connexe »).
    Une image sans quasi-doublon forme un groupe à elle seule.
    """
    image_ids = list(image_ids)
    hashes = np.array(
        [
            difference_hash(cv2.imread(str(Path(images_dir) / f"{i}.jpg"), cv2.IMREAD_GRAYSCALE))
            for i in image_ids
        ],
        dtype=np.int32,
    )
    # Distance de Hamming entre toutes les paires d'images (bits différents), comme dans l'EDA
    distances = hashes.shape[1] - (hashes @ hashes.T + (1 - hashes) @ (1 - hashes).T)
    _, labels = connected_components(csr_matrix(distances <= max_distance), directed=False)
    return pd.Series(labels, index=image_ids, name="group")
