"""Jeu de vignettes pour PyTorch : donne (image transformée, numéro de classe) à la demande."""

from collections.abc import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from avions.constants import CROPS_DIR, CROPS_INDEX


def load_crops_index(split: str | None = None) -> pd.DataFrame:
    """Lit data/crops/index.csv (une ligne par vignette), éventuellement filtré sur un jeu."""
    index = pd.read_csv(CROPS_INDEX, dtype={"image_id": str})
    return index if split is None else index[index["split"] == split].reset_index(drop=True)


class CropDataset(Dataset):
    """Les vignettes d'un jeu (train, val ou test), lues sur le disque une par une."""

    def __init__(self, index: pd.DataFrame, transform: Callable):
        self.paths = index["crop_path"].tolist()
        self.labels = index["label"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        image = Image.open(CROPS_DIR / self.paths[i]).convert("RGB")
        return self.transform(image), self.labels[i]
