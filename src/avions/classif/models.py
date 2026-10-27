"""Les réseaux de classification comparés dans le projet.

- resnet18, pré-entraîné sur ImageNet : notre classifieur principal (TRANSFER LEARNING).
  On garde son « backbone », qui sait déjà voir des formes, et on remplace sa dernière
  couche (1000 classes ImageNet) par une couche à 20 sorties (A1 à A20).
- resnet18 sans pré-entraînement : même réseau, poids au hasard -> mesure l'apport du
  transfer learning.
- smallcnn : petit CNN « maison » entraîné de zéro -> témoin simple, pour observer
  l'overfitting.
"""

from itertools import pairwise

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from avions.constants import CLASSES


class SmallCNN(nn.Module):
    """4 blocs (convolution, normalisation, ReLU, réduction de taille), puis une couche finale."""

    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()
        blocks = []
        channels = [3, 32, 64, 128, 256]
        for c_in, c_out in pairwise(channels):  # (3, 32), (32, 64), (64, 128), (128, 256)
            blocks += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),  # détecte des motifs locaux
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),  # divise la taille par 2 : 224 -> 112 -> 56 -> 28 -> 14
            ]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # moyenne de chaque carte : 256 nombres par image
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_model(architecture: str, pretrained: bool) -> nn.Module:
    """Construit le réseau demandé, avec une sortie par classe (20)."""
    if architecture == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))  # nouvelle « tête » à 20 sorties
        return model
    if architecture == "smallcnn":
        if pretrained:
            raise ValueError("Le petit CNN maison n'a pas de version pré-entraînée.")
        return SmallCNN()
    raise ValueError(f"Architecture inconnue : {architecture}")


def head_parameters(model: nn.Module):
    """Les paramètres de la dernière couche (la « tête »), les seuls entraînés pendant le gel."""
    return model.fc.parameters() if hasattr(model, "fc") else model.head.parameters()
