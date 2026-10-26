"""Tests de la conversion Pascal VOC -> YOLO."""

import pandas as pd
import pytest

from avions.data.to_yolo import voc_to_yolo_lines


def test_conversion_d_une_boite():
    # L'exemple de l'EDA : un A2 de (275, 22) à (385, 149) dans une image 800 x 800
    boxes = pd.DataFrame(
        [
            {
                "class_name": "A2",
                "xmin": 275,
                "ymin": 22,
                "xmax": 385,
                "ymax": 149,
                "image_width": 800,
                "image_height": 800,
            }
        ]
    )
    (line,) = voc_to_yolo_lines(boxes)
    class_index, x, y, w, h = line.split()
    assert class_index == "1"  # A2 -> 1 (A1 -> 0)
    assert float(x) == pytest.approx(330 / 800)  # centre x = (275 + 385) / 2
    assert float(y) == pytest.approx(85.5 / 800)  # centre y = (22 + 149) / 2
    assert float(w) == pytest.approx(110 / 800)
    assert float(h) == pytest.approx(127 / 800)


def test_image_rectangulaire():
    # Largeur et hauteur de l'image différentes : x divisé par la largeur, y par la hauteur
    boxes = pd.DataFrame(
        [
            {
                "class_name": "A20",
                "xmin": 0,
                "ymin": 0,
                "xmax": 100,
                "ymax": 50,
                "image_width": 1000,
                "image_height": 500,
            }
        ]
    )
    (line,) = voc_to_yolo_lines(boxes)
    assert line == "19 0.050000 0.050000 0.100000 0.100000"
