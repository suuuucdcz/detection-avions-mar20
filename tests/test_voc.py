"""Tests de la lecture des annotations Pascal VOC."""

import numpy as np

from avions.constants import CLASSES
from avions.data.voc import load_annotations, read_voc_file
from avions.visualization import draw_boxes

EXEMPLE_XML = """<annotation>
    <filename>42.jpg</filename>
    <size><width>800</width><height>600</height><depth>3</depth></size>
    <object>
        <name>A2</name><truncated>0</truncated><difficult>0</difficult>
        <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>110</xmax><ymax>70</ymax></bndbox>
    </object>
    <object>
        <name>A17</name><truncated>1</truncated><difficult>0</difficult>
        <bndbox><xmin>700</xmin><ymin>500</ymin><xmax>800</xmax><ymax>600</ymax></bndbox>
    </object>
</annotation>"""


def test_lecture_d_un_fichier(tmp_path):
    xml_path = tmp_path / "42.xml"
    xml_path.write_text(EXEMPLE_XML)

    rows = read_voc_file(xml_path)

    assert len(rows) == 2
    assert rows[0]["image_id"] == "42"
    assert rows[0]["class_name"] == "A2"
    assert (rows[0]["xmin"], rows[0]["ymin"], rows[0]["xmax"], rows[0]["ymax"]) == (10, 20, 110, 70)
    assert rows[0]["truncated"] is False
    assert rows[1]["truncated"] is True
    assert rows[1]["image_width"] == 800 and rows[1]["image_height"] == 600


def test_colonnes_calculees(tmp_path):
    (tmp_path / "42.xml").write_text(EXEMPLE_XML)

    df = load_annotations(tmp_path)

    premier = df.iloc[0]
    assert premier["box_width"] == 100 and premier["box_height"] == 50
    assert premier["box_area"] == 5000
    assert premier["aspect_ratio"] == 2.0
    assert premier["center_x"] == 60 / 800


def test_dataset_complet():
    df = load_annotations()
    assert len(df) == 7870  # nombre d'avions annotés
    assert df["image_id"].nunique() == 1331
    assert set(df["class_name"]) == set(CLASSES)


def test_draw_boxes_ne_modifie_pas_l_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    resultat = draw_boxes(image, [(10, 10, 50, 50)], ["A1"])
    assert image.sum() == 0  # l'original est intact
    assert resultat.sum() > 0  # la copie contient la boîte
