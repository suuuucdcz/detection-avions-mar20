"""Tests de l'export des prédictions au format Pascal VOC."""

import xml.etree.ElementTree as ET

from avions.data.voc import read_voc_file
from avions.export import write_voc
from avions.pipeline import Detection


def test_export_relu_comme_une_annotation(tmp_path):
    detections = [
        Detection((10.4, 20.6, 110.2, 70.8), "A2", 0.91, 0.95, 0.96),
        Detection((700.0, 500.0, 800.0, 600.0), "A17", 0.42, 0.88, 0.48, to_review=True),
    ]
    xml_path = tmp_path / "42.xml"

    write_voc(xml_path, "42.jpg", 800, 600, detections)

    # Notre lecteur d'annotations du dataset le relit sans problème
    rows = read_voc_file(xml_path)
    assert [r["class_name"] for r in rows] == ["A2", "A17"]
    assert (rows[0]["xmin"], rows[0]["ymin"], rows[0]["xmax"], rows[0]["ymax"]) == (10, 21, 110, 71)
    assert rows[0]["truncated"] is False
    assert rows[1]["truncated"] is True  # la boîte touche le bord droit et le bord bas
    assert rows[1]["image_width"] == 800 and rows[1]["image_height"] == 600

    # Les deux balises ajoutées : confiance et « à vérifier »
    objects = ET.parse(xml_path).getroot().findall("object")
    assert objects[0].findtext("score") == "0.9100"
    assert objects[0].findtext("a_verifier") == "0"
    assert objects[1].findtext("a_verifier") == "1"
