"""Tests for annotation workflow parsers (CVAT, Label Studio, Roboflow)."""

import json
import os
import tempfile
import textwrap

import pytest

from app.pipelines.annotation_loader import (
    load_cvat,
    load_label_studio,
    load_roboflow,
    merge_annotations,
    BoundingBox,
    AnnotationRecord,
)


# ---------------------------------------------------------------------------
# CVAT XML tests
# ---------------------------------------------------------------------------

CVAT_XML_SAMPLE = textwrap.dedent("""\
    <?xml version="1.0" encoding="utf-8"?>
    <annotations>
      <image id="0" name="invoice_001.jpg" width="640" height="480">
        <box label="invoice" xtl="10" ytl="20" xbr="600" ybr="460" occluded="0"/>
        <box label="field_vendor" xtl="30" ytl="40" xbr="300" ybr="70" occluded="0"/>
        <polygon label="stamp" points="100,100;200,100;150,150"/>
      </image>
      <image id="1" name="receipt_001.png" width="320" height="800">
        <box label="receipt" xtl="0" ytl="0" xbr="320" ybr="800" occluded="0"/>
      </image>
    </annotations>
""")


class TestCVATLoader:
    def test_parses_boxes(self, tmp_path):
        xml_file = tmp_path / "cvat.xml"
        xml_file.write_text(CVAT_XML_SAMPLE)
        records = load_cvat(str(xml_file))
        assert len(records) == 4  # 2 boxes + 1 polygon + 1 box

    def test_correct_labels(self, tmp_path):
        xml_file = tmp_path / "cvat.xml"
        xml_file.write_text(CVAT_XML_SAMPLE)
        records = load_cvat(str(xml_file))
        labels = {r.label for r in records}
        assert "invoice" in labels
        assert "field_vendor" in labels
        assert "receipt" in labels

    def test_source_tool(self, tmp_path):
        xml_file = tmp_path / "cvat.xml"
        xml_file.write_text(CVAT_XML_SAMPLE)
        records = load_cvat(str(xml_file))
        assert all(r.source_tool == "cvat" for r in records)

    def test_bbox_values(self, tmp_path):
        xml_file = tmp_path / "cvat.xml"
        xml_file.write_text(CVAT_XML_SAMPLE)
        records = load_cvat(str(xml_file))
        invoice_box = next(r for r in records if r.label == "invoice")
        assert invoice_box.bbox.x_min == 10
        assert invoice_box.bbox.y_min == 20
        assert invoice_box.bbox.x_max == 600
        assert invoice_box.bbox.y_max == 460

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_cvat("/nonexistent/path/annotations.xml")


# ---------------------------------------------------------------------------
# Label Studio JSON tests
# ---------------------------------------------------------------------------

LABEL_STUDIO_SAMPLE = [
    {
        "id": 1,
        "data": {"image": "invoice_002.jpg"},
        "annotations": [
            {
                "result": [
                    {
                        "type": "rectanglelabels",
                        "original_width": 640,
                        "original_height": 480,
                        "value": {
                            "rectanglelabels": ["invoice"],
                            "x": 5.0,   # 5% of 640 = 32
                            "y": 4.0,   # 4% of 480 = 19.2
                            "width": 90.0,  # 90% of 640 = 576
                            "height": 92.0, # 92% of 480 = 441.6
                            "rotation": 0,
                        },
                    }
                ]
            }
        ],
    }
]


class TestLabelStudioLoader:
    def test_parses_records(self, tmp_path):
        json_file = tmp_path / "ls_export.json"
        json_file.write_text(json.dumps(LABEL_STUDIO_SAMPLE))
        records = load_label_studio(str(json_file))
        assert len(records) == 1

    def test_label_value(self, tmp_path):
        json_file = tmp_path / "ls_export.json"
        json_file.write_text(json.dumps(LABEL_STUDIO_SAMPLE))
        records = load_label_studio(str(json_file))
        assert records[0].label == "invoice"

    def test_bbox_pixel_conversion(self, tmp_path):
        json_file = tmp_path / "ls_export.json"
        json_file.write_text(json.dumps(LABEL_STUDIO_SAMPLE))
        records = load_label_studio(str(json_file))
        bbox = records[0].bbox
        assert abs(bbox.x_min - 32.0) < 1e-3
        assert abs(bbox.y_min - 19.2) < 1e-3

    def test_source_tool(self, tmp_path):
        json_file = tmp_path / "ls_export.json"
        json_file.write_text(json.dumps(LABEL_STUDIO_SAMPLE))
        records = load_label_studio(str(json_file))
        assert all(r.source_tool == "label_studio" for r in records)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_label_studio("/nonexistent/export.json")


# ---------------------------------------------------------------------------
# Roboflow YOLO tests
# ---------------------------------------------------------------------------

class TestRoboflowLoader:
    def test_parses_yolo_labels(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "img001.txt").write_text("0 0.5 0.5 0.8 0.9\n")
        (labels_dir / "img002.txt").write_text("1 0.3 0.4 0.5 0.6\n")

        records = load_roboflow(str(labels_dir))
        assert len(records) == 2

    def test_class_id_mapping(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "img001.txt").write_text("0 0.5 0.5 0.8 0.9\n")
        (labels_dir / "img002.txt").write_text("1 0.3 0.4 0.5 0.6\n")

        records = load_roboflow(str(labels_dir), class_map={0: "invoice", 1: "receipt"})
        labels = {r.label for r in records}
        assert "invoice" in labels
        assert "receipt" in labels

    def test_source_tool(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "img001.txt").write_text("0 0.5 0.5 0.8 0.9\n")
        records = load_roboflow(str(labels_dir))
        assert all(r.source_tool == "roboflow" for r in records)

    def test_invalid_directory(self):
        with pytest.raises(NotADirectoryError):
            load_roboflow("/nonexistent/labels/")


# ---------------------------------------------------------------------------
# Merge test
# ---------------------------------------------------------------------------

class TestMergeAnnotations:
    def test_merge(self):
        a = [AnnotationRecord("img1.jpg", "invoice", None, "cvat")]
        b = [AnnotationRecord("img2.jpg", "receipt", None, "label_studio")]
        merged = merge_annotations(a, b)
        assert len(merged) == 2
        sources = {r.source_tool for r in merged}
        assert sources == {"cvat", "label_studio"}
