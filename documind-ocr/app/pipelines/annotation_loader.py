"""
annotation_loader.py — Annotation Workflow Integration
=======================================================

Parsers for the three annotation platforms used in the DocuMind OCR pipeline:

* **CVAT** (XML export)          — polygon / bounding-box annotations
* **Label Studio** (JSON export) — rectangle label annotations
* **Roboflow** (YOLO .txt format)— normalised bounding-box annotations

All parsers normalise output to a common ``AnnotationRecord`` structure that
feeds directly into model training and evaluation workflows.

Usage
-----
>>> from app.pipelines.annotation_loader import load_cvat, load_label_studio, load_roboflow
>>> records = load_cvat("path/to/annotations.xml")
>>> records = load_label_studio("path/to/export.json")
>>> records = load_roboflow("path/to/labels/", class_map={0: "invoice", 1: "receipt"})
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any


# ---------------------------------------------------------------------------
# Common data model
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Axis-aligned bounding box in absolute pixel coordinates."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class AnnotationRecord:
    """Normalised annotation record produced by all loaders."""
    image_path: str               # Relative or absolute path to the source image
    label: str                    # Class label (e.g. 'invoice', 'receipt', 'field_vendor')
    bbox: Optional[BoundingBox]   # None for image-level labels
    source_tool: str              # 'cvat' | 'label_studio' | 'roboflow'
    attributes: Dict[str, Any] = field(default_factory=dict)  # Extra metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "label": self.label,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "source_tool": self.source_tool,
            "attributes": self.attributes,
        }


# ---------------------------------------------------------------------------
# CVAT XML parser
# ---------------------------------------------------------------------------

def load_cvat(xml_path: str) -> List[AnnotationRecord]:
    """
    Parse a CVAT XML annotation export.

    Supports both CVAT for Images 1.1 and CVAT for Video exports.
    Each ``<box>`` element inside a ``<image>`` or ``<track>`` tag is
    converted to an ``AnnotationRecord``.

    Parameters
    ----------
    xml_path : str
        Path to the CVAT-exported XML file.

    Returns
    -------
    List[AnnotationRecord]
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"CVAT annotation file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    records: List[AnnotationRecord] = []

    # --- CVAT for Images export structure ---
    for image_elem in root.findall(".//image"):
        img_name = image_elem.get("name", "unknown")
        img_width = float(image_elem.get("width", 0))
        img_height = float(image_elem.get("height", 0))

        for box_elem in image_elem.findall("box"):
            label = box_elem.get("label", "unknown")
            xtl = float(box_elem.get("xtl", 0))
            ytl = float(box_elem.get("ytl", 0))
            xbr = float(box_elem.get("xbr", 0))
            ybr = float(box_elem.get("ybr", 0))
            occluded = box_elem.get("occluded", "0") == "1"

            # Collect attribute children
            attrs: Dict[str, Any] = {
                "occluded": occluded,
                "image_width": img_width,
                "image_height": img_height,
            }
            for attr_elem in box_elem.findall("attribute"):
                attr_name = attr_elem.get("name", "")
                attrs[attr_name] = attr_elem.text

            records.append(AnnotationRecord(
                image_path=img_name,
                label=label,
                bbox=BoundingBox(x_min=xtl, y_min=ytl, x_max=xbr, y_max=ybr),
                source_tool="cvat",
                attributes=attrs,
            ))

        # Polygon / polyline annotations → use bounding rect
        for poly_elem in image_elem.findall("polygon") + image_elem.findall("polyline"):
            label = poly_elem.get("label", "unknown")
            points_str = poly_elem.get("points", "")
            if not points_str:
                continue
            pts = [
                (float(p.split(",")[0]), float(p.split(",")[1]))
                for p in points_str.split(";") if "," in p
            ]
            if len(pts) < 2:
                continue
            xs, ys = zip(*pts)
            records.append(AnnotationRecord(
                image_path=img_name,
                label=label,
                bbox=BoundingBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys)),
                source_tool="cvat",
                attributes={"type": poly_elem.tag},
            ))

    # --- CVAT for Video / track structure ---
    for track_elem in root.findall(".//track"):
        label = track_elem.get("label", "unknown")
        source = track_elem.get("source", "")
        for box_elem in track_elem.findall("box"):
            frame = box_elem.get("frame", "0")
            xtl = float(box_elem.get("xtl", 0))
            ytl = float(box_elem.get("ytl", 0))
            xbr = float(box_elem.get("xbr", 0))
            ybr = float(box_elem.get("ybr", 0))
            records.append(AnnotationRecord(
                image_path=f"frame_{frame}",
                label=label,
                bbox=BoundingBox(x_min=xtl, y_min=ytl, x_max=xbr, y_max=ybr),
                source_tool="cvat",
                attributes={"frame": frame, "source": source},
            ))

    return records


# ---------------------------------------------------------------------------
# Label Studio JSON parser
# ---------------------------------------------------------------------------

def load_label_studio(json_path: str) -> List[AnnotationRecord]:
    """
    Parse a Label Studio JSON export.

    Handles the ``RectangleLabels`` and ``PolygonLabels`` result types that
    are commonly used for document OCR annotation projects in Label Studio.

    Parameters
    ----------
    json_path : str
        Path to the Label Studio-exported JSON file.

    Returns
    -------
    List[AnnotationRecord]
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Label Studio export not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    records: List[AnnotationRecord] = []

    # Label Studio exports are a list of task objects
    if isinstance(data, dict) and "annotations" in data:
        # Single-task export wrapped in a dict
        tasks = [data]
    elif isinstance(data, list):
        tasks = data
    else:
        return records

    for task in tasks:
        # Resolve image path from the `data` field
        task_data = task.get("data", {})
        image_path: str = (
            task_data.get("image")
            or task_data.get("ocr")
            or task_data.get("file_upload", "unknown")
        )

        annotations = task.get("annotations", [])
        for annotation in annotations:
            results = annotation.get("result", [])
            for result in results:
                result_type = result.get("type", "")
                value = result.get("value", {})

                # Image dimensions provided as percentages in Label Studio
                img_width = result.get("original_width", 1)
                img_height = result.get("original_height", 1)

                if result_type == "rectanglelabels":
                    labels = value.get("rectanglelabels", [])
                    label = labels[0] if labels else "unknown"

                    # Label Studio uses % values (0-100)
                    x_pct = value.get("x", 0) / 100.0
                    y_pct = value.get("y", 0) / 100.0
                    w_pct = value.get("width", 0) / 100.0
                    h_pct = value.get("height", 0) / 100.0

                    x_min = x_pct * img_width
                    y_min = y_pct * img_height
                    x_max = x_min + w_pct * img_width
                    y_max = y_min + h_pct * img_height

                    records.append(AnnotationRecord(
                        image_path=image_path,
                        label=label,
                        bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                        source_tool="label_studio",
                        attributes={"rotation": value.get("rotation", 0)},
                    ))

                elif result_type == "polygonlabels":
                    labels = value.get("polygonlabels", [])
                    label = labels[0] if labels else "unknown"
                    pts = value.get("points", [])
                    if len(pts) < 2:
                        continue
                    xs = [p[0] / 100.0 * img_width for p in pts]
                    ys = [p[1] / 100.0 * img_height for p in pts]
                    records.append(AnnotationRecord(
                        image_path=image_path,
                        label=label,
                        bbox=BoundingBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys)),
                        source_tool="label_studio",
                        attributes={"type": "polygon"},
                    ))

                elif result_type == "taxonomy":
                    # Image-level label — no bbox
                    taxonomy_labels = value.get("taxonomy", [[]])
                    for tax_path in taxonomy_labels:
                        label = tax_path[-1] if tax_path else "unknown"
                        records.append(AnnotationRecord(
                            image_path=image_path,
                            label=label,
                            bbox=None,
                            source_tool="label_studio",
                        ))

    return records


# ---------------------------------------------------------------------------
# Roboflow YOLO parser
# ---------------------------------------------------------------------------

def load_roboflow(
    labels_dir: str,
    class_map: Optional[Dict[int, str]] = None,
    images_dir: Optional[str] = None,
) -> List[AnnotationRecord]:
    """
    Parse Roboflow YOLO-format label files (one ``.txt`` per image).

    Each line in a YOLO label file has the format::

        <class_id> <x_center> <y_center> <width> <height>

    All values (except class_id) are normalised to [0, 1].  To convert to
    absolute pixel coordinates the corresponding image dimensions are needed;
    if ``images_dir`` is provided the loader will read the image file to get
    dimensions, otherwise it records the normalised coordinates as-is.

    Parameters
    ----------
    labels_dir : str
        Directory containing ``.txt`` label files.
    class_map : dict, optional
        Mapping from integer class ID to string label name.
        Defaults to ``{0: "invoice", 1: "receipt", 2: "field"}``.
    images_dir : str, optional
        Directory containing corresponding image files.  When provided the
        loader reads actual image dimensions for pixel-space conversion.

    Returns
    -------
    List[AnnotationRecord]
    """
    if not os.path.isdir(labels_dir):
        raise NotADirectoryError(f"Roboflow labels directory not found: {labels_dir}")

    if class_map is None:
        class_map = {0: "invoice", 1: "receipt", 2: "field_vendor",
                     3: "field_date", 4: "field_amount"}

    records: List[AnnotationRecord] = []

    for fname in sorted(os.listdir(labels_dir)):
        if not fname.endswith(".txt"):
            continue

        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_dir, fname)

        # Try to find matching image to get real dimensions
        img_width, img_height = 1.0, 1.0
        image_path = stem  # fallback
        if images_dir:
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                candidate = os.path.join(images_dir, stem + ext)
                if os.path.exists(candidate):
                    image_path = candidate
                    img = cv2.imread(candidate)
                    if img is not None:
                        img_height, img_width = float(img.shape[0]), float(img.shape[1])
                    break

        with open(label_path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_c, y_c, w, h = (float(p) for p in parts[1:5])
                label = class_map.get(class_id, f"class_{class_id}")

                # Convert YOLO normalised → absolute pixel coordinates
                x_min = (x_c - w / 2) * img_width
                y_min = (y_c - h / 2) * img_height
                x_max = (x_c + w / 2) * img_width
                y_max = (y_c + h / 2) * img_height

                records.append(AnnotationRecord(
                    image_path=image_path,
                    label=label,
                    bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                    source_tool="roboflow",
                    attributes={"class_id": class_id, "normalised": images_dir is None},
                ))

    return records


# ---------------------------------------------------------------------------
# Utility: merge from multiple sources
# ---------------------------------------------------------------------------

def merge_annotations(*record_lists: List[AnnotationRecord]) -> List[AnnotationRecord]:
    """Merge annotation records from multiple tools into one list."""
    merged: List[AnnotationRecord] = []
    for lst in record_lists:
        merged.extend(lst)
    return merged
