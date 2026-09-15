"""
VisionFlow-ObjectDetection-Pro
Telemetry & Detection Exporter (CSV, JSON, Images)
"""

import os
import json
import csv
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import cv2
import pandas as pd


def export_to_csv(detections: List[Dict[str, Any]], output_path: str) -> str:
    """Export detection records to a formatted CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    timestamp = datetime.now().isoformat()

    flattened = []
    for d in detections:
        bbox = d.get("bbox", [0, 0, 0, 0])
        center = d.get("center", [0, 0])
        flattened.append({
            "timestamp": timestamp,
            "detection_id": d.get("detection_id"),
            "class_id": d.get("class_id"),
            "class_name": d.get("class_name"),
            "confidence": d.get("confidence"),
            "bbox_x1": bbox[0],
            "bbox_y1": bbox[1],
            "bbox_x2": bbox[2],
            "bbox_y2": bbox[3],
            "center_x": center[0],
            "center_y": center[1],
            "width": d.get("width"),
            "height": d.get("height"),
            "area_px": d.get("area_px"),
            "in_roi": d.get("in_roi", False),
        })

    df = pd.DataFrame(flattened)
    df.to_csv(output_path, index=False)
    return output_path


def export_to_json(
    detections: List[Dict[str, Any]],
    metrics: Dict[str, float],
    output_path: str,
    meta: Dict[str, Any] = None
) -> str:
    """Export full detection telemetry and system metrics to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "total_detections": len(detections),
        "metrics": metrics,
        "metadata": meta or {},
        "detections": detections,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def save_annotated_image(image: np.ndarray, output_path: str) -> str:
    """Save annotated image frame to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    return output_path
