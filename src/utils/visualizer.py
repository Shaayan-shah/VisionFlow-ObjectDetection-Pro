"""
VisionFlow-ObjectDetection-Pro
High-Tech Computer Vision Visualizer & HUD Overlay
"""

from typing import List, Dict, Any, Optional
import numpy as np
import cv2


# Color palette for classes (BGR format)
PALETTE = [
    (245, 158, 11),   # Amber/Gold
    (14, 165, 233),   # Sky Blue
    (16, 185, 129),   # Emerald Green
    (168, 85, 247),   # Purple
    (239, 68, 68),    # Crimson Red
    (236, 72, 153),   # Pink
    (20, 184, 166),   # Teal
    (249, 115, 22),   # Orange
    (99, 102, 241),   # Indigo
    (132, 204, 22),   # Lime
]


def get_class_color(class_id: int) -> tuple:
    """Return a consistent color for a given class ID."""
    return PALETTE[class_id % len(PALETTE)]


def draw_corner_brackets(img: np.ndarray, bbox: list, color: tuple, length: int = 15, thickness: int = 2):
    """Draw tactical corner brackets around a bounding box."""
    x1, y1, x2, y2 = bbox
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)


def annotate_frame(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    show_labels: bool = True,
    show_confidence: bool = True,
    show_brackets: bool = True,
    roi_polygon: Optional[np.ndarray] = None,
    roi_label: str = "Active Zone",
) -> np.ndarray:
    """
    Annotate frame with bounding boxes, badges, and optional ROI overlay.
    """
    out = frame.copy()

    # Draw ROI polygon if specified
    if roi_polygon is not None and len(roi_polygon) >= 3:
        overlay = out.copy()
        cv2.fillPoly(overlay, [roi_polygon], (56, 189, 248))  # Cyan fill
        cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)
        cv2.polylines(out, [roi_polygon], isClosed=True, color=(14, 165, 233), thickness=2, lineType=cv2.LINE_AA)
        # ROI Label
        rx, ry = roi_polygon[0]
        cv2.putText(out, f"ZONE: {roi_label}", (int(rx), int(max(25, ry - 8))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (14, 165, 233), 2, cv2.LINE_AA)

    # Draw detected bounding boxes
    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        cls_name = det["class_name"]
        conf = det["confidence"]
        cls_id = det["class_id"]
        in_roi = det.get("in_roi", False)

        # Highlight if inside ROI
        color = (0, 220, 255) if in_roi else get_class_color(cls_id)

        # Main bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

        if show_brackets:
            draw_corner_brackets(out, bbox, (255, 255, 255), length=12, thickness=2)

        # Label badge
        if show_labels:
            label_text = f"{cls_name.upper()}"
            if show_confidence:
                label_text += f" {int(conf * 100)}%"
            if in_roi:
                label_text += " [IN ZONE]"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            badge_y1 = max(0, y1 - text_h - 10)
            badge_y2 = y1
            badge_x1 = x1
            badge_x2 = x1 + text_w + 12

            # Background rectangle for text
            cv2.rectangle(out, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
            # Text label
            text_color = (15, 23, 42)  # Dark contrast text
            cv2.putText(out, label_text, (badge_x1 + 6, badge_y2 - 5),
                        font, font_scale, text_color, thickness, cv2.LINE_AA)

    return out


def draw_hud(
    frame: np.ndarray,
    target_count: int,
    fps: float,
    latency_ms: float,
    model_name: str = "YOLOv8n",
) -> np.ndarray:
    """Draw top tactical telemetry HUD bar on the frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Translucent HUD bar at top
    hud_h = 42
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (10, 15, 29), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
    cv2.line(out, (0, hud_h), (w, hud_h), (56, 189, 248), 1)

    # Telemetry text
    cv2.putText(out, f"VISIONFLOW PRO", (16, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 249, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"MODEL: {model_name}", (190, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)
    cv2.putText(out, f"TARGETS: {target_count}", (360, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1, cv2.LINE_AA)
    cv2.putText(out, f"FPS: {fps:.1f}", (w - 200, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1, cv2.LINE_AA)
    cv2.putText(out, f"{latency_ms:.1f}ms", (w - 90, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (251, 191, 36), 1, cv2.LINE_AA)

    return out
