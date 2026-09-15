"""
VisionFlow-ObjectDetection-Pro
Core Detection Engine with YOLOv8 & Region of Interest (ROI) Analytics
"""

import time
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
from ultralytics import YOLO


class VisionFlowDetector:
    """
    Production-grade object detection engine wrapping YOLOv8.
    Supports confidence filtering, class filtering, latency telemetry, and ROI analytics.
    """

    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cpu"):
        """
        Initialize the YOLOv8 model.
        Args:
            model_name: Name of model weights ('yolov8n.pt', 'yolov8s.pt', etc.)
            device: 'cpu' or 'cuda' (auto-detected if available)
        """
        self.model_name = model_name
        self.device = device
        print(f"[VisionFlow] Loading model: {self.model_name} on device: {self.device}...")
        self.model = YOLO(self.model_name)
        self.class_names = self.model.names  # Dict of {class_id: class_name}

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        selected_classes: Optional[List[str]] = None,
        roi_polygon: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, float]]:
        """
        Run inference on a single image frame.
        
        Args:
            image: BGR numpy array (OpenCV format)
            conf_threshold: Minimum confidence score (0.0 to 1.0)
            iou_threshold: Non-Maximum Suppression (NMS) IoU threshold
            selected_classes: Optional list of class names to filter (e.g. ['person', 'car'])
            roi_polygon: Optional polygon (Nx2 array of points) for Zone analytics

        Returns:
            annotated_image: BGR image with drawn detections
            detections: List of detection dictionaries with metadata
            metrics: Dictionary of execution latency (ms) and FPS
        """
        start_time = time.perf_counter()

        # Run YOLO inference
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        raw_boxes = result.boxes
        inference_latency_ms = (time.perf_counter() - start_time) * 1000
        fps = 1000.0 / inference_latency_ms if inference_latency_ms > 0 else 0.0

        detections = []
        annotated_image = image.copy()

        # Extract timing metrics from ultralytics if available
        preprocess_ms = result.speed.get("preprocess", 0.0)
        inference_ms = result.speed.get("inference", inference_latency_ms)
        postprocess_ms = result.speed.get("postprocess", 0.0)

        metrics = {
            "total_latency_ms": round(inference_latency_ms, 2),
            "preprocess_ms": round(preprocess_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "postprocess_ms": round(postprocess_ms, 2),
            "fps": round(fps, 1),
        }

        if raw_boxes is None or len(raw_boxes) == 0:
            return annotated_image, detections, metrics

        # Parse detected boxes
        for idx, box in enumerate(raw_boxes):
            cls_id = int(box.cls[0].item())
            class_name = self.class_names.get(cls_id, f"class_{cls_id}")
            confidence = float(box.conf[0].item())

            # Apply class filter if specified
            if selected_classes is not None and class_name not in selected_classes:
                continue

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Check Region of Interest (ROI) inclusion if polygon provided
            in_roi = False
            if roi_polygon is not None:
                # Point-in-polygon test (positive if inside, 0 on edge, negative outside)
                dist = cv2.pointPolygonTest(roi_polygon, (float(cx), float(cy)), False)
                in_roi = dist >= 0

            detections.append({
                "detection_id": idx + 1,
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(confidence, 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": [cx, cy],
                "width": int(x2 - x1),
                "height": int(y2 - y1),
                "area_px": int((x2 - x1) * (y2 - y1)),
                "in_roi": in_roi,
            })

        return annotated_image, detections, metrics

    def get_available_classes(self) -> List[str]:
        """Return list of all classes supported by the loaded model."""
        return list(self.class_names.values())
