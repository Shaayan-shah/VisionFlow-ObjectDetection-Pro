"""
VisionFlow-ObjectDetection-Pro
Command-Line Interface (CLI) for Headless & Batch Object Detection
"""

import os
import sys
import argparse
import time
import cv2

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.core.detector import VisionFlowDetector
from src.utils.visualizer import annotate_frame, draw_hud
from src.utils.exporter import export_to_csv, export_to_json, save_annotated_image


def main():
    parser = argparse.ArgumentParser(description="VisionFlow-ObjectDetection-Pro CLI")
    parser.add_argument("--source", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 weights (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--classes", nargs="+", default=None, help="Specific classes to filter (e.g. person car)")
    parser.add_argument("--output", type=str, default="data/exports", help="Output directory for exports")
    parser.add_argument("--save-img", action="store_true", default=True, help="Save annotated image")
    parser.add_argument("--save-csv", action="store_true", default=True, help="Save detections to CSV")
    parser.add_argument("--save-json", action="store_true", default=True, help="Save detections to JSON")

    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"[ERROR] Source path does not exist: {args.source}")
        sys.exit(1)

    detector = VisionFlowDetector(model_name=args.model)

    # Process image
    print(f"\n[VisionFlow] Processing input: {args.source}")
    frame = cv2.imread(args.source)
    if frame is None:
        print(f"[ERROR] Could not read image at: {args.source}")
        sys.exit(1)

    _, detections, metrics = detector.detect(
        frame,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        selected_classes=args.classes
    )

    print(f"[VisionFlow] Detections Found : {len(detections)}")
    print(f"[VisionFlow] Inference Speed  : {metrics['fps']} FPS ({metrics['total_latency_ms']} ms)")

    for d in detections:
        print(f"  - [{d['class_name'].upper()}] Conf: {d['confidence']*100:.1f}% | BBox: {d['bbox']}")

    os.makedirs(args.output, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.source))[0]

    if args.save_img:
        annotated = annotate_frame(frame, detections)
        annotated = draw_hud(annotated, len(detections), metrics["fps"], metrics["total_latency_ms"], args.model)
        img_out = os.path.join(args.output, f"{base_name}_annotated.png")
        save_annotated_image(annotated, img_out)
        print(f"[VisionFlow] Saved Annotated Image : {img_out}")

    if args.save_csv and len(detections) > 0:
        csv_out = os.path.join(args.output, f"{base_name}_telemetry.csv")
        export_to_csv(detections, csv_out)
        print(f"[VisionFlow] Saved CSV Telemetry   : {csv_out}")

    if args.save_json and len(detections) > 0:
        json_out = os.path.join(args.output, f"{base_name}_metadata.json")
        export_to_json(detections, metrics, json_out, meta={"model": args.model, "source": args.source})
        print(f"[VisionFlow] Saved JSON Metadata  : {json_out}")

    print("\n[VisionFlow] Processing completed successfully.")


if __name__ == "__main__":
    main()
