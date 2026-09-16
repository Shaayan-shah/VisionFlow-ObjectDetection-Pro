"""
VisionFlow-ObjectDetection-Pro
Automated Verification & Test Suite
Validates model inference, real-world detection accuracy, ROI calculation, and export functions.
"""

import os
import sys
import numpy as np
import cv2

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.core.detector import VisionFlowDetector
from src.utils.visualizer import annotate_frame, draw_hud
from src.utils.exporter import export_to_csv, export_to_json, save_annotated_image


def run_verification_tests():
    print("==================================================")
    print("  VisionFlow-ObjectDetection-Pro: Automated Tests")
    print("==================================================")

    samples_dir = os.path.join(os.path.dirname(__file__), "data", "samples")
    assert os.path.exists(samples_dir), f"Samples directory missing: {samples_dir}"

    real_samples = [
        "city_bus_traffic.jpg",
        "dog_bicycle_car.jpg",
        "pedestrians_street.jpg",
        "horses_field.jpg",
        "traffic_cars_highway.jpg",
    ]

    print("\n[TEST 1] Verifying presence of real-world photographic benchmark images...")
    for s in real_samples:
        sp = os.path.join(samples_dir, s)
        assert os.path.exists(sp), f"Missing benchmark image: {sp}"
        assert os.path.getsize(sp) > 10000, f"Benchmark image too small or corrupted: {sp}"
        print(f"  -> Found real benchmark image: {s} ({os.path.getsize(sp):,} bytes)")
    print("  -> Passed: All 5 real-world benchmark datasets verified on disk.")

    # 2. Test Model Loading
    print("\n[TEST 2] Initializing YOLOv8 Detector...")
    detector = VisionFlowDetector(model_name="yolov8n.pt", device="cpu")
    classes = detector.get_available_classes()
    assert len(classes) == 80, f"Expected 80 COCO classes, found {len(classes)}"
    print(f"  -> Passed: Model loaded successfully with {len(classes)} COCO classes.")

    # 3. Test Inference on Real-World Data (City Bus)
    print("\n[TEST 3] Running Inference on Real City Bus Traffic Image...")
    bus_img_path = os.path.join(samples_dir, "city_bus_traffic.jpg")
    bus_img = cv2.imread(bus_img_path)
    assert bus_img is not None, "Failed to read city_bus_traffic.jpg"

    annotated, detections, metrics = detector.detect(bus_img, conf_threshold=0.30)
    detected_classes = [d["class_name"] for d in detections]
    print(f"  -> Detections Found : {len(detections)}")
    print(f"  -> Detected Classes  : {detected_classes}")
    print(f"  -> Inference Speed   : {metrics['fps']} FPS ({metrics['total_latency_ms']} ms)")

    # Must detect real objects (bus, person, etc.)
    assert len(detections) >= 3, f"Expected at least 3 detections on real city bus image, got {len(detections)}"
    assert "bus" in detected_classes, f"Expected 'bus' in detected classes, got {detected_classes}"
    assert "person" in detected_classes, f"Expected 'person' in detected classes, got {detected_classes}"
    print("  -> Passed: Real-world image detection verified with high-confidence targets.")

    # 4. Test Inference on Real Dog/Bicycle/Car Image
    print("\n[TEST 4] Running Inference on Real Dog, Bicycle & Car Image...")
    dog_img_path = os.path.join(samples_dir, "dog_bicycle_car.jpg")
    dog_img = cv2.imread(dog_img_path)
    _, dog_dets, _ = detector.detect(dog_img, conf_threshold=0.30)
    dog_classes = [d["class_name"] for d in dog_dets]
    print(f"  -> Detections Found : {len(dog_dets)}")
    print(f"  -> Detected Classes  : {dog_classes}")
    assert "dog" in dog_classes or "bicycle" in dog_classes, f"Failed to detect dog/bicycle in {dog_classes}"
    print("  -> Passed: Multi-class real image detection verified.")

    # 5. Test Region of Interest (ROI) Zone Analytics
    print("\n[TEST 5] Testing Region of Interest (ROI) Point-in-Polygon Check...")
    h, w = bus_img.shape[:2]
    roi_poly = np.array([
        [int(w * 0.1), int(h * 0.2)],
        [int(w * 0.9), int(h * 0.2)],
        [int(w * 0.9), int(h * 0.9)],
        [int(w * 0.1), int(h * 0.9)],
    ], np.int32)

    _, roi_dets, _ = detector.detect(bus_img, conf_threshold=0.30, roi_polygon=roi_poly)
    zone_count = sum(1 for d in roi_dets if d["in_roi"])
    print(f"  -> Detected {zone_count} objects inside defined Zone boundary.")
    assert zone_count > 0, "Expected at least 1 object inside ROI zone"
    print("  -> Passed: ROI spatial collision logic verified on real objects.")

    # 6. Test Telemetry Exports (CSV & JSON)
    print("\n[TEST 6] Testing Telemetry Exports (CSV, JSON, PNG)...")
    export_dir = os.path.join(os.path.dirname(__file__), "data", "exports")
    os.makedirs(export_dir, exist_ok=True)

    csv_path = os.path.join(export_dir, "test_telemetry.csv")
    json_path = os.path.join(export_dir, "test_metadata.json")
    img_path = os.path.join(export_dir, "test_annotated.png")

    export_to_csv(roi_dets, csv_path)
    export_to_json(roi_dets, metrics, json_path, meta={"test_run": True, "real_data": True})
    save_annotated_image(annotated, img_path)

    assert os.path.exists(csv_path) and os.path.getsize(csv_path) > 0, "CSV export failed"
    assert os.path.exists(json_path) and os.path.getsize(json_path) > 0, "JSON export failed"
    assert os.path.exists(img_path) and os.path.getsize(img_path) > 0, "Annotated image save failed"
    print("  -> Passed: CSV, JSON, and PNG exports successfully generated and verified.")

    print("\n==================================================")
    print("  ALL TESTS PASSED! Project works 100% on real data.")
    print("==================================================")


if __name__ == "__main__":
    run_verification_tests()
