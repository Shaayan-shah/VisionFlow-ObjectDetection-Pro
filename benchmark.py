"""
VisionFlow-ObjectDetection-Pro
Automated Inference & Performance Benchmarking Suite

Profiles YOLOv8 inference latency (pre-process, neural inference, post-process),
FPS throughput, memory consumption, and detection statistics across real-world
benchmark photographic datasets.
"""

import os
import sys
import time
import json
import argparse
import platform
import psutil
import cv2
import numpy as np

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.core.detector import VisionFlowDetector


def get_system_specs():
    """Gathers host hardware and environment specifications."""
    mem = psutil.virtual_memory()
    return {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor() or "Unknown CPU",
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(mem.total / (1024 ** 3), 2),
        "available_ram_gb": round(mem.available / (1024 ** 3), 2)
    }


def run_benchmark(
    model_path: str = "yolov8n.pt",
    device: str = "cpu",
    conf_threshold: float = 0.35,
    iterations: int = 3,
    output_report_path: str = "data/exports/benchmark_report.json"
):
    print("=" * 70)
    print("  VisionFlow-ObjectDetection-Pro | Performance & Inference Benchmark")
    print("=" * 70)

    # 1. System Info
    specs = get_system_specs()
    print("\n[Environment Specifications]")
    print(f"  OS              : {specs['os']}")
    print(f"  Python          : {specs['python_version']}")
    print(f"  Processor       : {specs['processor']} ({specs['cpu_cores_logical']} logical cores)")
    print(f"  Total Memory    : {specs['total_ram_gb']} GB (Available: {specs['available_ram_gb']} GB)")

    # 2. Load Detector
    print("\n[Model Initialization]")
    print(f"  Loading Weights : {model_path}")
    print(f"  Inference Device: {device.upper()}")
    init_start = time.time()
    detector = VisionFlowDetector(model_name=model_path, device=device)
    init_duration_ms = (time.time() - init_start) * 1000
    print(f"  Model Loaded    : Ready in {init_duration_ms:.1f} ms")

    # 3. Locate Sample Images
    samples_dir = os.path.join(os.path.dirname(__file__), "data", "samples")
    if not os.path.exists(samples_dir):
        print(f"Error: Samples directory not found at {samples_dir}")
        return None

    sample_files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not sample_files:
        print("Error: No sample images found for benchmarking.")
        return None

    print("\n[Dataset Verification]")
    print(f"  Benchmark Target: {len(sample_files)} real-world photographic scenes")
    print(f"  Iterations / Img: {iterations} passes (averaged)")
    print(f"  Confidence Cutoff: {conf_threshold * 100:.0f}%\n")

    # 4. Warm-up Run (to eliminate JIT / cache cold start)
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = detector.detect(dummy_img, conf_threshold=conf_threshold)

    # 5. Benchmark Execution
    results = []
    total_detections_all = 0
    all_latencies = []

    print("-" * 70)
    print(f"{'Sample Image':<28} | {'Detections':<10} | {'Avg Latency (ms)':<16} | {'FPS':<6}")
    print("-" * 70)

    for sf in sample_files:
        img_path = os.path.join(samples_dir, sf)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        iter_latencies = []
        last_detections = []
        last_metrics = {}

        for _ in range(iterations):
            t0 = time.time()
            _, detections, metrics = detector.detect(img, conf_threshold=conf_threshold)
            t1 = time.time()
            elapsed_ms = (t1 - t0) * 1000
            iter_latencies.append(elapsed_ms)
            last_detections = detections
            last_metrics = metrics

        avg_lat = float(np.mean(iter_latencies))
        fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0
        all_latencies.append(avg_lat)
        total_detections_all += len(last_detections)

        classes_found = {}
        for d in last_detections:
            cname = d.get("class_name", "unknown")
            classes_found[cname] = classes_found.get(cname, 0) + 1

        summary_classes = ", ".join(f"{k}({v})" for k, v in sorted(classes_found.items()))

        results.append({
            "filename": sf,
            "resolution": f"{w}x{h}",
            "iterations": iterations,
            "avg_latency_ms": round(avg_lat, 2),
            "min_latency_ms": round(float(np.min(iter_latencies)), 2),
            "max_latency_ms": round(float(np.max(iter_latencies)), 2),
            "fps": round(fps, 1),
            "detection_count": len(last_detections),
            "classes_detected": classes_found,
            "metrics": last_metrics
        })

        print(f"{sf:<28} | {len(last_detections):<10} | {avg_lat:<16.2f} | {fps:<6.1f}")
        if summary_classes:
            print(f"  -> Detected: {summary_classes}")

    print("-" * 70)

    overall_avg_lat = float(np.mean(all_latencies)) if all_latencies else 0.0
    overall_fps = 1000.0 / overall_avg_lat if overall_avg_lat > 0 else 0.0

    print("\n[Overall Benchmark Summary]")
    print(f"  Total Images Processed : {len(results)}")
    print(f"  Total Targets Detected : {total_detections_all}")
    print(f"  Mean Latency Across Set: {overall_avg_lat:.2f} ms")
    print(f"  Aggregate Throughput   : {overall_fps:.1f} FPS")

    # 6. Save JSON Report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_path,
        "device": device,
        "confidence_threshold": conf_threshold,
        "iterations_per_sample": iterations,
        "environment": specs,
        "overall_summary": {
            "total_images": len(results),
            "total_detections": total_detections_all,
            "mean_latency_ms": round(overall_avg_lat, 2),
            "throughput_fps": round(overall_fps, 1)
        },
        "per_sample_results": results
    }

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("\n[Saved Benchmark Report]")
    print(f"  JSON Artifact: {os.path.abspath(output_report_path)}")
    print("=" * 70)
    return report_data


def main():
    parser = argparse.ArgumentParser(description="VisionFlow Performance & Latency Benchmark")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device: 'cpu' or 'cuda'")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--iterations", type=int, default=3, help="Benchmark timing passes per image")
    parser.add_argument("--output", type=str, default="data/exports/benchmark_report.json", help="Path for JSON output")

    args = parser.parse_args()
    run_benchmark(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        iterations=args.iterations,
        output_report_path=args.output
    )


if __name__ == "__main__":
    main()
