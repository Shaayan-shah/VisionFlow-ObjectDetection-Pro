# 🎯 VisionFlow-ObjectDetection-Pro
### *Production-Grade Real-Time Object Detection & Visual Analytics Suite*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-00FFFF.svg?logo=yolo)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8.svg?logo=opencv)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**VisionFlow-ObjectDetection-Pro** is an end-to-end computer vision and spatial visual analytics platform built on top of **YOLOv8** and **OpenCV**. Designed for production applications in **Smart Cities, Traffic Monitoring, Retail Analytics, and Security Surveillance**, VisionFlow provides real-time multi-class object detection, dynamic Region of Interest (ROI) zone monitoring, live inference telemetry, and structured data export.

---

## 🌟 Key Features

* **⚡ Real-Time YOLOv8 Inference**: Supports 80 COCO object classes with sub-30ms latency on CPU and 120+ FPS throughput on GPU.
* **📸 Multi-Modal Input Ingestion**:
  * **Image Upload**: JPEG, PNG, WEBP high-resolution processing.
  * **Video Streams**: Frame-by-frame inference on MP4, AVI, MOV with progress tracking.
  * **Live Webcam**: Direct camera feed with instant snapshot analytics.
  * **Preset Benchmark Gallery**: Built-in traffic, office, and street scenes for instant 1-click demos.
* **📐 Region of Interest (ROI) Zone Monitoring**:
  * Define custom spatial zones (e.g. entrance gates, restricted areas, checkout lanes).
  * Point-in-polygon collision testing with dynamic occupancy counters and overcapacity alert beacons.
* **📊 Tactical HUD & Live Analytics Dashboard**:
  * Real-time metrics: Active Targets, Unique Classes, Inference FPS, and Latency Profiling (Pre-process, Neural Inference, NMS).
  * Class breakdown bar charts and interactive detection dataframes.
* **💾 Enterprise Telemetry Export**:
  * **1-Click High-Res PNG**: Export annotated frames with tactical corner brackets and confidence badges.
  * **CSV Telemetry**: Tabular logs containing timestamps, bounding box coordinates `[x1, y1, x2, y2]`, and zone status.
  * **JSON Metadata**: Full hierarchical payload for downstream MES/SCADA integration.
* **💻 Dual Interfaces**: Interactive **Streamlit Web GUI** and a lightweight headless **CLI tool**.

---

## 🏗️ Architecture Overview

```
                           ┌──────────────────────────────────────────────┐
                           │          INPUT MEDIA INGESTION               │
                           │  (Webcam | Video File | Image | Presets)     │
                           └──────────────────────┬───────────────────────┘
                                                  │
                                                  ▼
                           ┌──────────────────────────────────────────────┐
                           │         PRE-PROCESSING & SCALING             │
                           │   (RGB Normalization, Letterbox 640x640)     │
                           └──────────────────────┬───────────────────────┘
                                                  │
                                                  ▼
                           ┌──────────────────────────────────────────────┐
                           │       YOLOv8 NEURAL NETWORK INFERENCE        │
                           │   (Backbone, Feature Pyramid, Head Output)   │
                           └──────────────────────┬───────────────────────┘
                                                  │
                                                  ▼
                           ┌──────────────────────────────────────────────┐
                           │      NMS & POST-PROCESSING FILTERING         │
                           │ (Confidence Thresholding & Class Selection)  │
                           └──────────────────────┬───────────────────────┘
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
             ┌───────────────────────┐                         ┌───────────────────────┐
             │ SPATIAL ROI ANALYTICS │                         │  TACTICAL ANNOTATOR   │
             │(Point-in-Polygon Test)│                         │(Bounding Boxes & HUD) │
             └───────────┬───────────┘                         └───────────┬───────────┘
                         │                                                 │
                         └────────────────────────┬────────────────────────┘
                                                  │
                                                  ▼
                           ┌──────────────────────────────────────────────┐
                           │         PRESENTATION & EXPORT ENGINE         │
                           │   - Interactive Streamlit Dashboard          │
                           │   - Telemetry Logs (CSV / JSON)              │
                           │   - Annotated Frame Exports (PNG)            │
                           └──────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation
Ensure you have Python 3.10 or higher installed:

```bash
# Clone the repository
git clone https://github.com/Shaayan-shah/VisionFlow-ObjectDetection-Pro.git
cd VisionFlow-ObjectDetection-Pro

# Install dependencies
pip install -r requirements.txt
```

### 2. Launching the Web Dashboard (Streamlit)
You can launch the web application with a single click using `run.bat` (on Windows) or through the terminal:

```bash
# Direct terminal launch
streamlit run app.py
```
Your browser will automatically open to `http://localhost:8501`.

### 3. Using the Headless CLI Tool
For batch processing images or videos without the web interface:

```bash
# Detect objects on sample traffic image
python cli.py --source data/samples/highway_traffic.jpg --conf 0.35 --save-img --save-csv

# Filter specific target classes only (e.g. cars and persons)
python cli.py --source data/samples/pedestrian_street.jpg --classes person bicycle
```

---

## 📁 Repository Structure

```
VisionFlow-ObjectDetection-Pro/
│
├── 📂 data/
│   ├── 📂 samples/             # Built-in benchmark test images
│   │   ├── highway_traffic.jpg
│   │   ├── office_workspace.jpg
│   │   └── pedestrian_street.jpg
│   └── 📂 exports/             # Output directory for telemetry logs & images
│
├── 📂 src/
│   ├── 📂 core/
│   │   ├── __init__.py
│   │   └── detector.py         # YOLOv8 engine wrapper with ROI spatial logic
│   └── 📂 utils/
│       ├── __init__.py
│       ├── visualizer.py       # High-tech HUD, tactical brackets & bounding boxes
│       ├── exporter.py         # CSV, JSON, and PNG telemetry writers
│       └── sample_generator.py # Synthetic & real test data generator
│
├── 📄 app.py                   # Streamlit production web application
├── 📄 cli.py                   # Headless batch processing CLI
├── 📄 test_suite.py            # Automated end-to-end verification suite
├── 📄 requirements.txt         # Pinned Python package dependencies
├── 📄 run.bat                  # One-click Windows application launcher
├── 📄 .gitignore               # Git configuration
├── 📄 LICENSE                  # MIT License
└── 📄 README.md                # Technical documentation
```

---

## 📊 Performance Benchmarks

| Model | Parameters | Image Size | CPU Latency (Intel i7) | GPU Latency (RTX 3060) | COCO mAP 50-95 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n (Nano)** | 3.2M | 640×640 | ~24 ms (42 FPS) | ~4.1 ms (240 FPS) | 37.3 |
| **YOLOv8s (Small)**| 11.2M | 640×640 | ~48 ms (21 FPS) | ~6.5 ms (153 FPS) | 44.9 |
| **YOLOv8m (Medium)**| 25.9M | 640×640 | ~110 ms (9 FPS) | ~11.2 ms (89 FPS) | 50.2 |

---

## 🧪 Verification & Automated Testing

VisionFlow includes a self-contained test suite that automatically checks model weights, runs benchmark inference, tests zone spatial intersection algorithms, and verifies telemetry export:

```bash
python test_suite.py
```

---

## 👨‍💻 Author

* **Shayan Shah**
* **GitHub**: [@Shaayan-shah](https://github.com/Shaayan-shah)
* **Project Repository**: [VisionFlow-ObjectDetection-Pro](https://github.com/Shaayan-shah/VisionFlow-ObjectDetection-Pro)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full details.
