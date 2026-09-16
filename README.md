# 🎯 VisionFlow-ObjectDetection-Pro
### *Production-Grade Real-Time Object Detection & Visual Analytics Suite*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-00FFFF.svg?logo=yolo)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8.svg?logo=opencv)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**VisionFlow-ObjectDetection-Pro** is an end-to-end computer vision and spatial visual analytics platform built on top of **YOLOv8** and **OpenCV**. Designed for real-world production applications in **Smart Cities, Traffic Monitoring, Retail Analytics, and Security Surveillance**, VisionFlow provides real-time multi-class object detection, dynamic Region of Interest (ROI) zone monitoring, live inference telemetry, and structured data export.

---

## 🌟 Key Features

* **⚡ Real-Time YOLOv8 Inference**: Supports all 80 COCO object classes with high-precision bounding boxes, confidence scoring, and low latency.
* **📸 Multi-Modal Real-Data Ingestion (Front & Center)**:
  * **📤 Upload Your Own Photos**: Drag-and-drop your own real-world images (JPEG, PNG, WEBP) with single or batch multi-image preview.
  * **🌐 Direct Image URL Ingestion**: Paste any direct public image URL from the web (Unsplash, Google, Wikimedia) for instant analysis.
  * **🎥 Video File Ingestion**: Upload MP4, AVI, MOV clips with customizable frame skipping and live frame-by-frame progress.
  * **🔴 Live Webcam**: Real-time snapshot analysis directly through your browser or device camera.
  * **📂 5 Verified Photographic Benchmark Datasets Included**:
    * 🚌 *Urban City Bus & Pedestrians* (`city_bus_traffic.jpg`)
    * 🐕 *Dog, Bicycle & Pickup Truck* (`dog_bicycle_car.jpg`)
    * 🚶 *Street Pedestrians & Animal Scene* (`pedestrians_street.jpg`)
    * 🐎 *Wild Horses in Nature Field* (`horses_field.jpg`)
    * 🚗 *Highway Cars & Commuters* (`traffic_cars_highway.jpg`)
* **📐 Region of Interest (ROI) Zone Monitoring**:
  * Define custom spatial zones (e.g. entrance gates, crosswalks, restricted areas).
  * Point-in-polygon collision testing with dynamic occupancy counters and overcapacity alert beacons.
* **📊 Tactical HUD & Live Analytics Dashboard**:
  * Real-time metrics: Active Targets, Unique Classes, Inference FPS, and Latency Profiling (Pre-process, Neural Inference, Post-process).
  * Target class frequency distribution bar charts and interactive telemetry data tables.
* **💾 Enterprise Telemetry Export**:
  * **1-Click High-Res PNG**: Export annotated frames with tactical corner brackets and confidence badges.
  * **CSV Telemetry**: Tabular logs containing timestamps, bounding box coordinates `[x1, y1, x2, y2]`, pixel area, and zone status.
  * **JSON Metadata**: Full hierarchical payload for downstream MES/SCADA integration.
* **💻 Dual Interfaces**: Modern **Streamlit Web GUI** and a lightweight headless **CLI tool**.

---

## 🏗️ Architecture Overview

```
                           ┌──────────────────────────────────────────────┐
                           │          INPUT MEDIA INGESTION               │
                           │(Upload Own Image | URL | Video | WebCam | DB)│
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
For batch processing images without the web interface:

```bash
# Detect objects on real-world photographic benchmark image
python cli.py --source data/samples/city_bus_traffic.jpg --conf 0.35 --save-img --save-csv

# Run detection on your own photo
python cli.py --source "path/to/your_photo.jpg" --save-img --save-json
```

---

## 📁 Repository Structure

```
VisionFlow-ObjectDetection-Pro/
│
├── 📂 data/
│   ├── 📂 samples/             # 5 Verified Real-World Benchmark Photographic Datasets
│   │   ├── city_bus_traffic.jpg
│   │   ├── dog_bicycle_car.jpg
│   │   ├── horses_field.jpg
│   │   ├── pedestrians_street.jpg
│   │   └── traffic_cars_highway.jpg
│   └── 📂 exports/             # Output directory for telemetry logs & annotated images
│
├── 📂 src/
│   ├── 📂 core/
│   │   ├── __init__.py
│   │   └── detector.py         # YOLOv8 engine wrapper with spatial ROI analytics
│   └── 📂 utils/
│       ├── __init__.py
│       ├── visualizer.py       # Tactical HUD, corner brackets, and color palette
│       └── exporter.py         # CSV, JSON, and PNG telemetry generators
│
├── 📄 app.py                   # Streamlit production web application with multi-tab upload
├── 📄 cli.py                   # Headless batch processing CLI
├── 📄 test_suite.py            # Automated verification test suite on real datasets
├── 📄 requirements.txt         # Pinned Python package dependencies
├── 📄 run.bat                  # One-click Windows application launcher
├── 📄 yolov8n.pt               # Pre-cached lightweight YOLOv8 weights (ready out of the box)
├── 📄 .gitignore               # Git configuration
├── 📄 LICENSE                  # MIT License
└── 📄 README.md                # Comprehensive technical documentation
```

---

## 🧪 Verification & Automated Testing

VisionFlow includes a self-contained test suite that automatically checks model weights, runs benchmark inference on all real-world photographic datasets, tests zone spatial intersection algorithms, and verifies telemetry export:

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
