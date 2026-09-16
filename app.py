"""
VisionFlow-ObjectDetection-Pro
Production-Grade Real-Time Object Detection & Visual Analytics Suite
Powered by YOLOv8, OpenCV, and Streamlit
"""

import os
import sys
import tempfile
import time
import urllib.request
from io import BytesIO
from typing import List, Dict, Any, Optional

import numpy as np
import cv2
import pandas as pd
from PIL import Image
import streamlit as st

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.core.detector import VisionFlowDetector
from src.utils.visualizer import annotate_frame, draw_hud
from src.utils.exporter import export_to_csv, export_to_json

# ---------------------------------------------------------
# Page Configuration & Modern Glassmorphic CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="VisionFlow Pro — Object Detection & Analytics Suite",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Main title and header styling */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .zone-alert {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        color: #f87171;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .zone-normal {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 12px 16px;
        color: #34d399;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 18px;
        background-color: #0f172a;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        font-weight: bold;
        border-bottom: 2px solid #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Model Caching (Loads model once into memory)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLOv8 Model Weights...")
def load_detector(model_name: str):
    # If model file is in project directory, use absolute path
    local_model = os.path.join(os.path.dirname(__file__), model_name)
    if os.path.exists(local_model):
        return VisionFlowDetector(model_name=local_model)
    return VisionFlowDetector(model_name=model_name)


# ---------------------------------------------------------
# Directory Constants
# ---------------------------------------------------------
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "data", "samples")


# ---------------------------------------------------------
# Sidebar Controls & Parameters
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ VisionFlow AI Settings")

model_choice = st.sidebar.selectbox(
    "AI Detection Model",
    ["yolov8n.pt", "yolov8s.pt"],
    index=0,
    help="yolov8n (Nano) is ultra-fast for real-time video; yolov8s (Small) offers higher precision."
)

detector = load_detector(model_choice)
all_classes = detector.get_available_classes()

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🎯 Detection Thresholds")
conf_thresh = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.30,
    step=0.05,
    help="Minimum confidence score required to display a detection box."
)

iou_thresh = st.sidebar.slider(
    "NMS (IoU) Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.45,
    step=0.05,
    help="Non-Maximum Suppression threshold to eliminate duplicate overlapping boxes."
)

st.sidebar.markdown("#### 🔍 Class Filter")
selected_classes = st.sidebar.multiselect(
    "Filter Specific Classes",
    options=sorted(all_classes),
    default=[],
    help="Select specific classes to detect (leave empty to detect all 80 COCO classes)"
)
filter_classes = selected_classes if len(selected_classes) > 0 else None

st.sidebar.markdown("---")
st.sidebar.markdown("#### 📐 Region of Interest (Zone Analytics)")
enable_roi = st.sidebar.checkbox("Enable Target Zone Monitoring", value=False)
roi_preset = "Center Stage"
roi_max_capacity = 3

if enable_roi:
    roi_preset = st.sidebar.selectbox(
        "Zone Layout Preset",
        ["Center Stage", "Left Lane / Gate", "Right Zone"]
    )
    roi_max_capacity = st.sidebar.number_input(
        "Zone Alarm Limit (Max Objects)",
        min_value=1,
        max_value=50,
        value=3,
        help="Triggers an alert when count of objects inside the zone exceeds this number."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🎨 HUD & Visual Overlays")
show_labels = st.sidebar.checkbox("Show Class Labels", value=True)
show_conf = st.sidebar.checkbox("Show Confidence %", value=True)
show_brackets = st.sidebar.checkbox("Show Tactical Corner Brackets", value=True)
show_hud_bar = st.sidebar.checkbox("Show Top Telemetry HUD", value=True)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def get_roi_polygon(preset_name: str, width: int, height: int) -> np.ndarray:
    if preset_name == "Center Stage":
        return np.array([
            [int(width * 0.25), int(height * 0.25)],
            [int(width * 0.75), int(height * 0.25)],
            [int(width * 0.75), int(height * 0.85)],
            [int(width * 0.25), int(height * 0.85)],
        ], np.int32)
    elif preset_name == "Left Lane / Gate":
        return np.array([
            [int(width * 0.05), int(height * 0.15)],
            [int(width * 0.45), int(height * 0.15)],
            [int(width * 0.45), int(height * 0.90)],
            [int(width * 0.05), int(height * 0.90)],
        ], np.int32)
    else:  # Right Zone
        return np.array([
            [int(width * 0.55), int(height * 0.15)],
            [int(width * 0.95), int(height * 0.15)],
            [int(width * 0.95), int(height * 0.90)],
            [int(width * 0.55), int(height * 0.90)],
        ], np.int32)


# ---------------------------------------------------------
# Core Analytics & Display Pipeline
# ---------------------------------------------------------
def process_and_display_image(bgr_image: np.ndarray, source_name: str = "Input Frame"):
    h, w = bgr_image.shape[:2]
    active_roi = get_roi_polygon(roi_preset, w, h) if enable_roi else None

    # Run YOLOv8 inference
    with st.spinner("Analyzing frame with YOLOv8..."):
        _, detections, metrics = detector.detect(
            bgr_image,
            conf_threshold=conf_thresh,
            iou_threshold=iou_thresh,
            selected_classes=filter_classes,
            roi_polygon=active_roi,
        )

    # Apply tactical visual annotations
    annotated = annotate_frame(
        bgr_image,
        detections,
        show_labels=show_labels,
        show_confidence=show_conf,
        show_brackets=show_brackets,
        roi_polygon=active_roi,
        roi_label=roi_preset if enable_roi else ""
    )

    if show_hud_bar:
        annotated = draw_hud(
            annotated,
            target_count=len(detections),
            fps=metrics["fps"],
            latency_ms=metrics["total_latency_ms"],
            model_name=model_choice
        )

    # 1. Top KPI Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(detections)}</div><div class="kpi-label">Targets Detected</div></div>', unsafe_allow_html=True)
    with c2:
        unique_classes = len(set(d["class_name"] for d in detections))
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{unique_classes}</div><div class="kpi-label">Unique Classes</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{metrics["fps"]:.1f}</div><div class="kpi-label">Inference FPS</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{metrics["total_latency_ms"]:.1f} ms</div><div class="kpi-label">Total Latency</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Zone Overcapacity Alert
    if enable_roi:
        zone_count = sum(1 for d in detections if d.get("in_roi", False))
        if zone_count > roi_max_capacity:
            st.markdown(f'<div class="zone-alert">🚨 ZONE OVERCAPACITY ALERT: {zone_count} objects inside {roi_preset} (Threshold: {roi_max_capacity})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="zone-normal">✅ ZONE OCCUPANCY NORMAL: {zone_count} objects inside {roi_preset} (Threshold: {roi_max_capacity})</div>', unsafe_allow_html=True)

    # 3. Viewport (Annotated vs Original vs Side-by-Side)
    tab_annotated, tab_original, tab_comparison = st.tabs(["🎯 Annotated View (HUD)", "🖼️ Original Input", "⚖️ Side-by-Side Comparison"])
    with tab_annotated:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
    with tab_original:
        st.image(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), use_column_width=True)
    with tab_comparison:
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Original Raw Input")
            st.image(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), use_column_width=True)
        with col_b:
            st.caption("VisionFlow Detection & HUD Overlay")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

    # 4. Analytics & Class Distribution
    st.markdown("### 📊 Detection Analytics & Breakdown")
    if len(detections) > 0:
        col_chart, col_table = st.columns([1, 2])
        with col_chart:
            st.markdown("##### Target Class Frequency")
            df_counts = pd.DataFrame(detections)["class_name"].value_counts().reset_index()
            df_counts.columns = ["Class", "Count"]
            st.bar_chart(df_counts.set_index("Class"), color="#38bdf8")

        with col_table:
            st.markdown("##### Real-Time Detection Telemetry Log")
            table_data = []
            for d in detections:
                table_data.append({
                    "ID": d["detection_id"],
                    "Class": d["class_name"].upper(),
                    "Confidence": f"{d['confidence'] * 100:.1f}%",
                    "Bounding Box (X1, Y1, X2, Y2)": f"{d['bbox']}",
                    "Center (X, Y)": f"{d['center']}",
                    "Area (px)": f"{d['area_px']:,}",
                    "In Zone": "YES" if d["in_roi"] else "NO"
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, height=280)
    else:
        st.info("ℹ️ No objects detected with current confidence threshold. Try lowering the slider in the sidebar.")

    # 5. One-Click Export Action Bar
    st.markdown("### 💾 Export Telemetry & Annotated Media")
    exp1, exp2, exp3 = st.columns(3)
    with exp1:
        _, buffer = cv2.imencode(".png", annotated)
        st.download_button(
            label="📥 Download Annotated Image (PNG)",
            data=buffer.tobytes(),
            file_name=f"visionflow_detection_{int(time.time())}.png",
            mime="image/png",
            use_container_width=True
        )
    with exp2:
        if len(detections) > 0:
            csv_df = pd.DataFrame([{
                "detection_id": d["detection_id"],
                "class_name": d["class_name"],
                "confidence": d["confidence"],
                "bbox": str(d["bbox"]),
                "center": str(d["center"]),
                "area_px": d["area_px"],
                "in_roi": d["in_roi"],
                "source": source_name
            } for d in detections])
            st.download_button(
                label="📄 Export Telemetry Log (CSV)",
                data=csv_df.to_csv(index=False),
                file_name=f"detections_log_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.button("📄 Export Telemetry Log (CSV)", disabled=True, use_container_width=True)
    with exp3:
        if len(detections) > 0:
            import json
            json_str = json.dumps({
                "timestamp": time.time(),
                "source": source_name,
                "metrics": metrics,
                "total_detections": len(detections),
                "detections": detections
            }, indent=2)
            st.download_button(
                label="📦 Export Full Metadata (JSON)",
                data=json_str,
                file_name=f"visionflow_telemetry_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.button("📦 Export Full Metadata (JSON)", disabled=True, use_container_width=True)


# ---------------------------------------------------------
# Main Page Header & Title
# ---------------------------------------------------------
st.markdown('<div class="main-header">🎯 VisionFlow Pro — Object Detection Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-grade Computer Vision & Spatial Analytics powered by YOLOv8 | Upload your own real data or test instant benchmark datasets</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# Front-and-Center Ingestion Tabs
# ---------------------------------------------------------
tab_upload, tab_url, tab_gallery, tab_video, tab_webcam = st.tabs([
    "📤 Upload Real Image(s)",
    "🌐 Analyze from Image URL",
    "📂 Real-World Benchmark Datasets",
    "🎥 Upload Video Stream",
    "🔴 Live Webcam Camera"
])

# ---------------------------------------------------------
# TAB 1: UPLOAD REAL IMAGE(S)
# ---------------------------------------------------------
with tab_upload:
    st.markdown("#### 📤 Upload Your Real-World Images for Instant Object Detection")
    st.caption("Drag and drop your own photos (JPEG, PNG, WEBP). Supports street photos, personal photos, surveillance footage, and store images.")
    
    col_up1, col_up2 = st.columns([3, 2])
    with col_up1:
        st.markdown("##### Option A: Browse Files or Drag & Drop")
        uploaded_files = st.file_uploader(
            "Choose one or more image files from your PC",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="main_image_uploader"
        )
    with col_up2:
        st.markdown("##### Option B: Enter Direct File Path on PC")
        local_path_input = st.text_input(
            "Paste full file path on your PC:",
            placeholder="D:\\MyPhotos\\example.jpg",
            key="local_pc_path_input"
        )
        load_local_btn = st.button("📂 Load & Detect from PC Path", key="btn_load_local")

    if uploaded_files:
        if len(uploaded_files) == 1:
            pil_img = Image.open(uploaded_files[0])
            bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            process_and_display_image(bgr, source_name=uploaded_files[0].name)
        else:
            st.success(f"Uploaded {len(uploaded_files)} images from your PC! Select an image below to analyze:")
            img_choice = st.selectbox("Select image to view", [f.name for f in uploaded_files])
            for f in uploaded_files:
                if f.name == img_choice:
                    pil_img = Image.open(f)
                    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    process_and_display_image(bgr, source_name=f.name)
                    break
    elif load_local_btn and local_path_input.strip():
        clean_path = local_path_input.strip().strip('"').strip("'")
        if os.path.exists(clean_path):
            bgr = cv2.imread(clean_path)
            if bgr is not None:
                st.success(f"Loaded local PC file: `{clean_path}`")
                process_and_display_image(bgr, source_name=os.path.basename(clean_path))
            else:
                st.error("Could not decode image at the specified path. Please ensure it is a valid image file.")
        else:
            st.error(f"File not found on your PC: `{clean_path}`. Please verify the path.")
    else:
        st.info("👆 Click **'Browse files'** to pick any photo from your computer, or paste a local file path above!")

# ---------------------------------------------------------
# TAB 2: ANALYZE FROM IMAGE URL
# ---------------------------------------------------------
with tab_url:
    st.markdown("#### 🌐 Analyze Any Online Image via Direct URL")
    st.caption("Paste a link to any public photo on the web (e.g., from Unsplash, Wikimedia, or Google Images).")

    url_input = st.text_input(
        "Image URL",
        placeholder="https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?w=800",
        key="image_url_input"
    )

    if st.button("🚀 Fetch & Analyze Online Image", key="btn_url_analyze"):
        if url_input.strip():
            try:
                with st.spinner("Downloading image from web..."):
                    req = urllib.request.Request(url_input.strip(), headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=12) as response:
                        image_data = response.read()
                    image_np = np.frombuffer(image_data, np.uint8)
                    bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

                    if bgr is not None:
                        process_and_display_image(bgr, source_name=url_input.strip())
                    else:
                        st.error("Could not decode image from the provided URL. Please verify the link is a direct image file.")
            except Exception as e:
                st.error(f"Error fetching image: {e}")
        else:
            st.warning("Please enter a valid image URL first.")

# ---------------------------------------------------------
# TAB 3: REAL-WORLD BENCHMARK DATASETS
# ---------------------------------------------------------
with tab_gallery:
    st.markdown("#### 📂 Verified Real-World Photographic Benchmark Datasets")
    st.caption("Select any of these verified photographic datasets to test multi-class detections instantly.")

    sample_options = {
        "🚌 Urban City Bus & Pedestrians": "city_bus_traffic.jpg",
        "🐕 Dog, Bicycle & Pickup Truck": "dog_bicycle_car.jpg",
        "🚶 Street Pedestrians & Animal Scene": "pedestrians_street.jpg",
        "🐎 Wild Horses in Nature Field": "horses_field.jpg",
        "🚗 Highway Cars & Commuters": "traffic_cars_highway.jpg",
    }

    selected_sample = st.selectbox("Choose a Benchmark Scene", list(sample_options.keys()))
    sample_filename = sample_options[selected_sample]
    sample_file_path = os.path.join(SAMPLES_DIR, sample_filename)

    if os.path.exists(sample_file_path):
        bgr = cv2.imread(sample_file_path)
        if bgr is not None:
            process_and_display_image(bgr, source_name=sample_filename)
    else:
        st.error(f"Benchmark file not found at: {sample_file_path}")

# ---------------------------------------------------------
# TAB 4: UPLOAD VIDEO STREAM
# ---------------------------------------------------------
with tab_video:
    st.markdown("#### 🎥 Upload Video for Frame-by-Frame Inference")
    st.caption("Upload MP4, AVI, or MOV video clips to perform sequential object detection.")

    uploaded_video = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov", "mkv"], key="video_uploader")

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()

        vf = cv2.VideoCapture(tfile.name)
        total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = vf.get(cv2.CAP_PROP_FPS) or 30.0

        st.info(f"Video Loaded: **{total_frames} Frames** (~{total_frames/video_fps:.1f} seconds at {video_fps:.0f} FPS).")

        skip_frames = st.slider("Frame Processing Interval", min_value=1, max_value=10, value=2,
                                help="Process every Nth frame for higher speed.")

        if st.button("🚀 Start Video Analysis", key="btn_start_video"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_slot = st.empty()

            frame_idx = 0
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % skip_frames == 0:
                    _, dets, mets = detector.detect(
                        frame,
                        conf_threshold=conf_thresh,
                        iou_threshold=iou_thresh,
                        selected_classes=filter_classes
                    )
                    annotated = annotate_frame(
                        frame,
                        dets,
                        show_labels=show_labels,
                        show_confidence=show_conf,
                        show_brackets=show_brackets
                    )
                    if show_hud_bar:
                        annotated = draw_hud(annotated, len(dets), mets["fps"], mets["total_latency_ms"], model_choice)

                    frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
                    progress_bar.progress(min(1.0, frame_idx / max(1, total_frames)))
                    status_text.caption(f"Analyzing Frame {frame_idx}/{total_frames} — {mets['fps']:.1f} FPS ({len(dets)} targets detected)")

            vf.release()
            st.success("🎉 Video Inference Completed Successfully!")

# ---------------------------------------------------------
# TAB 5: LIVE WEBCAM CAMERA
# ---------------------------------------------------------
with tab_webcam:
    st.markdown("#### 🔴 Live Webcam Snapshot Analysis")
    st.caption("Capture a real-time photo from your device's camera for immediate detection.")

    cam_picture = st.camera_input("Take a Snapshot from Camera", key="webcam_capture")
    if cam_picture is not None:
        pil_img = Image.open(cam_picture)
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        process_and_display_image(bgr, source_name="Webcam Snapshot")


# ---------------------------------------------------------
# Page Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem; padding: 10px;'>"
    "<b>VisionFlow-ObjectDetection-Pro v2.0</b> &bull; Developed by <b>Shayan Shah</b> &bull; "
    "Production Computer Vision &bull; <a href='https://github.com/Shaayan-shah/VisionFlow-ObjectDetection-Pro' target='_blank' style='color: #38bdf8;'>GitHub Repository</a> &bull; MIT License"
    "</div>",
    unsafe_allow_html=True
)
