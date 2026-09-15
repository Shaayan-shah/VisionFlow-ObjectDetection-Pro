"""
VisionFlow-ObjectDetection-Pro
Production-Grade Real-Time Object Detection & Visual Analytics Suite
Powered by YOLOv8, OpenCV, and Streamlit
"""

import os
import sys
import tempfile
import time
from io import BytesIO
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
from src.utils.sample_generator import generate_sample_datasets

# ---------------------------------------------------------
# Page Configuration & Custom CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="VisionFlow Pro — Object Detection & Analytics Suite",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Dark glassmorphic theme styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .zone-alert {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 12px;
        color: #f87171;
        font-weight: 600;
    }
    .zone-normal {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 12px;
        color: #34d399;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Model Caching (Loads model once into memory)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLOv8 Model Weights...")
def load_detector(model_name: str):
    return VisionFlowDetector(model_name=model_name)


# ---------------------------------------------------------
# Ensure Sample Datasets Exist
# ---------------------------------------------------------
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "data", "samples")
if not os.path.exists(os.path.join(SAMPLES_DIR, "highway_traffic.jpg")):
    generate_sample_datasets(SAMPLES_DIR)


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ VisionFlow Controls")

model_choice = st.sidebar.selectbox(
    "Select Model Weights",
    ["yolov8n.pt", "yolov8s.pt"],
    index=0,
    help="yolov8n is ultra-fast (nano), yolov8s is balanced (small)."
)

detector = load_detector(model_choice)
all_classes = detector.get_available_classes()

input_mode = st.sidebar.radio(
    "Input Source",
    ["📂 Preset Gallery", "📸 Upload Image", "🎥 Video File", "🔴 Live Webcam"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🎯 Detection Thresholds")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.35, 0.05)
iou_thresh = st.sidebar.slider("NMS (IoU) Threshold", 0.10, 1.00, 0.45, 0.05)

st.sidebar.markdown("#### 🔍 Class Filter")
selected_classes = st.sidebar.multiselect(
    "Filter Specific Classes",
    options=all_classes,
    default=[],
    help="Leave empty to detect all 80 COCO classes"
)
filter_classes = selected_classes if len(selected_classes) > 0 else None

st.sidebar.markdown("---")
st.sidebar.markdown("#### 📐 Region of Interest (Zone Analytics)")
enable_roi = st.sidebar.checkbox("Enable Target Zone Monitoring", value=False)
roi_polygon = None
roi_max_capacity = 3

if enable_roi:
    roi_preset = st.sidebar.selectbox(
        "Zone Preset",
        ["Center Stage", "Left Lane / Gate", "Right Zone", "Custom Sliders"]
    )
    roi_max_capacity = st.sidebar.number_input("Zone Capacity Alarm Limit", min_value=1, max_value=50, value=3)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🎨 Overlay Settings")
show_labels = st.sidebar.checkbox("Show Class Labels", value=True)
show_conf = st.sidebar.checkbox("Show Confidence %", value=True)
show_brackets = st.sidebar.checkbox("Show Tactical Brackets", value=True)
show_hud_bar = st.sidebar.checkbox("Show Top Telemetry HUD", value=True)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def get_roi_polygon(preset_name: str, width: int, height: int) -> np.ndarray:
    if preset_name == "Center Stage":
        pts = np.array([
            [int(width * 0.25), int(height * 0.25)],
            [int(width * 0.75), int(height * 0.25)],
            [int(width * 0.75), int(height * 0.85)],
            [int(width * 0.25), int(height * 0.85)],
        ], np.int32)
    elif preset_name == "Left Lane / Gate":
        pts = np.array([
            [int(width * 0.05), int(height * 0.20)],
            [int(width * 0.45), int(height * 0.20)],
            [int(width * 0.45), int(height * 0.90)],
            [int(width * 0.05), int(height * 0.90)],
        ], np.int32)
    else:  # Right Zone
        pts = np.array([
            [int(width * 0.55), int(height * 0.20)],
            [int(width * 0.95), int(height * 0.20)],
            [int(width * 0.95), int(height * 0.90)],
            [int(width * 0.55), int(height * 0.90)],
        ], np.int32)
    return pts


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">🎯 VisionFlow Pro — Object Detection Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-Time Computer Vision & Visual Analytics powered by YOLOv8 | COCO 80-Class Recognition</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# Process Frame Pipeline
# ---------------------------------------------------------
def process_single_image(bgr_image: np.ndarray):
    h, w = bgr_image.shape[:2]
    active_roi = None
    if enable_roi:
        active_roi = get_roi_polygon(roi_preset, w, h)

    # Run inference
    _, detections, metrics = detector.detect(
        bgr_image,
        conf_threshold=conf_thresh,
        iou_threshold=iou_thresh,
        selected_classes=filter_classes,
        roi_polygon=active_roi,
    )

    # Draw visual annotations
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

    # 1. Metrics Summary Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{len(detections)}</div><div class="metric-lbl">Total Detected</div></div>', unsafe_allow_html=True)
    with m2:
        unique_classes = len(set(d["class_name"] for d in detections))
        st.markdown(f'<div class="metric-card"><div class="metric-val">{unique_classes}</div><div class="metric-lbl">Unique Classes</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{metrics["fps"]}</div><div class="metric-lbl">Inference FPS</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{metrics["total_latency_ms"]} ms</div><div class="metric-lbl">Latency</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Zone Alert Check
    if enable_roi:
        zone_count = sum(1 for d in detections if d.get("in_roi", False))
        if zone_count > roi_max_capacity:
            st.markdown(f'<div class="zone-alert">🚨 ZONE OVERCAPACITY ALERT: {zone_count} objects inside {roi_preset} (Max Limit: {roi_max_capacity})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="zone-normal">✅ ZONE OCCUPANCY NORMAL: {zone_count} objects inside {roi_preset} (Max Limit: {roi_max_capacity})</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # 3. Viewport (Original vs Annotated)
    tab_annotated, tab_original, tab_comparison = st.tabs(["🎯 Annotated Detections", "🖼️ Original Image", "⚖️ Side-by-Side Comparison"])
    with tab_annotated:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
    with tab_original:
        st.image(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), use_column_width=True)
    with tab_comparison:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Input Image")
            st.image(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), use_column_width=True)
        with c2:
            st.caption("VisionFlow Detection HUD")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

    # 4. Analytics & Class Distribution
    st.markdown("### 📊 Detection Analytics & Breakdown")
    if len(detections) > 0:
        col_chart, col_table = st.columns([1, 2])
        with col_chart:
            df_counts = pd.DataFrame(detections)["class_name"].value_counts().reset_index()
            df_counts.columns = ["Class", "Count"]
            st.bar_chart(df_counts.set_index("Class"), color="#38bdf8")

        with col_table:
            table_data = []
            for d in detections:
                table_data.append({
                    "ID": d["detection_id"],
                    "Class": d["class_name"].upper(),
                    "Confidence": f"{d['confidence'] * 100:.1f}%",
                    "Bounding Box (X1, Y1, X2, Y2)": f"{d['bbox']}",
                    "Area (px)": f"{d['area_px']:,}",
                    "Inside Zone": "YES" if d["in_roi"] else "NO"
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    else:
        st.info("No targets detected matching current confidence threshold and filters.")

    # 5. Export Action Bar
    st.markdown("### 💾 Export & Telemetry Data")
    exp1, exp2, exp3 = st.columns(3)
    with exp1:
        # Export Annotated Frame
        _, buffer = cv2.imencode(".png", annotated)
        st.download_button(
            label="📥 Download Annotated Image (PNG)",
            data=buffer.tobytes(),
            file_name=f"visionflow_detection_{int(time.time())}.png",
            mime="image/png"
        )
    with exp2:
        # Export CSV
        if len(detections) > 0:
            csv_df = pd.DataFrame([{
                "detection_id": d["detection_id"],
                "class_name": d["class_name"],
                "confidence": d["confidence"],
                "bbox": str(d["bbox"]),
                "in_roi": d["in_roi"]
            } for d in detections])
            st.download_button(
                label="📄 Export Telemetry Log (CSV)",
                data=csv_df.to_csv(index=False),
                file_name=f"detections_log_{int(time.time())}.csv",
                mime="text/csv"
            )
    with exp3:
        # Export JSON
        if len(detections) > 0:
            import json
            json_str = json.dumps({"metrics": metrics, "detections": detections}, indent=2)
            st.download_button(
                label="📦 Export Full Metadata (JSON)",
                data=json_str,
                file_name=f"visionflow_telemetry_{int(time.time())}.json",
                mime="application/json"
            )


# ---------------------------------------------------------
# Input Mode Dispatcher
# ---------------------------------------------------------
if input_mode == "📂 Preset Gallery":
    preset_choice = st.selectbox(
        "Choose a Preset Test Scene",
        ["City Bus & Pedestrian Traffic (Real Urban Street)",
         "Highway Traffic Scene (Vehicles, Trucks)",
         "Modern Tech Workspace (Laptops, Monitors, Tech)",
         "Pedestrian Street Crossing (People, Bicycles)"]
    )

    sample_map = {
        "City Bus & Pedestrian Traffic (Real Urban Street)": "city_bus_traffic.jpg",
        "Highway Traffic Scene (Vehicles, Trucks)": "highway_traffic.jpg",
        "Modern Tech Workspace (Laptops, Monitors, Tech)": "office_workspace.jpg",
        "Pedestrian Street Crossing (People, Bicycles)": "pedestrian_street.jpg",
    }

    img_path = os.path.join(SAMPLES_DIR, sample_map[preset_choice])
    if os.path.exists(img_path):
        bgr = cv2.imread(img_path)
        process_single_image(bgr)

elif input_mode == "📸 Upload Image":
    uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        pil_img = Image.open(uploaded)
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        process_single_image(bgr)
    else:
        st.info("👆 Please upload a JPEG, PNG, or WEBP image above to begin real-time analysis.")

elif input_mode == "🎥 Video File":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        vf = cv2.VideoCapture(tfile.name)
        total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        st.info(f"Video Loaded: {total_frames} Frames | Click 'Start Video Inference' below.")

        if st.button("🚀 Start Video Inference"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_slot = st.empty()

            frame_idx = 0
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % 2 == 0:  # Sample every 2nd frame for speed
                    _, dets, mets = detector.detect(frame, conf_threshold=conf_thresh, iou_threshold=iou_thresh)
                    annotated = annotate_frame(frame, dets, show_labels=show_labels, show_confidence=show_conf)
                    annotated = draw_hud(annotated, len(dets), mets["fps"], mets["total_latency_ms"], model_choice)

                    frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
                    progress_bar.progress(min(1.0, frame_idx / max(1, total_frames)))
                    status_text.caption(f"Processing Frame {frame_idx}/{total_frames} — {mets['fps']} FPS")

            vf.release()
            st.success("🎉 Video Inference Complete!")

elif input_mode == "🔴 Live Webcam":
    st.markdown("#### Live Webcam Feed")
    cam_picture = st.camera_input("Take a Snapshot from Webcam for Instant Analysis")
    if cam_picture is not None:
        pil_img = Image.open(cam_picture)
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        process_single_image(bgr)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "VisionFlow-ObjectDetection-Pro &bull; Developed by <b>Shayan Shah</b> &bull; "
    "Powered by YOLOv8 &bull; MIT License"
    "</div>",
    unsafe_allow_html=True
)
