import os
import sys
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Add src to path
sys.path.append(os.path.dirname(__file__))

from src.detect_plate import PlateDetector
from src.crop_plate import crop_plate
from src.ocr_plate import PlateOCR
from src.plate_postprocess import parse_plate


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="License Plate Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    :root {
        --bg: #f4f7fb;
        --bg2: #eef3ff;
        --panel: rgba(255,255,255,0.88);
        --panel-strong: rgba(255,255,255,0.96);
        --text: #132033;
        --muted: #5f6f86;
        --border: rgba(80, 102, 144, 0.16);
        --shadow: 0 16px 40px rgba(17, 24, 39, 0.08);
        --primary: #4f46e5;
        --primary-2: #06b6d4;
        --primary-3: #8b5cf6;
        --sidebar-width: 23rem;
        --selected-bg: rgba(79, 70, 229, 0.12);
        --selected-border: rgba(79, 70, 229, 0.52);
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --bg: #07111f;
            --bg2: #0b1730;
            --panel: rgba(10, 18, 32, 0.88);
            --panel-strong: rgba(15, 23, 42, 0.95);
            --text: #edf4ff;
            --muted: #95a7c4;
            --border: rgba(148, 163, 184, 0.18);
            --shadow: 0 18px 44px rgba(0, 0, 0, 0.34);
            --primary: #7c83ff;
            --primary-2: #22d3ee;
            --primary-3: #a78bfa;
            --selected-bg: rgba(124, 131, 255, 0.16);
            --selected-border: rgba(124, 131, 255, 0.56);
        }
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    .main,
    .block-container,
    [data-testid="stMainBlockContainer"] {
        background:
            radial-gradient(circle at top left, rgba(79, 70, 229, 0.14), transparent 26%),
            radial-gradient(circle at top right, rgba(6, 182, 212, 0.10), transparent 24%),
            linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%) !important;
        color: var(--text) !important;
    }

    .block-container,
    [data-testid="stMainBlockContainer"] {
        max-width: none !important;
        width: 100% !important;
        padding-top: 3rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        margin-left: 0 !important;
        margin-right: 0 !important;

    }

    [data-testid="stAppViewContainer"] > .main {
        flex: 1 1 auto !important;
        max-width: none !important;
    }

    /* Sidebar mở */
    section[data-testid="stSidebar"][aria-expanded="true"] {
        width: var(--sidebar-width) !important;
        min-width: var(--sidebar-width) !important;
        max-width: var(--sidebar-width) !important;
        background: linear-gradient(180deg, var(--panel-strong) 0%, var(--panel) 100%) !important;
        border-right: 1px solid var(--border);
    }
    
    /* Main khi sidebar mở */
    section[data-testid="stSidebar"][aria-expanded="true"] ~ div[data-testid="stAppViewContainer"] > .main {
        width: calc(100vw - var(--sidebar-width)) !important;
        max-width: calc(100vw - var(--sidebar-width)) !important;
        margin-left: 0 !important;
    }

    /* Sidebar đóng */
    section[data-testid="stSidebar"][aria-expanded="false"] {
        width: 0 !important;
        min-width: 0 !important;
        max-width: 0 !important;
        overflow: hidden !important;
        border-right: none !important;
    }

    /* Main full màn hình khi sidebar đóng */
    section[data-testid="stSidebar"][aria-expanded="false"] ~ div[data-testid="stAppViewContainer"] > .main {
        width: 100vw !important;
        max-width: 100vw !important;
        margin-left: 0 !important;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] ~ div[data-testid="stAppViewContainer"] [data-testid="stMainBlockContainer"] {
        width: 100% !important;
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
    }


    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    @media (max-width: 900px) {
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: auto !important;
            min-width: auto !important;
            max-width: none !important;
        }
        
        section[data-testid="stSidebar"][aria-expanded="true"] ~ div[data-testid="stAppViewContainer"] > .main,
        section[data-testid="stSidebar"][aria-expanded="false"] ~ div[data-testid="stAppViewContainer"] > .main {
            width: 100% !important;
            max-width: 100% !important;
        }

        .block-container,
        [data-testid="stMainBlockContainer"] {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }

    .app-title {
        font-size: 2.2rem;
        font-weight: 900;
        line-height: 1.08;
        letter-spacing: -0.03em;
        margin-bottom: 0.35rem;
        background: linear-gradient(135deg, var(--primary), var(--primary-2), var(--primary-3));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .app-subtitle {
        color: var(--muted) !important;
        font-size: 0.98rem;
        margin-bottom: 1.2rem;
    }

    .pipeline-wrap {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        align-items: center;
        margin: 0.2rem 0 1.2rem 0;
    }

    .pipeline-pill {
        background: var(--panel-strong);
        border: 1px solid var(--border);
        color: var(--text) !important;
        padding: 0.58rem 0.95rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 700;
        white-space: nowrap;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
    }

    .panel-block {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 1rem;
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
    }

    .panel-title {
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 0.85rem;
        color: var(--text) !important;
    }

    .viewer-image-wrap {
        width: 100%;
        border-radius: 18px;
        overflow: hidden;
        background: var(--panel-strong);
        border: 1px solid var(--border);
    }

    .viewer-image-wrap img {
        width: 100%;
        display: block;
        object-fit: contain;
    }

    .stats-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0 0.25rem 0;
        flex-wrap: wrap;
    }

    .stat-card {
        flex: 1;
        min-width: 180px;
        background: var(--panel-strong);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem;
    }

    .stat-label {
        font-size: 0.8rem;
        color: var(--muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 800;
    }

    .stat-value {
        font-size: 1.55rem;
        font-weight: 900;
        margin-top: 0.2rem;
        color: var(--text) !important;
    }

    .empty-box {
        background: var(--panel);
        border: 1px dashed var(--border);
        border-radius: 22px;
        padding: 3rem 1.2rem;
        text-align: center;
        color: var(--muted) !important;
        box-shadow: var(--shadow);
    }

    .loading-box {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.8rem 1rem;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
    }

    .loading-circle {
        width: 58px;
        height: 58px;
        border: 5px solid rgba(148,163,184,0.18);
        border-top: 5px solid var(--primary);
        border-right: 5px solid var(--primary-2);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto 0.9rem auto;
    }

    .loading-title {
        font-size: 1.06rem;
        font-weight: 900;
        color: var(--text) !important;
    }

    .loading-subtitle {
        color: var(--muted) !important;
        margin-top: 0.25rem;
        font-size: 0.92rem;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .section-title {
        font-size: 1.08rem;
        font-weight: 800;
        margin: 1rem 0 0.75rem 0;
        color: var(--text) !important;
    }

    .thumb-scroll-note {
        color: var(--muted) !important;
        font-size: 0.92rem;
        margin-bottom: 0.6rem;
    }

    .thumb-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 20px;

        padding: 16px 14px;        /* padding đều trên dưới */
        height: 300px;             /* cố định chiều cao */
        box-sizing: border-box;

        display: flex;             /* dùng flex để cân layout */
        flex-direction: column;
        justify-content: space-between;

        box-shadow: var(--shadow);
        backdrop-filter: blur(8px);
    }

    .thumb-card.selected {
        border-color: var(--selected-border);
        background: linear-gradient(180deg, var(--panel) 0%, var(--selected-bg) 100%);
    }

    .thumb-title {
        font-size: 0.92rem;
        font-weight: 900;
        color: var(--text) !important;
        margin-bottom: 0.7rem;
        text-align: center;
    }

    .thumb-media {
        width: 100%;
        height: 110px;
        border-radius: 14px;
        border: 1px solid var(--border);
        background: var(--panel-strong);
        overflow: hidden;
        margin-bottom: 0.7rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .thumb-media img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
    }

    .thumb-plate {
        width: 100%;
        font-family: "Courier New", monospace;
        font-size: 1.04rem;
        font-weight: 900;
        background: var(--panel-strong);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.55rem 0.45rem;
        color: var(--text) !important;
        margin-bottom: 0.55rem;
        text-align: center;
    }

    .thumb-confidence {
        color: var(--muted) !important;
        font-size: 0.86rem;
        font-weight: 700;
        text-align: center;
    }

    .detail-button-wrap .stButton > button {
        width: 100% !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.72rem 0.9rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important;
        box-shadow: 0 12px 24px rgba(79, 70, 229, 0.18) !important;
    }

    .detail-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 1rem;
        margin-top: 1rem;
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
    }

    .detail-heading {
        font-size: 1.08rem;
        font-weight: 900;
        color: var(--text) !important;
        margin-bottom: 0.9rem;
    }

    .detail-grid {
        display: grid;
        grid-template-columns: minmax(240px, 340px) 1fr;
        gap: 1rem;
        align-items: start;
    }

    .detail-image-box {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid var(--border);
        background: var(--panel-strong);
    }

    .detail-image-box img {
        width: 100%;
        display: block;
        object-fit: contain;
    }

    .plate-badge {
        background: linear-gradient(135deg, var(--primary), var(--primary-2), var(--primary-3));
        color: #ffffff !important;
        border-radius: 18px;
        text-align: center;
        padding: 1rem;
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        font-family: "Courier New", monospace;
        margin-bottom: 1rem;
        box-shadow: 0 12px 28px rgba(79, 70, 229, 0.20);
    }

    .kv-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(180px, 1fr));
        gap: 0.8rem;
    }

    .kv {
        background: var(--panel-strong);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.9rem 1rem;
    }

    .kv-label {
        font-size: 0.8rem;
        color: var(--muted) !important;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.2rem;
    }

    .kv-value {
        font-size: 1rem;
        color: var(--text) !important;
        font-weight: 800;
        word-break: break-word;
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 14px;
        padding: 0.82rem 1rem;
        font-weight: 800;
        color: #ffffff !important;
        background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important;
        box-shadow: 0 12px 24px rgba(79, 70, 229, 0.18);
    }

    .stFileUploader {
        background: var(--panel-strong);
        border: 1px dashed var(--border);
        border-radius: 16px;
        padding: 0.5rem;
    }

    .stAlert {
        border-radius: 16px;
    }

    [data-testid="StyledFullScreenButton"] {
        display: none !important;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 900px) {
        .detail-grid {
            grid-template-columns: 1fr;
        }

        .kv-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)


# =========================
# SESSION STATE
# =========================
defaults = {
    "results": [],
    "processed_image": None,
    "selected_plate": None,
    "is_processing": False,
    "uploaded_name": None,
    "uploaded_bytes": None,
    "uploaded_type": None,
    "media_kind": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================
# HELPERS
# =========================
@st.cache_resource
def load_models():
    model_path = "models/plate_yolo.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None

    detector = PlateDetector(model_path)
    ocr = PlateOCR()
    return detector, ocr


def process_image(image, detector, ocr):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    plates = detector.detect(img_cv)
    results = []

    for i, plate_info in enumerate(plates):
        bbox = plate_info["bbox"]
        conf = plate_info["conf"]

        plate_img = crop_plate(img_cv, bbox)
        texts = ocr.read_text(plate_img)
        plate_number = parse_plate(texts)
        plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)

        results.append({
            "id": i,
            "bbox": bbox,
            "conf": conf,
            "raw_ocr": texts,
            "plate_number": plate_number if plate_number else "N/A",
            "cropped_image": plate_rgb
        })

    return img_cv, results


def draw_bounding_boxes(img_cv, results, selected_id=None):
    img_display = img_cv.copy()

    for result in results:
        x1, y1, x2, y2 = result["bbox"]
        plate_number = result["plate_number"]
        conf = result["conf"]
        plate_id = result["id"]

        if selected_id is not None and plate_id == selected_id:
            color = (255, 120, 0)
            thickness = 4
        else:
            color = (99, 76, 242)
            thickness = 2

        cv2.rectangle(img_display, (x1, y1), (x2, y2), color, thickness)

        text = f"{plate_number} ({conf:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        top_y = max(y1 - text_height - 10, 0)

        cv2.rectangle(
            img_display,
            (x1, top_y),
            (x1 + text_width + 8, y1),
            color,
            -1
        )

        cv2.putText(
            img_display,
            text,
            (x1 + 4, max(y1 - 5, text_height + 4)),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )

    return cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)


def image_to_html_img(np_img, alt="image"):
    pil_img = Image.fromarray(np_img)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = __import__("base64").b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" />'


def pil_to_html_img(pil_img, alt="image"):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = __import__("base64").b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" />'


def reset_result_state():
    st.session_state.results = []
    st.session_state.processed_image = None
    st.session_state.selected_plate = None
    st.session_state.is_processing = False


def save_uploaded_file_state(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_type = uploaded_file.type or ""
    media_kind = "video" if file_type.startswith("video/") else "image"

    changed = (
        st.session_state.uploaded_name != uploaded_file.name
        or st.session_state.uploaded_bytes != file_bytes
    )

    if changed:
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.uploaded_bytes = file_bytes
        st.session_state.uploaded_type = file_type
        st.session_state.media_kind = media_kind
        reset_result_state()


def get_current_pil_image():
    if not st.session_state.uploaded_bytes:
        return None
    return Image.open(BytesIO(st.session_state.uploaded_bytes)).convert("RGB")


def render_loading_box():
    st.markdown("""
<div class="loading-box">
    <div class="loading-circle"></div>
    <div class="loading-title">Đang nhận diện biển số</div>
    <div class="loading-subtitle">YOLOv8 → Detect → Crop plate → OCR → Display results</div>
</div>
""", unsafe_allow_html=True)


def render_viewer_panel(image_rgb, title="Main content"):
    if image_rgb is None:
        st.markdown("""
<div class="empty-box">
    <h3 style="margin-bottom:0.4rem;">Chưa có dữ liệu đầu vào</h3>
    <div>Hãy tải ảnh hoặc video ở sidebar bên trái để bắt đầu.</div>
</div>
""", unsafe_allow_html=True)
        return

    img_tag = image_to_html_img(image_rgb, "main-image") if isinstance(image_rgb, np.ndarray) else pil_to_html_img(image_rgb, "main-image")

    st.markdown(f"""
<div class="panel-block">
    <div class="panel-title">{title}</div>
    <div class="viewer-image-wrap">
        {img_tag}
    </div>
</div>
""", unsafe_allow_html=True)


def render_stats(results, selected_plate):
    avg_conf = sum(r["conf"] for r in results) / len(results) if results else 0
    selected_text = f"#{selected_plate + 1}" if selected_plate is not None else "-"
    st.markdown(f"""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-label">Detected Plates</div>
        <div class="stat-value">{len(results)}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Selected Plate</div>
        <div class="stat-value">{selected_text}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Average Confidence</div>
        <div class="stat-value">{avg_conf:.1%}</div>
    </div>
</div>
""", unsafe_allow_html=True)


def render_thumb_card(plate, idx, selected=False):
    card_class = "thumb-card selected" if selected else "thumb-card"
    img_tag = image_to_html_img(plate["cropped_image"], f"plate-{idx}")
    st.markdown(f"""
<div class="{card_class}">
    <div class="thumb-title">Biển số #{idx + 1}</div>
    <div class="thumb-media">
        {img_tag}
    </div>
    <div class="thumb-plate">{plate["plate_number"]}</div>
    <div class="thumb-confidence">Độ tin cậy: {plate["conf"]:.2%}</div>
</div>
""", unsafe_allow_html=True)


def render_thumb_strip(results, selected_plate):
    st.markdown('<div class="section-title">Kết quả nhận diện</div>', unsafe_allow_html=True)
    st.markdown('<div class="thumb-scroll-note">Nhấn “Xem chi tiết” trong từng thumbnail để hiển thị thông tin bên dưới.</div>', unsafe_allow_html=True)

    cards_per_row = 7
    for row_start in range(0, len(results), cards_per_row):
        row_items = results[row_start:row_start + cards_per_row]
        cols = st.columns(cards_per_row, gap="small")

        for offset, plate in enumerate(row_items):
            idx = row_start + offset
            with cols[offset]:
                render_thumb_card(plate, idx, selected=(selected_plate == idx))
                st.markdown('<div class="detail-button-wrap">', unsafe_allow_html=True)
                if st.button("Xem chi tiết", key=f"view_plate_{idx}", use_container_width=True):
                    st.session_state.selected_plate = idx
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)


def render_detail_card(plate, index):
    raw_ocr_text = ", ".join(plate["raw_ocr"]) if plate["raw_ocr"] else "N/A"
    img_tag = image_to_html_img(plate["cropped_image"], "detail-plate")

    st.markdown(f"""
<div class="detail-card">
    <div class="detail-heading">Chi tiết biển số #{index + 1}</div>
    <div class="detail-grid">
        <div class="detail-image-box">
            {img_tag}
        </div>
        <div>
            <div class="plate-badge">{plate["plate_number"]}</div>
            <div class="kv-grid">
                <div class="kv">
                    <div class="kv-label">Số biển số</div>
                    <div class="kv-value">{plate["plate_number"]}</div>
                </div>
                <div class="kv">
                    <div class="kv-label">Tỉ lệ chính xác</div>
                    <div class="kv-value">{plate["conf"]:.2%}</div>
                </div>
                <div class="kv">
                    <div class="kv-label">Raw OCR</div>
                    <div class="kv-value">{raw_ocr_text}</div>
                </div>
                <div class="kv">
                    <div class="kv-label">Bounding Box</div>
                    <div class="kv-value">{plate["bbox"]}</div>
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================
# APP HEADER
# =========================
st.markdown('<div class="app-title">Vietnamese License Plate Recognition</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload image/video → detect license plates → crop plates → OCR → display results</div>',
    unsafe_allow_html=True
)
st.markdown("""
<div class="pipeline-wrap">
    <div class="pipeline-pill">1. Upload image / video</div>
    <div class="pipeline-pill">2. Detect license plates</div>
    <div class="pipeline-pill">3. Crop plates</div>
    <div class="pipeline-pill">4. OCR</div>
    <div class="pipeline-pill">5. Display results</div>
</div>
""", unsafe_allow_html=True)


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## Upload image / video")

    uploaded_file = st.file_uploader(
        "Chọn file từ máy",
        type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"],
        help="Hỗ trợ ảnh và video. Pipeline backend hiện tại đang xử lý trực tiếp cho ảnh."
    )

    if uploaded_file is not None:
        save_uploaded_file_state(uploaded_file)

    if st.session_state.uploaded_bytes:
        st.markdown("### Image preview" if st.session_state.media_kind == "image" else "### Video preview")

        if st.session_state.media_kind == "image":
            st.image(get_current_pil_image(), use_container_width=True)
        else:
            st.video(st.session_state.uploaded_bytes)

        st.caption(f"File: {st.session_state.uploaded_name}")

    start_disabled = st.session_state.uploaded_bytes is None

    if st.button("Bắt đầu nhận diện", disabled=start_disabled, use_container_width=True):
        if st.session_state.media_kind == "video":
            st.warning("UI đã hỗ trợ upload/preview video. Pipeline backend hiện tại trong app này mới chạy trực tiếp cho ảnh.")
        else:
            st.session_state.is_processing = True
            st.session_state.selected_plate = None
            st.rerun()

    st.markdown("---")
    st.markdown("### Trạng thái")
    if st.session_state.uploaded_bytes is None:
        st.info("Chưa có file nào được tải lên.")
    elif st.session_state.is_processing:
        st.warning("Hệ thống đang xử lý...")
    elif st.session_state.results:
        st.success(f"Đã nhận diện {len(st.session_state.results)} biển số.")
    else:
        st.info("Sẵn sàng nhận diện.")


# =========================
# PROCESSING
# =========================
if (
    st.session_state.uploaded_bytes
    and st.session_state.media_kind == "image"
    and st.session_state.is_processing
):
    render_loading_box()

    detector, ocr = load_models()

    if detector and ocr:
        image = get_current_pil_image()
        img_cv, results = process_image(image, detector, ocr)

        st.session_state.results = results
        st.session_state.processed_image = img_cv
        st.session_state.is_processing = False
        st.session_state.selected_plate = 0 if results else None
        st.rerun()
    else:
        st.session_state.is_processing = False


# =========================
# MAIN CONTENT
# =========================
if st.session_state.uploaded_bytes is None:
    st.markdown("""
<div class="empty-box">
    <h3 style="margin-bottom:0.4rem;">Chưa có dữ liệu đầu vào</h3>
    <div>Hãy tải ảnh hoặc video ở sidebar bên trái để bắt đầu.</div>
</div>
""", unsafe_allow_html=True)
else:
    if st.session_state.media_kind == "video":
        st.markdown("""
<div class="panel-block">
    <div class="panel-title">Main content</div>
</div>
""", unsafe_allow_html=True)
        st.video(st.session_state.uploaded_bytes)
        st.info("Giao diện đã hỗ trợ upload/preview video. Với code backend hiện tại, nhận diện trực tiếp mới áp dụng cho ảnh.")
    else:
        original_image = get_current_pil_image()

        if st.session_state.is_processing:
            render_viewer_panel(original_image, "Main content")
        elif st.session_state.processed_image is not None:
            img_display = draw_bounding_boxes(
                st.session_state.processed_image,
                st.session_state.results,
                st.session_state.selected_plate
            )
            render_viewer_panel(img_display, "Main content")
        else:
            render_viewer_panel(original_image, "Main content")

    if st.session_state.results:
        render_stats(st.session_state.results, st.session_state.selected_plate)
        render_thumb_strip(st.session_state.results, st.session_state.selected_plate)

        if st.session_state.selected_plate is not None:
            plate = st.session_state.results[st.session_state.selected_plate]
            render_detail_card(plate, st.session_state.selected_plate)

    elif st.session_state.uploaded_bytes and not st.session_state.is_processing:
        if st.session_state.media_kind == "image" and st.session_state.processed_image is not None:
            st.info("Không phát hiện được biển số nào trong ảnh.")