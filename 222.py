import streamlit as st
import streamlit.components.v1 as components

# Streamlit Cloud 的原生 OpenCV 錯誤可能被平台隱藏。
# 在載入失敗時，顯示真正缺少的共用函式庫及已安裝的套件版本。
def _show_native_import_diagnostic(module_name, err):
    from importlib import metadata
    import platform
    import sys
    st.set_page_config(page_title="RULA v6｜環境相容性檢查", layout="centered")
    st.error(f"環境初始化失敗：無法載入 {module_name}，RULA 分析尚未開始。")
    st.info("請複製下方完整錯誤文字，這一行才能確認是缺少系統函式庫，還是 NumPy／OpenCV 版本衝突。")
    st.code(f"{type(err).__name__}: {err}", language="text")
    installed = {}
    for dist in ("mediapipe", "numpy", "opencv-python", "opencv-python-headless",
                 "opencv-contrib-python", "opencv-contrib-python-headless", "streamlit"):
        try:
            installed[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            installed[dist] = "未安裝"
    st.write("Python：", sys.version.split()[0], "；作業系統：", platform.system())
    st.json(installed, expanded=False)
    st.warning("這是套件環境錯誤，與上傳影片、手部模型檔或 RULA 評分本身無關。")
    st.stop()

try:
    import cv2
except Exception as exc:
    _show_native_import_diagnostic("OpenCV (cv2)", exc)

try:
    import mediapipe as mp
except Exception as exc:
    _show_native_import_diagnostic("MediaPipe", exc)
import math
import numpy as np
import pandas as pd
import os
import urllib.request
import zipfile
import io
import gc
import logging
import base64
import tempfile
import time
import json
import re
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# Streamlit Cloud: keep MediaPipe legacy Pose at complexity 1.
# Complexity 2 asks MediaPipe to download pose_landmark_heavy.tflite into site-packages,
# which is typically read-only on hosted deployments.
POSE_MODEL_COMPLEXITY = 1
APP_BUILD_TAG = "RULA_V9_2_IMAGE_SYNC_WRIST_QC_2026-10-28"

# ==================== 【沿用 v6 RULA 查表矩陣：仍需標準工作表逐格核對】 ====================
TABLE_A = {
    1: { 1: {1: {1:1, 2:2}, 2: {1:2, 2:2}, 3: {1:2, 2:3}, 4: {1:3, 2:3}},
         2: {1: {1:2, 2:2}, 2: {1:2, 2:3}, 3: {1:3, 2:3}, 4: {1:3, 2:4}},
         3: {1: {1:2, 2:3}, 2: {1:3, 2:3}, 3: {1:3, 2:4}, 4: {1:4, 2:4}} },
    2: { 1: {1: {1:1, 2:2}, 2: {1:2, 2:2}, 3: {1:3, 2:3}, 4: {1:3, 2:4}},
         2: {1: {1:2, 2:3}, 2: {1:3, 2:3}, 3: {1:3, 2:4}, 4: {1:4, 2:4}},
         3: {1: {1:3, 2:4}, 2: {1:3, 2:4}, 3: {1:4, 2:4}, 4: {1:4, 2:5}} },
    3: { 1: {1: {1:2, 2:3}, 2: {1:3, 2:3}, 3: {1:3, 2:4}, 4: {1:4, 2:4}},
         2: {1: {1:3, 2:3}, 2: {1:3, 2:4}, 3: {1:4, 2:4}, 4: {1:4, 2:5}},
         3: {1: {1:3, 2:4}, 2: {1:4, 2:4}, 3: {1:4, 2:5}, 4: {1:5, 2:5}} },
    4: { 1: {1: {1:3, 2:4}, 2: {1:4, 2:4}, 3: {1:4, 2:5}, 4: {1:5, 2:5}},
         2: {1: {1:4, 2:4}, 2: {1:4, 2:5}, 3: {1:4, 2:5}, 4: {1:5, 2:6}},
         3: {1: {1:4, 2:5}, 2: {1:5, 2:5}, 3: {1:5, 2:6}, 4: {1:6, 2:7}} },
    5: { 1: {1: {1:4, 2:4}, 2: {1:4, 2:5}, 3: {1:4, 2:5}, 4: {1:5, 2:6}},
         2: {1: {1:4, 2:5}, 2: {1:5, 2:5}, 3: {1:5, 2:6}, 4: {1:6, 2:7}},
         3: {1: {1:5, 2:6}, 2: {1:6, 2:6}, 3: {1:6, 2:7}, 4: {1:7, 2:7}} },
    6: { 1: {1: {1:5, 2:5}, 2: {1:5, 2:6}, 3: {1:5, 2:6}, 4: {1:6, 2:7}},
         2: {1: {1:5, 2:6}, 2: {1:6, 2:6}, 3: {1:6, 2:7}, 4: {1:7, 2:7}},
         3: {1: {1:6, 2:7}, 2: {1:7, 2:7}, 3: {1:7, 2:8}, 4: {1:8, 2:8}} }
}

TABLE_B = {
    1: {1: {1:1, 2:3}, 2: {1:2, 2:3}, 3: {1:3, 2:4}, 4: {1:5, 2:5}, 5: {1:6, 2:6}, 6: {1:7, 2:7}},
    2: {1: {1:2, 2:3}, 2: {1:3, 2:3}, 3: {1:4, 2:4}, 4: {1:5, 2:5}, 5: {1:6, 2:7}, 6: {1:7, 2:8}},
    3: {1: {1:3, 2:3}, 2: {1:4, 2:4}, 3: {1:4, 2:5}, 4: {1:6, 2:6}, 5: {1:7, 2:7}, 6: {1:7, 2:8}},
    4: {1: {1:5, 2:5}, 2: {1:5, 2:6}, 3: {1:6, 2:7}, 4: {1:7, 2:7}, 5: {1:8, 2:8}, 6: {1:8, 2:8}},
    5: {1: {1:6, 2:6}, 2: {1:6, 2:7}, 3: {1:7, 2:8}, 4: {1:8, 2:8}, 5: {1:8, 2:8}, 6: {1:8, 2:8}},
    6: {1: {1:7, 2:7}, 2: {1:7, 2:8}, 3: {1:8, 2:8}, 4: {1:8, 2:8}, 5: {1:8, 2:8}, 6: {1:8, 2:8}}
}

TABLE_C = [
    [1, 2, 3, 3, 4, 5, 5], [2, 2, 3, 4, 4, 5, 5], [3, 3, 3, 4, 4, 5, 6], [3, 3, 3, 4, 5, 6, 6],
    [4, 4, 4, 5, 6, 7, 7], [4, 4, 5, 6, 6, 7, 7], [5, 5, 6, 6, 7, 7, 7], [5, 5, 6, 7, 7, 7, 7]
]

# ==================== 【自動中文字型載入器】 ====================
@st.cache_resource
def ensure_chinese_font():
    candidates = [
        "font.ttf",
        "/tmp/chinese_font.otf",
        "/tmp/chinese_font.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msjh.ttc",
        "msjh.ttc"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    if os.path.exists("/usr/share/fonts"):
        for root, _, files in os.walk("/usr/share/fonts"):
            for f in files:
                if any(k in f.lower() for k in ["wqy", "noto", "cjk", "zenhei", "microhei", "droid"]):
                    return os.path.join(root, f)

    urls = [
        ("https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf", "/tmp/chinese_font.otf"),
        ("https://raw.githubusercontent.com/StellarCN/scp_zh/master/fonts/SimHei.ttf", "/tmp/chinese_font.ttf")
    ]
    for url, target_path in urls:
        try:
            urllib.request.urlretrieve(url, target_path)
            if os.path.exists(target_path) and os.path.getsize(target_path) > 100000:
                return target_path
        except Exception:
            continue
    return None

FONT_FILE_PATH = ensure_chinese_font()

def put_chinese_text(img, text, position, color, size=18):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = None
    if FONT_FILE_PATH and os.path.exists(FONT_FILE_PATH):
        try:
            font = ImageFont.truetype(FONT_FILE_PATH, size)
        except Exception:
            font = None
    if font is None:
        font = ImageFont.load_default()
    
    x, y = position
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, 1), (1, 0)]:
        draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(color[2], color[1], color[0]), font=font)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==================== 【繞過 /media/ 的瀏覽器直出下載模組】 ====================
def render_blob_download_button(zip_bytes, filename, label_text):
    b64_zip = base64.b64encode(zip_bytes).decode("utf-8")
    html_code = f"""
    <div style="margin: 10px 0;">
        <button id="dl_btn" style="
            width: 100%;
            background-color: #FF4B4B;
            color: white;
            padding: 14px 20px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        ">{label_text}</button>
    </div>
    <script>
    document.getElementById("dl_btn").addEventListener("click", function() {{
        const b64Data = "{b64_zip}";
        const byteCharacters = atob(b64Data);
        const byteArrays = [];
        const sliceSize = 1024;
        for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {{
            const slice = byteCharacters.slice(offset, offset + sliceSize);
            const byteNumbers = new Array(slice.length);
            for (let i = 0; i < slice.length; i++) {{
                byteNumbers[i] = slice.charCodeAt(i);
            }}
            const byteArray = new Uint8Array(byteNumbers);
            byteArrays.push(byteArray);
        }}
        const blob = new Blob(byteArrays, {{type: "application/zip"}});
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.style.display = "none";
        a.href = url;
        a.download = "{filename}";
        document.body.appendChild(a);
        a.click();
        setTimeout(function() {{
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }}, 2000);
    }});
    </script>
    """
    components.html(html_code, height=70)

# ==================== 【幾何與人因工程計分邏輯】 ====================
def calculate_2d_angle(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-5 or norm_bc < 1e-5:
        return float("nan")
    cosine = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def calculate_3d_angle(a, b, c):
    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-5 or norm_bc < 1e-5:
        return float("nan")
    cosine = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def enhance_image_for_pose(bgr_img):
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ==================== 【V6：Hand Landmarker 實驗；不混用 Pose/Hand world 原點】 ====================
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_MODEL_FILENAME = "hand_landmarker.task"
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]


def ensure_hand_model():
    """優先使用專案 models/；否則將官方模型下載到可寫的 /tmp，絕不寫入 site-packages。"""
    root = os.path.dirname(os.path.abspath(__file__))
    bundled = os.path.join(root, "models", HAND_MODEL_FILENAME)
    if os.path.isfile(bundled) and os.path.getsize(bundled) > 1_000_000:
        return bundled
    fallback = os.path.join(tempfile.gettempdir(), "rula_hand_assets", HAND_MODEL_FILENAME)
    if os.path.isfile(fallback) and os.path.getsize(fallback) > 1_000_000:
        return fallback
    os.makedirs(os.path.dirname(fallback), exist_ok=True)
    staged = fallback + ".partial"
    try:
        with urllib.request.urlopen(HAND_MODEL_URL, timeout=35) as response, open(staged, "wb") as out:
            total = 0
            while True:
                chunk = response.read(128 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > 35_000_000:
                    raise ValueError("Hand 模型下載大小異常")
                out.write(chunk)
        if os.path.getsize(staged) < 1_000_000:
            raise ValueError("Hand 模型下載不完整")
        os.replace(staged, fallback)
    finally:
        if os.path.isfile(staged):
            try:
                os.remove(staged)
            except OSError:
                pass
    return fallback


def make_hand_landmarker(model_path):
    """使用 IMAGE 模式：每個選取影格獨立推論，適合 v5 的稀疏等距取樣。"""
    task_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(task_options)


def _landmark_xy(landmarks, width, height):
    return np.asarray([[float(x.x) * width, float(x.y) * height] for x in landmarks], dtype=float)


def extract_hand_comparison(hand_results, shape, pose_wrist, opposite_wrist, pose_elbow):
    """Same-frame image-space comparison only; never subtract Pose World and Hand World.

    Matched indicates hand-to-wrist association, NOT anatomical accuracy.
    Hand World values are local palm descriptors only.
    """
    empty = {"matched": False, "reason": "未偵測到可匹配的目標手", "angle_2d": float("nan"),
             "distance_ratio": float("nan"), "other_hand_margin_ratio": float("nan"),
             "palm_forearm_ratio": float("nan"), "hand_world_palm_length": float("nan"),
             "hand_world_palm_spread_deg": float("nan"), "handedness": "", "hand_points": None}
    if hand_results is None or not getattr(hand_results, "hand_landmarks", None):
        return empty
    height, width = shape[:2]
    arm_len = float(np.linalg.norm(pose_wrist - pose_elbow))
    if not np.isfinite(arm_len) or arm_len < 15.0:
        return {**empty, "reason": "Pose 前臂過短或關節定位不可靠"}
    options = []
    for i, hand in enumerate(hand_results.hand_landmarks):
        if len(hand) < 21:
            continue
        xy = _landmark_xy(hand, width, height)
        if not np.all(np.isfinite(xy)):
            continue
        target_dist = float(np.linalg.norm(xy[0] - pose_wrist))
        other_dist = float(np.linalg.norm(xy[0] - opposite_wrist)) if opposite_wrist is not None else float("inf")
        options.append((target_dist / arm_len, other_dist, target_dist, i, xy))
    if not options:
        return empty
    ratio, other_dist, target_dist, i, xy = min(options, key=lambda x: x[0])
    margin_ratio = (other_dist - target_dist) / arm_len if np.isfinite(other_dist) else float("nan")
    # One must not silently substitute the opposite hand. A small positive margin
    # may be geometrically ambiguous and is rejected at scoring-quality stage.
    if ratio > 0.42 or (other_dist + 8 < target_dist):
        return {**empty, "distance_ratio": round(ratio, 3),
                "other_hand_margin_ratio": round(margin_ratio, 3) if np.isfinite(margin_ratio) else np.nan,
                "reason": "Hand 腕點未對應目標 Pose 手腕"}
    palm_center = xy[[5, 9, 13, 17]].mean(axis=0)
    palm_len = float(np.linalg.norm(palm_center - xy[0]))
    palm_ratio = palm_len / arm_len
    if palm_len < 8:
        return {**empty, "distance_ratio": round(ratio, 3), "palm_forearm_ratio": round(palm_ratio, 3),
                "reason": "掌長像素不足"}
    angle = float(abs(180.0 - calculate_2d_angle(pose_elbow, xy[0], palm_center)))
    if not np.isfinite(angle):
        return {**empty, "distance_ratio": round(ratio, 3), "reason": "手部角度計算無效"}
    world_length, world_spread = float("nan"), float("nan")
    world_hands = getattr(hand_results, "hand_world_landmarks", [])
    if i < len(world_hands) and len(world_hands[i]) >= 21:
        w3 = np.asarray([[float(z.x), float(z.y), float(z.z)] for z in world_hands[i]])
        if np.all(np.isfinite(w3)):
            world_center = w3[[5, 9, 13, 17]].mean(axis=0)
            world_length = float(np.linalg.norm(world_center - w3[0]))
            world_spread = calculate_3d_angle(w3[5], w3[0], w3[17])
    handed = getattr(hand_results, "handedness", [])
    label = ""
    if i < len(handed) and handed[i]:
        label = str(getattr(handed[i][0], "category_name", ""))
    return {"matched": True, "reason": "", "angle_2d": angle,
            "distance_ratio": round(ratio, 3),
            "other_hand_margin_ratio": round(margin_ratio, 3) if np.isfinite(margin_ratio) else float("nan"),
            "palm_forearm_ratio": round(palm_ratio, 3),
            "hand_world_palm_length": world_length,
            "hand_world_palm_spread_deg": world_spread,
            "handedness": label, "hand_points": xy.tolist()}


def evaluate_hand_geometry(hand, pose_2d_angle, pose_wrist_visibility, pose_elbow_visibility):
    """Conservative engineering plausibility check, NOT probability or validated accuracy."""
    problems = []
    if not hand.get("matched"):
        return False, [hand.get("reason", "沒有目標手")]
    angle = float(hand.get("angle_2d", np.nan))
    ratio = float(hand.get("distance_ratio", np.nan))
    palm_ratio = float(hand.get("palm_forearm_ratio", np.nan))
    margin = float(hand.get("other_hand_margin_ratio", np.nan))
    if not np.isfinite(angle) or angle > 65:
        problems.append("Hand 2D 角度極端或無效")
    if not np.isfinite(ratio) or ratio > 0.22:
        problems.append("兩模型手腕位置距離偏大")
    if not np.isfinite(palm_ratio) or not 0.12 <= palm_ratio <= 0.85:
        problems.append("手掌與前臂尺度比例不合理")
    if np.isfinite(margin) and margin < 0.12:
        problems.append("左右手空間匹配不明確")
    if pose_wrist_visibility < 0.50 or pose_elbow_visibility < 0.50:
        problems.append("Pose 手肘／腕點可見度偏低")
    if not np.isfinite(pose_2d_angle) or abs(angle - pose_2d_angle) > 20:
        problems.append("Hand 2D 與 Pose 2D 差異較大")
    return len(problems) == 0, problems


def make_pose_keypoint_snapshot(landmarks, width, height, indices):
    """Image pixel coordinates + visibility for later independent diagnosis."""
    return {str(i): {"x_px": round(float(landmarks[i].x * width), 2),
                     "y_px": round(float(landmarks[i].y * height), 2),
                     "visibility": round(float(getattr(landmarks[i], "visibility", 0.0)), 4)}
            for i in indices}


def draw_hand_overlay(pose_bgr, hand_points, hand_angle=None):
    """在相同 Pose 圖像上畫 21 點；不修改原圖。"""
    out = pose_bgr.copy()
    if hand_points is None:
        return put_chinese_text(out, "Hand: 未匹配目標手", (15, 105), (0, 140, 255), 18)
    points = [(int(round(p[0])), int(round(p[1]))) for p in hand_points]
    for a, b in HAND_CONNECTIONS:
        cv2.line(out, points[a], points[b], (255, 180, 0), 2, cv2.LINE_AA)
    for n, point in enumerate(points):
        cv2.circle(out, point, 4 if n not in (0, 5, 9, 13, 17) else 6,
                   (0, 0, 255) if n == 0 else (255, 255, 0), -1, cv2.LINE_AA)
    label = f"Hand 2D: {hand_angle:.1f} deg" if hand_angle is not None and np.isfinite(hand_angle) else "Hand: n/a"
    return put_chinese_text(out, label, (15, 105), (0, 255, 255), 18)


def _encode_preview(bgr):
    if bgr is None:
        return None
    oh, ow = bgr.shape[:2]
    factor = min(720.0 / max(oh, ow), 1.0)
    if factor < 1:
        bgr = cv2.resize(bgr, (max(1, int(ow * factor)), max(1, int(oh * factor))))
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return buf.tobytes() if ok else None


def _save_hand_photo(side, sample_id, sec, pose_jpeg, overlay_jpeg, zip_hand_images):
    if pose_jpeg is None or overlay_jpeg is None:
        return
    st.session_state.hand_review_images[(side, int(sample_id))] = (
        base64.b64encode(pose_jpeg).decode("ascii"),
        base64.b64encode(overlay_jpeg).decode("ascii"),
    )
    tag = "Right" if side == "右手側" else "Left"
    zip_hand_images.append((f"Hand_Comparison/{tag}/Sample_{sample_id:03d}_{sec}s_Pose.jpg", pose_jpeg))
    zip_hand_images.append((f"Hand_Comparison/{tag}/Sample_{sample_id:03d}_{sec}s_PoseHand.jpg", overlay_jpeg))

ANGLE_DEFINITIONS = ("RULA v7 角度說明\n"
    "上臂: 評估側肩點到肘點相對投影軀幹方向的 2D 角度。\n"
    "前臂: 180° - 肩(評估側)／肘／腕在影像上的夾角。\n"
    "軀幹: 雙肩中心與雙髖中心連線相對影像垂直向上的投影角度，可能超過 90°。\n"
    "頸部: 頭部代理點相對肩髖軸的 2D 角度，不是臨床頸椎解剖角。\n"
    "Pose2D: Pose 肘、腕、食指／小指掌部代理點的 2D 角度。\n"
    "Pose3D: Pose 模型世界座標的 3D 角度，可能受模型深度推論影響。\n"
    "Pose混合: Pose2D*0.6+Pose3D*0.4（原版演算法，保留供比較）。\n"
    "Hand2D: 同張影像 Pose 肘 + Hand 腕／掌指方向的投影角度。\n"
    "HandWorld: 只用於手掌局部量測，並未與 Pose World 做 3D 座標融合。\n"
    "品質門檻僅為工程篩檢，不能視為實際角度準確率。\n"
    "RULA 查表矩陣保留既有版本，尚待標準工作表逐格獨立驗證。\n")


def safe_json_value(value):
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def clean_json(obj):
    # Recursively eliminate NaN; JSONL conforms to strict JSON, not NaN extensions.
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj

# ==================== 【姿態偵測重試與關鍵點品質檢查】 ====================
# Visibility 是模型對關鍵點可見性的估計，不是臨床級準確率。
POSE_MIN_VISIBILITY = 0.25
POSE_RETRY_MEAN_VISIBILITY = 0.58


def _visibility_values(results, indices):
    if not results or not getattr(results, "pose_landmarks", None):
        return []
    landmarks = results.pose_landmarks.landmark
    values = []
    for idx in indices:
        if idx >= len(landmarks):
            values.append(0.0)
            continue
        value = float(getattr(landmarks[idx], "visibility", 0.0) or 0.0)
        values.append(value if np.isfinite(value) else 0.0)
    return values


def detect_pose_with_fallback(bgr_img, pose_model, required_indices=None):
    """先用原圖；品質不足時再試 CLAHE 與放大圖，回傳品質最佳的結果。"""
    if bgr_img is None or bgr_img.size == 0:
        return None, 0.0, "empty_frame"

    if required_indices is None:
        required_indices = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    h, w = bgr_img.shape[:2]
    attempts = [(bgr_img, "original")]
    if h >= 2 and w >= 2:
        try:
            attempts.append((enhance_image_for_pose(bgr_img), "clahe"))
        except cv2.error:
            pass
        # 小人像／遠距離影片常因像素太少而失準；只在必要時才跑放大版本。
        up_scale = min(2.0, 1600.0 / max(h, w))
        up_w, up_h = max(1, int(w * up_scale)), max(1, int(h * up_scale))
        attempts.append((cv2.resize(bgr_img, (up_w, up_h), interpolation=cv2.INTER_CUBIC), "upscale"))

    best_results, best_score, best_method = None, -1.0, "none"
    for attempt_img, method in attempts:
        try:
            rgb = cv2.cvtColor(attempt_img, cv2.COLOR_BGR2RGB)
            results = pose_model.process(rgb)
        except Exception:
            logging.exception("MediaPipe pose detection attempt failed (%s)", method)
            continue

        if not results or not getattr(results, "pose_landmarks", None):
            continue
        values = _visibility_values(results, required_indices)
        if not values:
            continue
        mean_vis = float(np.mean(values))
        min_vis = float(np.min(values))
        # 同時考量平均與低可見度點，避免一兩個很差的關鍵點被平均值掩蓋。
        quality = 0.65 * mean_vis + 0.35 * min_vis
        if quality > best_score:
            best_results, best_score, best_method = results, quality, method

        if mean_vis >= POSE_RETRY_MEAN_VISIBILITY and min_vis >= POSE_MIN_VISIBILITY:
            break

    if best_results is None:
        return None, 0.0, best_method
    return best_results, max(0.0, best_score), best_method


def landmarks_meet_visibility(results, required_indices, threshold=POSE_MIN_VISIBILITY):
    values = _visibility_values(results, required_indices)
    return bool(values) and all(v >= threshold for v in values)

def get_upper_arm_score(angle, is_abducted=False, is_raised=False, is_supported=False, is_extended=False):
    """RULA upper-arm base score + modifiers; extension direction needs visual/manual confirmation."""
    if is_extended and angle > 20:
        base = 2  # >20° extension: score 2 (do not treat it as forward flexion)
    elif angle <= 20:
        base = 1
    elif angle <= 45:
        base = 2
    elif angle <= 90:
        base = 3
    else:
        base = 4
    if is_abducted:
        base += 1
    if is_raised:
        base += 1
    if is_supported:
        base -= 1
    return min(max(base, 1), 6)


def get_lower_arm_score(angle, is_across_midline=False, is_out_to_side=False):
    base = 1 if 60 <= angle <= 100 else 2
    # RULA adds one point if working across the midline OR out to the side.
    if is_across_midline or is_out_to_side:
        base += 1
    return min(max(base, 1), 3)


def get_wrist_score(angle, is_deviated=False):
    # RULA: neutral wrist = 1; non-neutral flexion/extension up to 15 degrees = 2.
    if angle <= 2.0:  # measurement tolerance; not an officially validated threshold
        base = 1
    elif angle <= 15:
        base = 2
    else:
        base = 3
    if is_deviated:
        base += 1
    return min(max(base, 1), 4)


def get_neck_score(angle, is_twisted=False, is_side_bending=False, is_extended=False):
    if is_extended:
        base = 4  # Neck extension is a distinct RULA category; unsigned 2D angles cannot infer it reliably.
    elif angle <= 10:
        base = 1
    elif angle <= 20:
        base = 2
    else:
        base = 3
    if is_twisted:
        base += 1
    if is_side_bending:
        base += 1
    return min(max(base, 1), 6)


def get_trunk_score(angle, is_twisted=False, is_side_bending=False, is_extended=False):
    if is_extended:
        base = 2
    elif angle <= 5:
        base = 1
    elif angle <= 20:
        base = 2
    elif angle <= 60:
        base = 3
    else:
        base = 4
    if is_twisted:
        base += 1
    if is_side_bending:
        base += 1
    return min(max(base, 1), 6)

def compute_rula_full(ua, la, w, wt, nk, tk, lg, m_val, f_val):
    ua = min(max(ua, 1), 6)
    la = min(max(la, 1), 3)
    w  = min(max(w, 1), 4)
    wt = min(max(wt, 1), 2)
    nk = min(max(nk, 1), 6)
    tk = min(max(tk, 1), 6)
    lg = min(max(lg, 1), 2)
    score_a = TABLE_A[ua][la][w][wt]
    score_b = TABLE_B[nk][tk][lg]
    score_c = score_a + m_val + f_val
    score_d = score_b + m_val + f_val
    grand_score = TABLE_C[min(score_c, 8) - 1][min(score_d, 7) - 1]
    al_num = 1 if grand_score <= 2 else (2 if grand_score <= 4 else (3 if grand_score <= 6 else 4))
    return score_a, score_b, score_c, score_d, grand_score, f"AL{al_num}"

# ==================== 【全 33 點生物力學渲染器 (全身雙側完整繪製 + 目標側高亮標註)】 ====================
def draw_full_biomechanical_skeleton(
    img, pts_dict, ang_ua, ang_la, ang_wr, ang_tk, ang_nk, wt_score, target_side_label=""
):
    """
    完整繪製全身 33 個解剖點：
    1. 臉部五官 11 點網格 (鼻、雙眼內中外、雙耳、嘴角)
    2. 全身左右雙側骨架 (主評估側以高對比力學色塊與角度標籤呈現，對側亦完整繪製手掌三角、拇指與足底三角)
    3. 手部精細結構 (手腕、拇指軸線、食指-小指手掌平面三角形)
    4. 足底支撐結構 (腳踝-腳跟-腳尖支撐三角形)
    """
    out = img.copy()
    overlay = img.copy()

    def to_pt(arr):
        return (int(round(arr[0])), int(round(arr[1])))

    # 取出對側與主側座標
    o_sh, o_el, o_wr = to_pt(pts_dict["o_sh"]), to_pt(pts_dict["o_el"]), to_pt(pts_dict["o_wr"])
    o_th, o_ix, o_pk = to_pt(pts_dict["o_th"]), to_pt(pts_dict["o_ix"]), to_pt(pts_dict["o_pk"])
    o_hp, o_kn, o_an = to_pt(pts_dict["o_hp"]), to_pt(pts_dict["o_kn"]), to_pt(pts_dict["o_an"])
    o_hl, o_ft = to_pt(pts_dict["o_hl"]), to_pt(pts_dict["o_ft"])

    p_ear = to_pt(pts_dict["ear"])
    p_sh  = to_pt(pts_dict["sh"])
    p_el  = to_pt(pts_dict["el"])
    p_wr  = to_pt(pts_dict["wr"])
    p_th  = to_pt(pts_dict["th"])
    p_ix  = to_pt(pts_dict["ix"])
    p_pk  = to_pt(pts_dict["pk"])
    p_hp  = to_pt(pts_dict["hp"])
    p_kn  = to_pt(pts_dict["kn"])
    p_an  = to_pt(pts_dict["an"])
    p_hl  = to_pt(pts_dict["hl"])
    p_ft  = to_pt(pts_dict["ft"])

    # ---- 第一層：半透明力學平面填充 (雙側手掌三角、雙側足底三角、軀幹網格) ----
    o_palm_poly = np.array([o_wr, o_ix, o_pk], dtype=np.int32)
    o_foot_poly = np.array([o_an, o_hl, o_ft], dtype=np.int32)
    palm_poly = np.array([p_wr, p_ix, p_pk], dtype=np.int32)
    foot_poly = np.array([p_an, p_hl, p_ft], dtype=np.int32)
    torso_poly = np.array([p_sh, o_sh, o_hp, p_hp], dtype=np.int32)

    cv2.fillPoly(overlay, [torso_poly], (0, 180, 60))
    cv2.fillPoly(overlay, [o_palm_poly], (255, 200, 0))
    cv2.fillPoly(overlay, [o_foot_poly], (255, 180, 0))
    cv2.fillPoly(overlay, [palm_poly], (0, 220, 255))
    cv2.fillPoly(overlay, [foot_poly], (0, 200, 255))
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

    # ---- 第二層：對側完整骨架與關節點 (青藍色系，確保全身雙側皆清晰呈現) ----
    cv2.line(out, p_sh, o_sh, (0, 210, 120), 2, cv2.LINE_AA)
    cv2.line(out, p_hp, o_hp, (0, 210, 120), 2, cv2.LINE_AA)
    cv2.line(out, o_sh, o_hp, (180, 200, 80), 2, cv2.LINE_AA)

    cv2.line(out, o_sh, o_el, (255, 190, 0), 3, cv2.LINE_AA)
    cv2.line(out, o_el, o_wr, (255, 210, 50), 3, cv2.LINE_AA)
    cv2.line(out, o_wr, o_th, (255, 100, 220), 2, cv2.LINE_AA)
    cv2.polylines(out, [o_palm_poly], True, (255, 240, 80), 2, cv2.LINE_AA)

    cv2.line(out, o_hp, o_kn, (255, 170, 50), 3, cv2.LINE_AA)
    cv2.line(out, o_kn, o_an, (255, 190, 80), 3, cv2.LINE_AA)
    cv2.polylines(out, [o_foot_poly], True, (255, 220, 100), 2, cv2.LINE_AA)

    for p_opp in [o_sh, o_el, o_wr, o_th, o_ix, o_pk, o_hp, o_kn, o_an, o_hl, o_ft]:
        cv2.circle(out, p_opp, 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, p_opp, 3, (255, 140, 0), -1, cv2.LINE_AA)

    # ---- 第三層：頭部五官 11 點解剖網格 (鼻、雙眼內中外、雙耳、嘴角) ----
    f_pts = [to_pt(pts_dict[f"f_{i}"]) for i in range(11)]
    # 右眼軌跡 (0->4->5->6->8) 與 左眼軌跡 (0->1->2->3->7)
    for a_i, b_i in [(0,4), (4,5), (5,6), (6,8), (0,1), (1,2), (2,3), (3,7), (2,5), (9,10)]:
        cv2.line(out, f_pts[a_i], f_pts[b_i], (0, 255, 255), 2, cv2.LINE_AA)
    mouth_mid = ((f_pts[9][0] + f_pts[10][0]) // 2, (f_pts[9][1] + f_pts[10][1]) // 2)
    cv2.line(out, f_pts[0], mouth_mid, (0, 255, 255), 2, cv2.LINE_AA)

    for fp in f_pts:
        cv2.circle(out, fp, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, fp, 3, (0, 180, 255), -1, cv2.LINE_AA)

    # ---- 第四層：主評估側核心人因力學鏈 (高亮顯示) ----
    cv2.line(out, p_hp, (p_hp[0], max(0, p_sh[1] - 70)), (200, 200, 200), 2, cv2.LINE_AA)

    # 下肢與足底支撐三角形
    cv2.line(out, p_hp, p_kn, (0, 165, 255), 4, cv2.LINE_AA)
    cv2.line(out, p_kn, p_an, (0, 190, 255), 4, cv2.LINE_AA)
    cv2.polylines(out, [foot_poly], True, (0, 235, 255), 3, cv2.LINE_AA)

    # 軀幹與頭頸軸線
    cv2.line(out, p_hp, p_sh, (0, 240, 0), 5, cv2.LINE_AA)
    cv2.line(out, p_sh, p_ear, (0, 255, 255), 4, cv2.LINE_AA)
    cv2.line(out, p_ear, f_pts[0], (0, 215, 255), 3, cv2.LINE_AA)

    # 上肢與手掌力學三角 + 拇指扭轉軸
    cv2.line(out, p_sh, p_el, (0, 60, 255), 5, cv2.LINE_AA)
    cv2.line(out, p_el, p_wr, (0, 140, 255), 5, cv2.LINE_AA)
    cv2.polylines(out, [palm_poly], True, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.line(out, p_wr, p_th, (255, 50, 255), 3, cv2.LINE_AA)

    for pt in [p_ear, p_sh, p_el, p_wr, p_hp, p_kn, p_an]:
        cv2.circle(out, pt, 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, pt, 5, (0, 0, 255), -1, cv2.LINE_AA)

    for pt in [p_th, p_ix, p_pk, p_hl, p_ft]:
        cv2.circle(out, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, pt, 4, (255, 0, 200), -1, cv2.LINE_AA)

    # 標註各關節即時角度與手腕扭轉狀態
    twist_str = "接近旋轉極限(2分)" if wt_score == 2 else "中間範圍(1分)"
    out = put_chinese_text(out, f"頸部:{ang_nk:.0f}°", (p_ear[0] + 12, max(10, p_ear[1] - 12)), (255, 255, 120), 14)
    out = put_chinese_text(out, f"上臂:{ang_ua:.0f}°", (p_sh[0] + 12, p_sh[1] - 8), (0, 255, 255), 14)
    out = put_chinese_text(out, f"前臂:{ang_la:.0f}°", (p_el[0] + 12, p_el[1] + 2), (0, 200, 255), 14)
    out = put_chinese_text(out, f"手腕:{ang_wr:.0f}°|{twist_str}", (p_wr[0] + 12, p_wr[1] + 5), (255, 160, 255), 14)
    out = put_chinese_text(out, f"軀幹:{ang_tk:.0f}°", (p_hp[0] + 12, p_hp[1] - 15), (100, 255, 100), 14)

    if target_side_label:
        h_i, _ = out.shape[:2]
        out = put_chinese_text(out, f"🎥 獨立擷取視角: {target_side_label}專用鏡頭", (12, h_i - 28), (255, 255, 255), 15)

    return out

# ==================== 【正面鏡頭：冠狀面/水平面補正分析 (支援指定單側)】 ====================
def analyze_front_camera(frame, pose_model, target_side):
    """正面影像補正。若核心點品質不足，回報失敗而不是默認所有補正皆為否。"""
    if frame is None or frame.size == 0:
        return False, {}

    orig_h, orig_w = frame.shape[:2]
    scale = min(900 / max(orig_h, orig_w), 1.0)
    frame_resized = cv2.resize(frame, (max(1, int(orig_w * scale)), max(1, int(orig_h * scale))))

    sh_idx = 12 if target_side == "右手側" else 11
    el_idx = 14 if target_side == "右手側" else 13
    wr_idx = 16 if target_side == "右手側" else 15
    required = [11, 12, 13, 14, 15, 16, 23, 24, 7, 8, sh_idx, el_idx, wr_idx]
    results, _, _ = detect_pose_with_fallback(frame_resized, pose_model, required)
    if not results or not getattr(results, "pose_landmarks", None):
        return False, {}

    lm = results.pose_landmarks.landmark
    if not landmarks_meet_visibility(results, required, threshold=0.25):
        return False, {}

    h, w = frame_resized.shape[:2]
    def xy(i):
        return np.array([lm[i].x * w, lm[i].y * h], dtype=float)

    sh_left, sh_right = xy(11), xy(12)
    hp_left, hp_right = xy(23), xy(24)
    # 人體相對座標：肩線決定水平軸，雙肩/雙髖中點決定軀幹軸。
    # 避免畫面傾斜時直接把影像 X 位移誤認成「跨中線」。
    shoulder_mid = (sh_left + sh_right) * 0.5
    hip_mid = (hp_left + hp_right) * 0.5
    shoulder_vec = sh_right - sh_left
    shoulder_width = float(np.linalg.norm(shoulder_vec))
    torso_vec = hip_mid - shoulder_mid
    torso_length = float(np.linalg.norm(torso_vec))
    if shoulder_width < 12.0 or torso_length < 16.0:
        return False, {}
    horiz = shoulder_vec / shoulder_width
    torso_axis = torso_vec / torso_length
    # 正面旋轉程度以世界座標的肩線深度相對肩寬估計，僅用來降低品質評級。
    # 此數值不是相機標定或準確率。
    world = getattr(results, "pose_world_landmarks", None)
    rotation_warning = False
    if world is not None:
        wlm = world.landmark
        shoulder_depth_delta = abs(float(wlm[12].z - wlm[11].z))
        shoulder_world_width = float(np.linalg.norm([wlm[12].x-wlm[11].x, wlm[12].y-wlm[11].y, wlm[12].z-wlm[11].z]))
        rotation_warning = shoulder_depth_delta / max(shoulder_world_width, 1e-6) > 0.55
    # 目標側肩膀位於軸向哪一邊，避免鏡像影像造成左右符號反轉。
    target_sh = xy(sh_idx)
    elbow = xy(el_idx)
    wrist = xy(wr_idx)
    side_sign = float(np.sign(np.dot(target_sh - shoulder_mid, horiz)))
    if side_sign == 0:
        return False, {}
    elbow_out = side_sign * float(np.dot(elbow - target_sh, horiz)) / shoulder_width
    wrist_midline = side_sign * float(np.dot(wrist - shoulder_mid, horiz)) / shoulder_width
    elbow_midline = side_sign * float(np.dot(elbow - shoulder_mid, horiz)) / shoulder_width
    # 只把具有明確幾何證據的情況視為 True，設中性帶避免邊界抖動。
    is_abducted = bool(elbow_out > 0.33)
    is_across_midline = bool(wrist_midline < -0.12)
    is_out_to_side = bool(not is_across_midline and elbow_midline > 0.65 and wrist_midline > 0.80)
    # 軀幹左右傾斜用骨盆中點到肩中點相對鉛垂線，避免單邊聳肩造成假陽性。
    trunk_lateral_angle = float(np.degrees(np.arctan2(abs(shoulder_mid[0] - hip_mid[0]), max(abs(shoulder_mid[1] - hip_mid[1]), 1e-6))))
    # 耳中點相對肩中點的側移，與軀幹偏斜量相減，用於估測頸部相對側彎。
    ear_left, ear_right = xy(7), xy(8)
    ear_mid = (ear_left + ear_right) * 0.5
    ear_axis = ear_mid - shoulder_mid
    neck_relative_angle = float(np.degrees(np.arctan2(abs(float((-torso_axis)[0]*ear_axis[1]-(-torso_axis)[1]*ear_axis[0])), max(abs(np.dot(-torso_axis, ear_axis)), 1e-6)))) if np.linalg.norm(ear_axis) >= 5 else 0.0
    # 以上都是影像代理規則；不能單憑這些數據宣稱臨床準確。
    mods = {
        "is_abducted": is_abducted,
        "is_across_midline": is_across_midline,
        "is_out_to_side": is_out_to_side,
        "is_side_bending": bool(trunk_lateral_angle > 12.0),
        "is_neck_side_bending": bool(neck_relative_angle > 14.0),
        "front_keypoint_visibility": float(np.mean(_visibility_values(results, required)) * 100.0),
        "front_rotation_warning": rotation_warning,
        "cross_midline_normalized": round(wrist_midline, 3),
        "trunk_lateral_angle_proxy": round(trunk_lateral_angle, 1),
        "neck_lateral_angle_proxy": round(neck_relative_angle, 1),
    }
    return True, mods

# ==================== 【側視角鏡頭：單側獨立擷取、分析與 RULA 計算】 ====================
def analyze_side_camera(frame, pose_model, target_side, front_mods, m_val, f_val, manual_mods=None, hand_task=None, hand_scoring_policy="compare"):
    if frame is None or frame.size == 0 or not isinstance(front_mods, dict):
        return False, None, frame, 0.0
    manual_mods = manual_mods or {}
    auto_selection = manual_mods.get("auto_selection", {})
    manual_front = manual_mods.get("manual_front", {})
    front_mods = dict(front_mods)
    for flag in ("is_abducted", "is_across_midline", "is_out_to_side", "is_side_bending", "is_neck_side_bending"):
        if not auto_selection.get(flag, True):
            front_mods[flag] = bool(manual_front.get(flag, False))
    # 對於鏡頭轉身造成的視角不佳，保留警告，不把模型 visibility 當準確率。
    orig_h, orig_w = frame.shape[:2]
    scale = min(1000 / max(orig_h, orig_w), 1.0)
    work_w, work_h = max(1, int(orig_w * scale)), max(1, int(orig_h * scale))
    frame_work = cv2.resize(frame, (work_w, work_h))

    if target_side == "右手側":
        t_ear, t_sh, t_el, t_wr, t_pk, t_ix, t_th = 8, 12, 14, 16, 18, 20, 22
        t_hp, t_kn, t_an, t_hl, t_ft = 24, 26, 28, 30, 32
        o_ear, o_sh, o_el, o_wr, o_pk, o_ix, o_th = 7, 11, 13, 15, 17, 19, 21
        o_hp, o_kn, o_an, o_hl, o_ft = 23, 25, 27, 29, 31
    else:
        t_ear, t_sh, t_el, t_wr, t_pk, t_ix, t_th = 7, 11, 13, 15, 17, 19, 21
        t_hp, t_kn, t_an, t_hl, t_ft = 23, 25, 27, 29, 31
        o_ear, o_sh, o_el, o_wr, o_pk, o_ix, o_th = 8, 12, 14, 16, 18, 20, 22
        o_hp, o_kn, o_an, o_hl, o_ft = 24, 26, 28, 30, 32

    # 只允許評估指定側；不再因對側 visibility 較高就悄悄換手，避免左右手分數標錯。
    required = [t_sh, t_el, t_wr, t_ix, t_pk, t_hp, o_sh, o_hp]
    side_pose_start = time.perf_counter()
    results, _, _ = detect_pose_with_fallback(frame_work, pose_model, required)
    side_pose_ms = round((time.perf_counter() - side_pose_start) * 1000.0, 1)
    if not results or not getattr(results, "pose_landmarks", None) or not getattr(results, "pose_world_landmarks", None):
        return False, None, frame_work, 0.0

    n_lm = results.pose_landmarks.landmark
    w_lm = results.pose_world_landmarks.landmark
    if not landmarks_meet_visibility(results, required, threshold=0.25):
        return False, None, frame_work, float(np.mean(_visibility_values(results, required)) * 100.0)
    # 側面常會遮住遠側眼睛／耳朵，改用可見度較高的臉部點，而不是要求兩眼同時可見。
    if max(float(n_lm[t_ear].visibility), float(n_lm[o_ear].visibility)) < 0.25:
        return False, None, frame_work, float(np.mean(_visibility_values(results, required)) * 100.0)
    if max(float(n_lm[2].visibility), float(n_lm[5].visibility)) < 0.25:
        return False, None, frame_work, float(np.mean(_visibility_values(results, required)) * 100.0)

    try:
        def pt2d(idx):
            return np.array([n_lm[idx].x * work_w, n_lm[idx].y * work_h], dtype=float)

        def pt3d(idx):
            return np.array([w_lm[idx].x, w_lm[idx].y, w_lm[idx].z], dtype=float)

        key_vis = _visibility_values(results, required)
        conf_side = float(np.mean(key_vis) * 100.0) if key_vis else 0.0

        # 肩關節與前臂角度使用「評估側真實肩點」，避免雙肩加權造成關節頂點偏移。
        # 軀幹與頭頸則用肩／髖中心估算影像投影；不能當作解剖學脊椎角度。
        sh, hp = pt2d(t_sh), pt2d(t_hp)
        sh_center = (pt2d(t_sh) + pt2d(o_sh)) * 0.5
        hp_center = (pt2d(t_hp) + pt2d(o_hp)) * 0.5
        ear = pt2d(t_ear) if n_lm[t_ear].visibility >= n_lm[o_ear].visibility else pt2d(o_ear)
        eye_idx = 2 if n_lm[2].visibility >= n_lm[5].visibility else 5
        visible_eye = pt2d(eye_idx)

        # 目標側上肢與手掌點；不以對側手臂替換目標側。
        el, wr, th, ix, pk = pt2d(t_el), pt2d(t_wr), pt2d(t_th), pt2d(t_ix), pt2d(t_pk)
        b_el, b_wr, b_th, b_ix, b_pk = pt2d(o_el), pt2d(o_wr), pt2d(o_th), pt2d(o_ix), pt2d(o_pk)
        w3_el, w3_wr, w3_th, w3_ix, w3_pk = pt3d(t_el), pt3d(t_wr), pt3d(t_th), pt3d(t_ix), pt3d(t_pk)

        # 腿部點只用於畫圖，不再用雙腳影像高度差自動代替 RULA 的支撐／平衡判斷。
        kn, an, hl, ft = pt2d(t_kn), pt2d(t_an), pt2d(t_hl), pt2d(t_ft)
        b_kn, b_an, b_hl, b_ft = pt2d(o_kn), pt2d(o_an), pt2d(o_hl), pt2d(o_ft)

        torso_len = np.linalg.norm(hp - sh)

        # ==================== 【RULA 獨立關節角度解算】 ====================
        ang_lower_arm = abs(180.0 - calculate_2d_angle(sh, el, wr))
        
        trunk_vec = hp_center - sh_center
        ref_sh_down = sh + trunk_vec
        ang_upper_arm = calculate_2d_angle(ref_sh_down, sh, el)
        
        palm_mid_2d = (ix + pk) * 0.5
        palm_len_2d = np.linalg.norm(palm_mid_2d - wr)
        raw_wrist_2d = abs(180.0 - calculate_2d_angle(el, wr, palm_mid_2d)) if palm_len_2d > 6.0 else float("nan")
        palm_mid_3d = (w3_ix + w3_pk) * 0.5
        raw_wrist_3d = abs(180.0 - calculate_3d_angle(w3_el, w3_wr, palm_mid_3d))
        if not np.isfinite(raw_wrist_2d):
            ang_wrist = raw_wrist_3d
        else:
            ang_wrist = float(raw_wrist_2d * 0.6 + raw_wrist_3d * 0.4)
        # Preserve the raw estimated angle. Do not silently turn e.g. 95° into 60°.
        wrist_raw_angle = float(ang_wrist)
        wrist_angle_flag = bool(wrist_raw_angle > 60.0 or abs(raw_wrist_2d - raw_wrist_3d) > 25.0) if np.isfinite(raw_wrist_2d) and np.isfinite(raw_wrist_3d) else bool(wrist_raw_angle > 60.0)
        ang_wrist = wrist_raw_angle

        # V6：同影格使用 Hand 的手腕／MCP 點與 Pose 肘部計算 2D 代理角度。
        # Hand world 只做手掌局部幾何品質資訊，不拿它與 Pose world 相減。
        hand_ms = float("nan")
        hand_comp = {"matched": False, "reason": "Hand 模式未啟用或模型不可用", "angle_2d": float("nan"),
                     "distance_ratio": float("nan"), "other_hand_margin_ratio": float("nan"),
                     "palm_forearm_ratio": float("nan"), "hand_world_palm_length": float("nan"),
                     "hand_world_palm_spread_deg": float("nan"), "handedness": "", "hand_points": None}
        if hand_task is not None:
            start_hand = time.perf_counter()
            try:
                rgb = np.ascontiguousarray(cv2.cvtColor(frame_work, cv2.COLOR_BGR2RGB))
                mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                hand_results = hand_task.detect(mp_frame)
                hand_comp = extract_hand_comparison(
                    hand_results, frame_work.shape, wr,
                    b_wr if float(n_lm[o_wr].visibility) >= 0.40 else None, el
                )
            except Exception as exc:
                logging.exception("Hand Landmarker inference failed")
                hand_comp["reason"] = "Hand 模型執行失敗：" + type(exc).__name__
            finally:
                hand_ms = round((time.perf_counter() - start_hand) * 1000.0, 1)
        pose_wrist_angle = wrist_raw_angle
        hand_angle = float(hand_comp["angle_2d"])
        hand_diff = abs(hand_angle - pose_wrist_angle) if np.isfinite(hand_angle) else float("nan")
        pose_2d_hand_diff = abs(hand_angle - raw_wrist_2d) if np.isfinite(hand_angle) and np.isfinite(raw_wrist_2d) else float("nan")
        hand_quality_ok, hand_quality_problems = evaluate_hand_geometry(
            hand_comp, raw_wrist_2d, float(n_lm[t_wr].visibility), float(n_lm[t_el].visibility)
        )
        hand_comp["quality_notes"] = "；".join(hand_quality_problems) if hand_quality_problems else "幾何一致性通過（非準確率認證）"
        hand_used_for_score = bool(hand_scoring_policy == "gated" and hand_quality_ok)
        if hand_used_for_score:
            # 僅在使用者主動切換實驗模式、且兩個模型大致一致時才以 Hand 角度計分。
            ang_wrist = hand_angle

        # MediaPipe Pose 的拇指／手腕三維夾角不是前臂旋前旋後角，因此腕扭轉與偏移改由使用者目視確認。
        wt_score = int(manual_mods.get("wrist_twist_score", 1))
        is_wrist_deviated = bool(manual_mods.get("wrist_deviated", False))

        vertical_up = np.array([hp_center[0], hp_center[1] - 100.0])
        ang_trunk = calculate_2d_angle(vertical_up, hp_center, sh_center)

        head_center = ear * 0.6 + visible_eye * 0.4
        ang_neck = abs(180.0 - calculate_2d_angle(hp_center, sh_center, head_center))
        neck_v8 = v8_neck_candidate(n_lm, work_w, work_h)

        # RULA 腿足支撐與平衡需依實際作業情境確認，不再用兩腳高度差猜測。
        lg_score = int(manual_mods.get("leg_score", 2))
        if not np.isfinite(ang_lower_arm) or not np.isfinite(ang_upper_arm) or not np.isfinite(ang_trunk) or not np.isfinite(ang_neck) or not np.isfinite(ang_wrist):
            return False, None, frame_work, conf_side

        # 上臂向後伸展：側面鼻尖指向作為身體朝前的代理方向。
        # 側視不足／臉部遮擋時，回到人工輸入，而不是默認「沒有後伸」。
        auto_extension_used = False
        auto_extension_warning = ""
        is_extended = bool(manual_mods.get("upper_arm_extended", False))
        if auto_selection.get("upper_arm_extended", False):
            nose_visible = float(getattr(n_lm[0], "visibility", 0.0)) >= 0.45
            face_forward_delta = float(pt2d(0)[0] - sh[0]) if nose_visible else 0.0
            upper_arm_delta = float(el[0] - sh[0])
            arm_length = float(np.linalg.norm(el - sh))
            # 朝向清楚且上臂後移超過約 20° 才自動判定；其餘仍標註幾何限制。
            if nose_visible and abs(face_forward_delta) > max(12.0, 0.15 * torso_len) and arm_length > 15:
                backward_projection = -np.sign(face_forward_delta) * upper_arm_delta / arm_length
                is_extended = bool(backward_projection > np.sin(np.radians(20.0)) and ang_upper_arm > 20.0)
                auto_extension_used = True
            else:
                auto_extension_warning = "側面朝向不明，使用人工設定"

        # RULA 查表計算；不可由影像可靠辨識的條件繼續採用人工輸入。
        s_ua = get_upper_arm_score(
            ang_upper_arm,
            is_abducted=front_mods.get("is_abducted", False),
            is_raised=manual_mods.get("shoulder_raised", False),
            is_supported=manual_mods.get("arm_supported", False),
            is_extended=is_extended,
        )
        s_la = get_lower_arm_score(
            ang_lower_arm,
            is_across_midline=front_mods.get("is_across_midline", False),
            is_out_to_side=front_mods.get("is_out_to_side", False),
        )
        s_wr = get_wrist_score(ang_wrist, is_deviated=is_wrist_deviated)
        s_tk = get_trunk_score(
            ang_trunk,
            is_twisted=manual_mods.get("trunk_twisted", False),
            is_side_bending=front_mods.get("is_side_bending", False),
            is_extended=manual_mods.get("trunk_extended", False),
        )
        s_nk = get_neck_score(
            ang_neck,
            is_twisted=manual_mods.get("neck_twisted", False),
            is_side_bending=front_mods.get("is_neck_side_bending", False),
            is_extended=manual_mods.get("neck_extended", False),
        )
        
        sc_a, sc_b, sc_c, sc_d, grand, al = compute_rula_full(
            s_ua, s_la, s_wr, wt_score, s_nk, s_tk, lg_score, m_val, f_val
        )
        # 同一影格相同的其他關節分數，只替換手腕「屈曲/伸展」部位分數做情境對照。
        # 即使選擇安全比較模式，兩種情境仍可報表比較，原本 RULA 分數不被覆蓋。
        score_pose_ref = compute_rula_full(
            s_ua, s_la, get_wrist_score(pose_wrist_angle, is_deviated=is_wrist_deviated),
            wt_score, s_nk, s_tk, lg_score, m_val, f_val
        )
        score_hand_ref = (compute_rula_full(
            s_ua, s_la, get_wrist_score(hand_angle, is_deviated=is_wrist_deviated),
            wt_score, s_nk, s_tk, lg_score, m_val, f_val
        ) if np.isfinite(hand_angle) else None)
        
        # 打包全部 33 點座標供繪圖器渲染
        pts_dict = {
            "ear": ear, "sh": sh, "el": el, "wr": wr, "th": th, "ix": ix, "pk": pk,
            "hp": hp, "kn": kn, "an": an, "hl": hl, "ft": ft,
            "o_sh": pt2d(o_sh), "o_el": b_el, "o_wr": b_wr, "o_th": b_th, "o_ix": b_ix, "o_pk": b_pk,
            "o_hp": pt2d(o_hp), "o_kn": b_kn, "o_an": b_an, "o_hl": b_hl, "o_ft": b_ft
        }
        for f_i in range(11):
            pts_dict[f"f_{f_i}"] = pt2d(f_i)

        frame_annotated = draw_full_biomechanical_skeleton(
            frame_work, pts_dict,
            ang_upper_arm, ang_lower_arm, ang_wrist, ang_trunk, ang_neck, wt_score,
            target_side_label=target_side
        )
        
        side_prefix = "右手" if target_side == "右手側" else "左手"
        data = {
            "頸部角度": round(ang_neck, 1),
            "頸部_v7_原角度": round(ang_neck,1),
            "頸部_v8_候選角度": neck_v8['angle'],
            "頸部_v8_候選品質": neck_v8['quality'],
            "頸部_v8_關鍵點": neck_v8['landmarks'],
            "頸部_v8_診斷": neck_v8['note'],
            "軀幹角度": round(ang_trunk, 1),
            f"{side_prefix}上臂角度": round(ang_upper_arm, 1),
            f"{side_prefix}前臂角度": round(ang_lower_arm, 1),
            f"{side_prefix}手腕角度": round(ang_wrist, 1),
            f"{side_prefix}Pose舊版手腕角度": round(pose_wrist_angle, 1),
            f"{side_prefix}PoseHand實驗手腕角度": round(hand_angle, 1) if np.isfinite(hand_angle) else np.nan,
            f"{side_prefix}新舊手腕差異": round(hand_diff, 1) if np.isfinite(hand_diff) else np.nan,
            f"{side_prefix}Pose2D與Hand2D差異": round(pose_2d_hand_diff, 1) if np.isfinite(pose_2d_hand_diff) else np.nan,
            f"{side_prefix}Hand品質原因": hand_comp["quality_notes"],
            f"{side_prefix}Hand掌長前臂比": hand_comp["palm_forearm_ratio"],
            f"{side_prefix}Hand對側手匹配差距比": hand_comp["other_hand_margin_ratio"],
            f"{side_prefix}Hand匹配成功": bool(hand_comp["matched"]),
            f"{side_prefix}Pose基準GrandScore": score_pose_ref[4],
            f"{side_prefix}Hand情境GrandScore": score_hand_ref[4] if score_hand_ref else np.nan,
            f"{side_prefix}Hand情境與Pose總分差": score_hand_ref[4] - score_pose_ref[4] if score_hand_ref else np.nan,
            f"{side_prefix}Hand情境說明": "僅替換手腕屈伸代理值，其他部位與加分固定；不是新的標準評分方法",
            f"{side_prefix}Hand品質通過": hand_quality_ok,
            f"{side_prefix}Hand用於計分": hand_used_for_score,
            f"{side_prefix}Hand警示原因": hand_comp["reason"],
            f"{side_prefix}Hand與Pose腕點距離比例": hand_comp["distance_ratio"],
            f"{side_prefix}Hand左右手模型標籤": hand_comp["handedness"],
            f"{side_prefix}HandWorld掌心長度(模型單位)": round(hand_comp["hand_world_palm_length"], 4) if np.isfinite(hand_comp["hand_world_palm_length"]) else np.nan,
            f"{side_prefix}HandWorld掌指張角(度)": round(hand_comp["hand_world_palm_spread_deg"], 1) if np.isfinite(hand_comp["hand_world_palm_spread_deg"]) else np.nan,
            f"{side_prefix}側面Pose偵測耗時(ms)": side_pose_ms,
            f"{side_prefix}Pose腕點可見度": round(float(n_lm[t_wr].visibility), 3),
            f"{side_prefix}Pose肘點可見度": round(float(n_lm[t_el].visibility), 3),
            "角度定義註記": "肩肘腕／軀幹為單一鏡頭 2D 投影代理值，非經校準人體解剖角度",
            f"{side_prefix}Hand偵測耗時(ms)": hand_ms,
            f"{side_prefix}手腕計分演算法": "Hand實驗" if hand_used_for_score else "Pose v5",
            f"{side_prefix}手腕角度_2D": round(raw_wrist_2d, 1) if np.isfinite(raw_wrist_2d) else np.nan,
            f"{side_prefix}手腕角度_3D": round(raw_wrist_3d, 1) if np.isfinite(raw_wrist_3d) else np.nan,
            f"{side_prefix}手腕角度需複核": wrist_angle_flag,
            f"{side_prefix}手腕扭轉": "接近旋轉極限(2分)" if wt_score == 2 else "中間範圍(1分；人工確認)",
            f"{side_prefix}手腕偏移補正": "是 (+1分)" if is_wrist_deviated else "否（人工確認）",
            "足底支撐狀態": "未受支撐或不平衡(2分；人工確認)" if lg_score == 2 else "受支撐且平衡(1分；人工確認)",
            f"{side_prefix}上臂外展補正": "是 (+1分)" if front_mods.get("is_abducted") else "否",
            f"{side_prefix}肩部抬高補正": "是 (+1分)" if manual_mods.get("shoulder_raised") else "否（人工確認）",
            f"{side_prefix}上臂支撐／倚靠補正": "是 (-1分)" if manual_mods.get("arm_supported") else "否（人工確認）",
            f"{side_prefix}上臂向後伸展補正": "是（後伸）" if is_extended else "否",
            f"{side_prefix}橫越中線補正": "是 (+1分)" if front_mods.get("is_across_midline") else "否",
            f"{side_prefix}前臂向外側工作補正": "是 (+1分)" if front_mods.get("is_out_to_side") else "否（影像代理判斷）",
            "軀幹側彎補正": "是 (+1分)" if front_mods.get("is_side_bending") else "否（影像代理判斷）",
            "軀幹扭轉補正": "是 (+1分)" if manual_mods.get("trunk_twisted") else "否（人工確認）",
            "軀幹向後伸展": "是（軀幹基礎分 2）" if manual_mods.get("trunk_extended") else "否（人工確認）",
            "頸部扭轉補正": "是 (+1分)" if manual_mods.get("neck_twisted") else "否（人工確認）",
            "頸部後仰／伸展": "是（頸部基礎分 4）" if manual_mods.get("neck_extended") else "否（人工確認）",
            "自動辨識設定": ", ".join(k for k, v in auto_selection.items() if v) or "未啟用",
            "正面鏡頭轉向提醒": bool(front_mods.get("front_rotation_warning", False)),
            "跨越中線標準化位移": front_mods.get("cross_midline_normalized", np.nan),
            "軀幹側彎代理角度": front_mods.get("trunk_lateral_angle_proxy", np.nan),
            "頸部側彎代理角度": front_mods.get("neck_lateral_angle_proxy", np.nan),
            "上臂後伸自動判定成功": auto_extension_used,
            "上臂後伸判定備註": auto_extension_warning,
            f"{side_prefix}上臂分數": s_ua,
            f"{side_prefix}前臂分數": s_la,
            f"{side_prefix}手腕分數": s_wr,
            f"{side_prefix}手腕扭轉分數": wt_score,
            "頸部分數": s_nk,
            "軀幹分數": s_tk,
            "腿部分數": lg_score,
            "肌肉使用加分": m_val,
            "負重施力加分": f_val,
            f"{side_prefix} Score A": sc_a,
            "頸軀腿 Score B": sc_b,
            f"{side_prefix} Score C": sc_c,
            "頸軀腿 Score D": sc_d,
            f"{side_prefix} Grand Score": grand,
            f"{side_prefix} AL": al,
            "最危害側": target_side,
            "最終最高分": grand,
            "最終 AL": al
        }
        # 暫存於回傳 dict，呼叫端在建立 CSV 前 pop，避免大量照片或座標混入 CSV。
        data["_v6_hand_points"] = hand_comp["hand_points"]
        data["_v6_hand_angle"] = hand_angle
        data["_v7_keypoints"] = {
            "評估側": target_side, "影像寬": work_w, "影像高": work_h,
            "Pose影像像素座標": make_pose_keypoint_snapshot(
                n_lm, work_w, work_h, [0,2,5,7,8,11,12,13,14,15,16,17,18,19,20,23,24,25,26,27,28]),
            "Hand影像像素座標": hand_comp["hand_points"],
            "Hand模型標籤": hand_comp["handedness"],
            "角度原始估計": {"pose2d": raw_wrist_2d, "pose3d": raw_wrist_3d,
                         "pose混合": pose_wrist_angle, "hand2d": hand_angle,
                         "上臂": ang_upper_arm, "前臂屈曲": ang_lower_arm,
                         "軀幹2d投影": ang_trunk, "頸部2d代理": ang_neck},
        }
        return True, data, frame_annotated, conf_side
    except Exception:
        logging.exception("analyze_side_camera failed for %s", target_side)
        return False, None, frame_work, 0.0


# ==================== 【後處理：Hand 角度時序孤立尖峰檢查】 ====================
# 必須先定義，Streamlit 執行分析時才能正常呼叫。
# 此函式僅新增診斷欄位，不會修改 RULA 分數。
def add_hand_temporal_notes(df, side):
    """Post-hoc diagnostic only. Real fast movements are NOT automatically incorrect.

    Only mark a central sample when its two sampled neighbors agree closely, while
    the central Hand2D reading is distant from both. Does not alter RULA scoring.
    """
    if df is None or df.empty:
        return df
    out = df.copy().sort_values("時間(秒)")
    prefix = "右手" if side == "右手側" else "左手"
    column = f"{prefix}PoseHand實驗手腕角度"
    if column not in out:
        return out
    values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
    sample_ids = pd.to_numeric(out["樣本編號"], errors="coerce").to_numpy(dtype=float)
    flags = np.zeros(len(out), dtype=bool)
    for i in range(1,len(out)-1):
        if not (np.all(np.isfinite(values[i-1:i+2])) and
                sample_ids[i]-sample_ids[i-1] == 1 and
                sample_ids[i+1]-sample_ids[i] == 1):
            continue
        if (abs(values[i-1]-values[i+1]) <= 15 and
            abs(values[i]-values[i-1]) >= 35 and
            abs(values[i]-values[i+1]) >= 35):
            flags[i] = True
    out["Hand時序孤立尖峰（僅供研究）"] = flags
    return out.sort_index()



# ==================== 【V9.2：無裁切的內嵌圖片顯示】 ====================
def v92_image_html(image_data, caption="", encoded=False, max_width="100%"):
    """Data-URI JPEG image with natural aspect ratio; unlike fixed-height iframe, never crops bottom."""
    import html as html_lib
    if not image_data:
        return '<div>（無法取得圖片）</div>'
    payload = image_data if encoded else base64.b64encode(image_data).decode('ascii')
    width_css = '100%' if max_width == '100%' else str(max_width)
    return (f'<div style="text-align:center;width:100%;color:inherit">'
            f'<img alt="RULA影格" src="data:image/jpeg;base64,{payload}" '
            f'style="display:block;width:{width_css};max-width:100%;height:auto;object-fit:contain;'
            f'border-radius:8px;margin:0 auto" />'
            f'<div style="font-size:.85rem;margin:7px 0;opacity:.85">{html_lib.escape(str(caption))}</div>'
            '</div>')


def v92_show_image(image_data, caption="", encoded=False):
    """Avoid both Streamlit media URLs and components.html iframe clipping."""
    st.markdown(v92_image_html(image_data, caption=caption, encoded=encoded),
                unsafe_allow_html=True)


def v92_thumbnail_b64(image_b64, max_width=400):
    """Small preview without changing saved full-resolution JPEG."""
    try:
        original = cv2.imdecode(np.frombuffer(base64.b64decode(image_b64), dtype=np.uint8),
                                cv2.IMREAD_COLOR)
        if original is None:
            return image_b64
        h, w = original.shape[:2]
        if w > max_width:
            original = cv2.resize(original, (int(max_width), max(1, round(h * max_width / w))),
                                  interpolation=cv2.INTER_AREA)
        ok, out = cv2.imencode('.jpg', original, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return base64.b64encode(out).decode('ascii') if ok else image_b64
    except Exception:
        return image_b64


def v92_pose_wrist_qc(df, side):
    """Conservative POST-HOC Pose wrist warnings, no score/value modifications.

    Do not conflate pose 2D-vs-3D disagreement or real rapid motion with errors.
    An isolated spike requires contiguous samples with similar neighbors.
    """
    if df is None or df.empty:
        return df
    out = df.copy().sort_values('樣本編號')
    prefix = '右手' if side == '右手側' else '左手'
    angle_col = f'{prefix}Pose舊版手腕角度'
    if angle_col not in out.columns:
        return out
    angles = pd.to_numeric(out[angle_col], errors='coerce').to_numpy(dtype=float)
    ids = pd.to_numeric(out['樣本編號'], errors='coerce').to_numpy(dtype=float)
    notes = [''] * len(out)
    for i, angle in enumerate(angles):
        reasons = []
        if not np.isfinite(angle):
            reasons.append('Pose 手腕角無效')
        elif angle >= 65:
            reasons.append(f'手腕角度達 {angle:.1f}°，請查看手掌遮擋')
        if 0 < i < len(out)-1 and np.isfinite(angle):
            left, right = angles[i-1], angles[i+1]
            if (np.isfinite(left) and np.isfinite(right)
                and ids[i]-ids[i-1] == 1 and ids[i+1]-ids[i] == 1
                and abs(left-right) <= 15
                and abs(angle-left) >= 35 and abs(angle-right) >= 35):
                reasons.append('Pose 手腕孤立尖峰（非直接判定錯誤）')
        notes[i] = '；'.join(reasons)
    out['Pose手腕品質警示'] = notes
    out['Pose手腕需複核（保守）'] = [bool(x) for x in notes]
    return out.sort_index()


# ==================== 【V8：雙視角頸部品質分析，暫不改動正式 RULA 分數】 ====================
def v8_neck_candidate(landmarks, width, height):
    """Returns an experimental 2D neck-to-torso proxy; NOT a calibrated anatomical angle."""
    lm = landmarks
    def pt(i):
        return np.array([float(lm[i].x)*width, float(lm[i].y)*height])
    def vis(i):
        return float(getattr(lm[i], "visibility", 0.0) or 0.0)
    sh_mid = (pt(11)+pt(12))/2.0
    hip_mid = (pt(23)+pt(24))/2.0
    torso_up = sh_mid-hip_mid
    torso_len = float(np.linalg.norm(torso_up))
    base = {"angle": float('nan'), "quality": 0.0, "landmarks": "", "note": "", "valid": False}
    if torso_len < 18 or min(vis(i) for i in (11,12,23,24)) < 0.25:
        base['note'] = '肩髖中心品質不足'
        return base
    # Ears are the primary proxy; eyes only refine head direction if visible.
    ear_ids = [i for i in (7,8) if vis(i) >= .40]
    eye_ids = [i for i in (2,5) if vis(i) >= .45]
    if not ear_ids:
        base['note'] = '耳部關鍵點不可用'
        return base
    ear_ids.sort(key=vis, reverse=True)
    # Using both ears can help frontal projections; not claiming 3D pose recovery.
    chosen_ears = ear_ids[:2] if len(ear_ids)==2 else ear_ids[:1]
    ear_center = np.mean([pt(i) for i in chosen_ears], axis=0)
    quality_values = [vis(i) for i in chosen_ears]
    selected = [f'耳{i}' for i in chosen_ears]
    head_center = ear_center
    if eye_ids:
        eye_ids.sort(key=vis, reverse=True)
        chosen_eye = eye_ids[:2] if len(eye_ids)==2 else eye_ids[:1]
        eye_center = np.mean([pt(i) for i in chosen_eye], axis=0)
        head_center = .78*ear_center + .22*eye_center
        quality_values += [vis(i) for i in chosen_eye]
        selected += [f'眼{i}' for i in chosen_eye]
    v = head_center - sh_mid
    ratio = float(np.linalg.norm(v)/max(torso_len,1e-6))
    base['landmarks'] = ','.join(selected)
    if ratio < .12 or ratio > 1.0:
        base['note'] = '頭部－肩中心距離比例異常'
        return base
    angle = calculate_2d_angle(hip_mid, sh_mid, head_center)
    if not np.isfinite(angle):
        base['note'] = '頸部幾何向量退化'
        return base
    quality = min(quality_values+[vis(11),vis(12),vis(23),vis(24)])
    base.update(angle=round(abs(180.0-angle),2), quality=round(float(quality),3),
                valid=bool(quality >= .45), note='側視角候選角；尚未量角驗證')
    return base


def v8_neck_cross_view(right, left, max_diff=15.0, min_quality=0.55):
    """Never force incompatible camera estimates to agree. Candidate is diagnostic only."""
    try:
        rv,lv=float(right.get('頸部_v8_候選角度')),float(left.get('頸部_v8_候選角度'))
        rq,lq=float(right.get('頸部_v8_候選品質',0)),float(left.get('頸部_v8_候選品質',0))
    except (TypeError,ValueError):
        return float('nan'), float('nan'), '候選數據不足'
    if not (np.isfinite(rv) and np.isfinite(lv)):
        return float('nan'),float('nan'),'任一鏡頭沒有有效候選角'
    delta=round(abs(rv-lv),2)
    if rq < min_quality or lq < min_quality:
        return float('nan'),delta,'候選角品質不足；不融合'
    if delta > max_diff:
        return float('nan'),delta,'左右鏡頭差異過大；不融合'
    return round((rv*rq+lv*lq)/(rq+lq),2),delta,'品質與差異條件通過；僅供對照'


def v8_make_rejected_visual(frame, pose_model):
    """Use actual returned MediaPipe landmarks only. Failures show no fabricated skeleton."""
    if frame is None or not isinstance(frame,np.ndarray) or frame.size==0:
        return None,None,0,'影片無可讀取影格',0
    h,w=frame.shape[:2]
    scale=min(900.0/max(h,w),1.0)
    img=cv2.resize(frame,(max(1,int(w*scale)),max(1,int(h*scale)))) if scale<1 else frame.copy()
    raw_bytes=_encode_preview(img)
    draw=img.copy()
    result,quality,method=detect_pose_with_fallback(img,pose_model,required_indices=[7,8,11,12,13,14,15,16,23,24,25,26,27,28])
    n_points=0
    if result is not None and getattr(result,'pose_landmarks',None):
        lm=result.pose_landmarks.landmark
        hh,ww=draw.shape[:2]
        good={i for i,p in enumerate(lm) if (float(getattr(p,'visibility',0) or 0)>=.30 and
            np.isfinite(p.x) and np.isfinite(p.y) and -.1<=p.x<=1.1 and -.1<=p.y<=1.1)}
        n_points=len(good)
        for a,b in mp.solutions.pose.POSE_CONNECTIONS:
            if a in good and b in good:
                pa=(int(lm[a].x*ww),int(lm[a].y*hh)); pb=(int(lm[b].x*ww),int(lm[b].y*hh))
                cv2.line(draw,pa,pb,(0,210,255),2,cv2.LINE_AA)
        for i in good:
            p=lm[i]
            cv2.circle(draw,(int(p.x*ww),int(p.y*hh)),4,(0,255,128),-1,cv2.LINE_AA)
        note=f'Detected partial landmarks: {n_points}/33; method={method}'
    else:
        note='No pose landmarks returned; no skeleton drawn'
    cv2.putText(draw,note[:90],(8,28),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),2,cv2.LINE_AA)
    return raw_bytes,_encode_preview(draw),n_points,note,quality


def v8_add_rejected_sample(items, photos, sample_id, seconds, side, reason, camera_frames, pose_model):
    """Save rejected reason AND synchronized raw/diagnostic images for available cameras."""
    item={"樣本編號":int(sample_id),"時間(秒)":float(seconds),"評估側":side,"拒絕原因":reason,
          "建議查看鏡頭":"Front" if '正面' in reason else ('Right' if side=='右手側' else 'Left'),
          "評分狀態":"排除，不自動計入 RULA"}
    for camera,frame in camera_frames.items():
        raw,overlay,points,note,quality=v8_make_rejected_visual(frame,pose_model)
        if raw is None:
            continue
        key=f'Sample_{int(sample_id):03d}_{side}_{camera}'
        photos[key]={"raw":raw,"overlay":overlay,"points":points,"note":note,"quality":float(quality),"camera":camera}
        item[f'{camera}_原始圖片']=f'Diagnostics/Rejected_Images/{key}_Raw.jpg'
        item[f'{camera}_診斷圖片']=f'Diagnostics/Rejected_Images/{key}_Attempt.jpg'
    items.append(item)


def v8_zip_rejected_photos(zf, rejected_photos):
    for key,photos in rejected_photos.items():
        zf.writestr(f'Diagnostics/Rejected_Images/{key}_Raw.jpg',photos['raw'])
        zf.writestr(f'Diagnostics/Rejected_Images/{key}_Attempt.jpg',photos['overlay'])


def v8_encode_html_image(payload, caption=''):
    import html
    if not payload:
        return '<div style="padding:20px">無可用圖片</div>'
    return (f'<div style="text-align:center"><img src="data:image/jpeg;base64,{base64.b64encode(payload).decode()}" '
            f'style="max-width:100%;max-height:540px;border-radius:8px"/><p>{html.escape(caption)}</p></div>')


def v8_download_report_zip(original_zip, review_map):
    """Append manual review statuses without changing automated CSV or scores."""
    if not original_zip:
        return b''
    stream=io.BytesIO(original_zip)
    with zipfile.ZipFile(stream,'a',zipfile.ZIP_DEFLATED) as archive:
        rows=[{'樣本側別':k,'人工複核狀態':v} for k,v in sorted(review_map.items())]
        archive.writestr('Diagnostics/Rejected_Manual_Reviews.csv',
                         pd.DataFrame(rows,columns=['樣本側別','人工複核狀態']).to_csv(index=False).encode('utf-8-sig'))
    return stream.getvalue()


def v8_show_rejected_center():
    st.subheader('被剔除樣本：原圖與實際偵測嘗試')
    st.caption('這裡只顯示被排除的樣本；兩張圖來自相同時間與鏡頭，右圖為事後重新執行模型得到的診斷結果。標記人工狀態不會自動恢復 RULA 評分。')
    records=st.session_state.get('v8_rejected_rows',[])
    photos=st.session_state.get('v8_rejected_photos',{})
    if not records:
        st.success('此次沒有剔除樣本。')
        return
    st.metric('剔除紀錄數（側別）',len(records))
    labels=[f"第{r['樣本編號']}筆｜{r['時間(秒)']}秒｜{r['評估側']}｜{r['拒絕原因']}" for r in records]
    idx=st.selectbox('選擇紀錄',range(len(records)),format_func=lambda i:labels[i],key='v8_rejected_pick')
    item=records[idx]
    available=[camera for camera in ('Front','Right','Left')
               if f"Sample_{item['樣本編號']:03d}_{item['評估側']}_{camera}" in photos]
    if not available:
        st.warning('影片讀取失敗，沒有可用的原始影格。')
    else:
        preferred=item.get('建議查看鏡頭','Front')
        camera=st.radio('選擇鏡頭（預設失敗來源）',available,index=available.index(preferred) if preferred in available else 0,
                        horizontal=True,key=f"v8_rejected_camera_{item['樣本編號']}_{item['評估側']}")
        img=photos[f"Sample_{item['樣本編號']:03d}_{item['評估側']}_{camera}"]
        a,b=st.columns(2)
        with a:
            st.markdown('**未渲染原始照片**')
            v92_show_image(img['raw'])
        with b:
            st.markdown('**系統重新偵測（可能只有局部）**')
            v92_show_image(img['overlay'])
        st.caption(f"有效繪製點：{img['points']}/33；偵測品質代理值：{img['quality']:.2f}；{img['note']}")
    st.warning('拒絕原因：'+str(item['拒絕原因']))
    key=f"{item['樣本編號']}_{item['評估側']}"
    st.session_state.setdefault('v8_manual_rejected',{})
    options=['尚未複核','需要人工量測','照片可辨識，但不恢復自動評分','確認無法辨識']
    initial=st.session_state.v8_manual_rejected.get(key,'尚未複核')
    mark=st.selectbox('人工檢查註記（不影響自動 RULA 分數）',options,index=options.index(initial),key='v8_reject_status_'+key)
    st.session_state.v8_manual_rejected[key]=mark
    st.caption('人工註記會加入最終 ZIP 的 Diagnostics/Rejected_Manual_Reviews.csv。')


def v8_show_neck_center():
    st.subheader('頸部角度：左右鏡頭品質與候選估測')
    st.caption('v8 候選角與左右差異是實驗診斷，不會直接改寫原始 RULA 頸部分數。')
    df=st.session_state.get('df_records')
    if df is None or df.empty:
        st.info('目前沒有可用的有效分析資料。')
        return
    if st.session_state.get('analyzed_mode')=='左右側同步計算':
        cols=[c for c in ['樣本編號','時間(秒)','頸部角度','右側頸部原角度','左側頸部原角度',
              '頸部_v8_右側候選','頸部_v8_左側候選','頸部_v8_右側品質','頸部_v8_左側品質',
              '頸部_v8_候選差值','頸部_v8_候選融合角','頸部_v8_融合說明'] if c in df.columns]
        st.dataframe(df[cols],use_container_width=True,hide_index=True)
        if '頸部_v8_候選差值' in df.columns:
            gaps=pd.to_numeric(df['頸部_v8_候選差值'],errors='coerce')
            st.metric('候選左右差異中位數',f'{gaps.median():.1f}°' if gaps.notna().any() else '無足夠資料')
            st.line_chart(df.set_index('時間(秒)')[[c for c in ('右側頸部原角度','左側頸部原角度','頸部_v8_右側候選','頸部_v8_左側候選') if c in df.columns]])
    else:
        cols=[c for c in ['樣本編號','時間(秒)','頸部角度','頸部_v8_候選角度','頸部_v8_候選品質','頸部_v8_關鍵點','頸部_v8_診斷'] if c in df.columns]
        st.dataframe(df[cols],use_container_width=True,hide_index=True)
    st.info('左右差異變小 ≠ 接近人體真值。建議以人工參考角度驗證候選公式後，才考慮用於正式 RULA 計分。')

# ==================== 【V9：多攝影機時間同步，Pose/RULA 演算法不變】 ====================
# 每台影片的錨點是「相同真實事件」出現在該影片的秒數。
# 相機時間 = 共同時間 + 錨點。共同時間可以為負，代表同步事件之前的動作。
V9_AUTO_RATE_HZ = 4.0
V9_AUTO_MAX_SECONDS = 35.0
V9_AUTO_MAX_LAG_SECONDS = 9.0


def v9_frame_meta(fps, count):
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    count = max(0, int(count))
    return {"fps": fps, "count": count,
            "duration": count / fps,
            "last_time": max(0.0, (count - 1) / fps)}


def v9_common_window(metadata, anchors, cameras):
    """Return a shared time range where every selected camera has a frame.

    metadata maps camera labels to {'fps', 'count', 'duration', 'last_time'}.
    anchors maps labels to time of same real-world event in each source video.
    """
    if not cameras:
        raise ValueError('沒有任何可用鏡頭')
    start, finish = -float('inf'), float('inf')
    for cam in cameras:
        if cam not in metadata or cam not in anchors:
            raise ValueError(f'{cam} 影片資料或同步錨點不足')
        m = metadata[cam]
        if int(m['count']) < 2 or float(m['fps']) <= 0:
            raise ValueError(f'{cam} 影片無法讀取影格')
        anchor = float(anchors[cam])
        if not np.isfinite(anchor) or not (0 <= anchor <= float(m['last_time']) + 0.001):
            raise ValueError(f'{cam} 同步錨點超出影片範圍')
        start = max(start, -anchor)
        finish = min(finish, float(m['last_time']) - anchor)
    if finish <= start + 1e-5:
        raise ValueError('同步之後沒有共同重疊影格，請重新調整三個同步時間')
    return float(start), float(finish)


def v9_camera_seconds(shared_second, anchors, camera):
    return float(shared_second) + float(anchors[camera])


def v9_frame_index(shared_second, anchors, camera, metadata):
    seconds = v9_camera_seconds(shared_second, anchors, camera)
    fps = float(metadata[camera]['fps'])
    index = int(round(seconds * fps))
    return max(0, min(int(metadata[camera]['count']) - 1, index))


def v9_read_frame(cap, index):
    if index < 0 or not cap.set(cv2.CAP_PROP_POS_FRAMES, int(index)):
        return False, None
    ok, frame = cap.read()
    return bool(ok and frame is not None and frame.size > 0), frame


def v9_motion_features(video_path, limit_seconds=V9_AUTO_MAX_SECONDS, rate_hz=V9_AUTO_RATE_HZ):
    """Quick camera-independent *motion activity* proxy, NOT pose-based ground truth.

    Uses low-res grayscale frame differences. Avoids a separate Pose pass before analysis.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return np.array([], dtype=np.float64)
    meta = v9_frame_meta(cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if meta['count'] < 8:
        cap.release()
        return np.array([], dtype=np.float64)
    n = int(min(float(limit_seconds), meta['last_time']) * float(rate_hz)) + 1
    n = max(0, min(n, 260))
    energy = []
    prior = None
    for i in range(n):
        frame_idx = int(round(i / float(rate_hz) * meta['fps']))
        ok, frame = v9_read_frame(cap, min(meta['count'] - 1, frame_idx))
        if not ok:
            energy.append(float('nan'))
            prior = None
            continue
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(round(w * min(176.0 / max(w, h), 1.0))) or 1,
                                    int(round(h * min(176.0 / max(w, h), 1.0))) or 1))
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if prior is None or prior.shape != gray.shape:
            energy.append(0.0)
        else:
            energy.append(float(np.mean(cv2.absdiff(gray, prior))))
        prior = gray
    cap.release()
    return np.array(energy, dtype=np.float64)


def v9_best_motion_lag(reference, other, rate_hz=V9_AUTO_RATE_HZ,
                       max_lag_seconds=V9_AUTO_MAX_LAG_SECONDS):
    """Positive lag means the shared movement occurs *later* in other video.

    Returns diagnostic correlation and peak separation; suggest only, never auto-approve.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(other, dtype=np.float64)
    if len(a) < 24 or len(b) < 24:
        return {'usable': False, 'reason': '影片過短／可比較影格不足'}
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {'usable': False, 'reason': '部分取樣影格無法讀取'}
    if np.std(a) < 0.13 or np.std(b) < 0.13:
        return {'usable': False, 'reason': '動作訊號過弱，無法可靠對齊'}
    max_lag = int(round(max_lag_seconds * rate_hz))
    candidates = []
    for lag in range(-max_lag, max_lag + 1):
        # Compare reference[t] to other[t + lag]
        lower = max(0, -lag)
        upper = min(len(a), len(b) - lag)
        if upper - lower < min(24, int(min(len(a), len(b)) * 0.40)):
            continue
        x, y = a[lower:upper], b[lower + lag:upper + lag]
        sx, sy = float(np.std(x)), float(np.std(y))
        if sx < 0.1 or sy < 0.1:
            continue
        score = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(score):
            candidates.append((score, lag, upper - lower))
    if not candidates:
        return {'usable': False, 'reason': '找不到足夠共同動作訊號'}
    candidates.sort(reverse=True)
    best_score, lag, overlap = candidates[0]
    distant = [v for v in candidates if abs(v[1] - lag) > int(rate_hz * 0.8)]
    next_best = max((v[0] for v in distant), default=-1.0)
    margin = best_score - next_best
    if best_score >= 0.67 and margin >= 0.10:
        confidence = '較高'
    elif best_score >= 0.45 and margin >= 0.05:
        confidence = '中等'
    else:
        confidence = '低（請手動對齊）'
    return {'usable': True, 'lag_seconds': lag / float(rate_hz),
            'correlation': round(float(best_score), 3),
            'margin': round(float(margin), 3), 'confidence': confidence,
            'overlap_samples': int(overlap)}


def v9_suggest_anchors(motion, metadata, camera_names):
    """Use Front as reference; never mark the result 'confirmed'."""
    front = 'Front'
    if front not in camera_names:
        raise ValueError('必須有正面鏡頭')
    results = {}
    deltas = {front: 0.0}
    for cam in camera_names:
        if cam == front:
            continue
        result = v9_best_motion_lag(motion[front], motion[cam])
        results[cam] = result
        if not result.get('usable'):
            return {'usable': False, 'pairs': results, 'reason': f'{cam} 動作訊號不足'}
        deltas[cam] = float(result['lag_seconds'])
    low = max(0.0, *[-offset for offset in deltas.values()])
    high = min(float(metadata[c]['last_time']) - deltas[c] for c in camera_names)
    if high <= low:
        return {'usable': False, 'pairs': results, 'reason': '推薦偏移沒有共同影片區間'}
    front_energy = np.asarray(motion[front])
    lo_i, hi_i = max(1, int(low * V9_AUTO_RATE_HZ)), min(len(front_energy), int(high * V9_AUTO_RATE_HZ) + 1)
    if hi_i > lo_i:
        prominent = (np.argmax(front_energy[lo_i:hi_i]) + lo_i) / V9_AUTO_RATE_HZ
    else:
        prominent = (low + high) * 0.5
    reference_event = min(max(float(prominent), low), high)
    anchors = {cam: round(reference_event + deltas[cam], 3) for cam in camera_names}
    return {'usable': True, 'anchors': anchors, 'pairs': results,
            'caution': '自動比對使用動作強度，可能對到相似但不同次的動作；請查看預覽再按確認。'}


@st.cache_data(show_spinner=False, max_entries=9, ttl=2400)
def v9_video_metadata_from_bytes(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                raise ValueError('OpenCV 無法讀取影片編碼')
            return v9_frame_meta(cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
    finally:
        os.unlink(path)


@st.cache_data(show_spinner=False, max_entries=24, ttl=1200)
def v9_preview_jpeg(data, at_seconds, max_width=640):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        cap = cv2.VideoCapture(path)
        try:
            meta = v9_frame_meta(cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if meta['count'] < 1:
                return None
            index = max(0, min(meta['count'] - 1, int(round(float(at_seconds) * meta['fps']))))
            ok, frame = v9_read_frame(cap, index)
            if not ok:
                return None
            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, (max_width, max(1, int(h * max_width / w))))
            ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            return jpg.tobytes() if ok else None
        finally:
            cap.release()
    finally:
        os.unlink(path)


@st.cache_data(show_spinner=False, max_entries=5, ttl=1800)
def v9_motion_from_bytes(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        return v9_motion_features(path)
    finally:
        os.unlink(path)


def v9_sync_signature(upload_map, eval_mode):
    import hashlib
    names = []
    cams = ['Front', 'Right', 'Left'] if eval_mode.startswith('左右側') else (
        ['Front', 'Right'] if eval_mode == '右手側' else ['Front', 'Left'])
    for cam in cams:
        f = upload_map.get(cam)
        if f is None:
            names.append((cam, None))
        else:
            raw = f.getvalue()
            names.append((cam, getattr(f, 'name', ''), len(raw), hashlib.sha256(raw).hexdigest()[:16]))
    return tuple(names)


def v9_confirm_sync(mode):
    st.session_state.v9_confirmed = True
    st.session_state.v9_sync_mode = mode
    st.session_state.v9_confirmed_signature = v9_sync_signature(
        {'Front': st.session_state.v8_uploads.get('front'),
         'Right': st.session_state.v8_uploads.get('right'),
         'Left': st.session_state.v8_uploads.get('left')},
        st.session_state.v8_values.get('eval_mode', '右手側'))


def v9_sync_input_changed(name):
    st.session_state.v8_values['sync_'+name] = float(st.session_state['_v9_anchor_'+name])
    st.session_state.v9_confirmed = False


def v92_trim_changed():
    selected = tuple(float(x) for x in st.session_state['_v92_trim_range'])
    st.session_state.v8_values['v92_analysis_window'] = selected
    st.session_state.v9_confirmed = False


def v9_nudge(name, step):
    key = '_v9_anchor_'+name
    maximum = float(st.session_state.get('v9_max_'+name, 0.0))
    current = float(st.session_state.get(key, st.session_state.v8_values.get('sync_'+name, 0.0)))
    value = round(min(maximum, max(0.0, current + float(step))), 3)
    st.session_state.v8_values['sync_'+name] = value
    st.session_state[key] = value
    st.session_state.v9_confirmed = False


def v9_apply_auto_suggestion():
    suggestion = st.session_state.get('v9_auto_suggestion', {})
    if not suggestion.get('usable'):
        return
    for cam, value in suggestion['anchors'].items():
        key = '_v9_anchor_'+cam
        st.session_state.v8_values['sync_'+cam] = float(value)
        st.session_state[key] = float(value)
    st.session_state.v9_confirmed = False


def v9_use_zero_anchors(cameras):
    """Run before widget creation (Streamlit forbids modifying a rendered widget)."""
    for cam in cameras:
        st.session_state.v8_values['sync_'+cam]=0.0
        st.session_state['_v9_anchor_'+cam]=0.0
    v9_confirm_sync('未校正（使用者選擇 0 秒）')


def v9_sync_report(metadata, anchors, cameras, begin, end, mode):
    return pd.DataFrame([{
        '鏡頭': cam, '影片FPS': round(float(metadata[cam]['fps']), 5),
        '影片影格數': int(metadata[cam]['count']),
        '影片長度(秒)': round(float(metadata[cam]['duration']), 3),
        '同步事件出現時間(秒)': round(float(anchors[cam]), 3),
        '相對正面偏移(秒)': round(float(anchors[cam])-float(anchors['Front']), 3),
        '共同時間起點(秒)': round(float(begin), 3),
        '共同時間終點(秒)': round(float(end), 3),
        '校正方式': mode, '使用者確認': '是'
    } for cam in cameras])


def v9_make_sample_times(n, metadata, anchors, cameras, clip_range=None):
    begin, end = v9_common_window(metadata, anchors, cameras)
    if clip_range is not None:
        selected_begin, selected_end = map(float, clip_range)
        if not (np.isfinite(selected_begin) and np.isfinite(selected_end)):
            raise ValueError('分析起訖時間不是有效數字')
        begin, end = max(begin, selected_begin), min(end, selected_end)
        if end <= begin + 0.02:
            raise ValueError('選取的共同分析區間太短，請回到時間同步頁調整')
    return np.linspace(begin, end, int(n)), begin, end


def v9_sample_row(shared_time, metadata, anchors, cameras, sample_id, analysis_start=0.0):
    elapsed = float(shared_time) - float(analysis_start)
    result = {'樣本編號': int(sample_id), '時間(秒)': round(elapsed, 3),
              '分析經過時間(秒)': round(elapsed, 3),
              '共同時間(秒)': round(float(shared_time), 3),
              '同步事件相對時間(秒)': round(float(shared_time), 3)}
    for cam in cameras:
        absolute = v9_camera_seconds(shared_time, anchors, cam)
        idx = v9_frame_index(shared_time, anchors, cam, metadata)
        result[cam+'影片時間(秒)'] = round(absolute, 3)
        result[cam+'實際抽取時間估計(秒)'] = round(idx / metadata[cam]['fps'], 3)
        result[cam+'影格索引'] = idx
    return result


def v9_decorate_records(rows, time_rows, columns=None):
    for row in rows:
        sample_id = int(row.get('樣本編號', -1))
        if sample_id in time_rows:
            row.update({k: v for k, v in time_rows[sample_id].items() if k not in ('樣本編號', '時間(秒)')})
    return rows


# ==================== 【Streamlit 網頁應用介面】 ====================
st.set_page_config(page_title="RULA 三視角人因工程 AI 評估系統", layout="wide")
st.title("RULA v9.2 智慧姿勢評估｜同步分析")
st.caption(f"部署版本：{APP_BUILD_TAG}｜執行檔：{os.path.basename(__file__)}｜MediaPipe Pose complexity={POSE_MODEL_COMPLEXITY}")

# 啟動自檢：顯示實際部署版本並在分析前確認時序檢查函式已可呼叫。
# Streamlit 會在每次互動時由上到下重跑，因此函式必須在此之前宣告。
if not callable(globals().get("add_hand_temporal_notes")):
    st.error("部署自檢失敗：add_hand_temporal_notes 未完成定義。請檢查 GitHub 上傳的 222.py 是否完整。")
    st.stop()
st.caption("✅ v9.2：修復同步預覽與縮圖顯示；新增分析區間裁切、Pose 手腕離群值診斷；保留原始 RULA 分數。")

st.markdown(
    "> **系統說明**：本工具以 MediaPipe Pose 關鍵點估算部分姿勢角度，遇到關鍵點可見度不足時會嘗試原圖、對比增強及放大圖；仍不可靠的樣本會略過。RULA 部分影像補正可選擇自動判讀；手腕扭轉、腿足支撐、聳肩、上臂支撐及扭轉等項目仍需人工確認，輸出仍須由受訓評估者複核，不應單獨作為醫療或職安決策依據。"
)

if "v4_overrides" not in st.session_state:
    st.session_state.v4_overrides = {}

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.df_records = None
    st.session_state.df_right_only = None
    st.session_state.df_left_only = None
    st.session_state.display_images_b64 = []
    st.session_state.display_right_b64 = []
    st.session_state.display_left_b64 = []
    st.session_state.zip_bytes = None
    st.session_state.zip_filename = "RULA_Report.zip"
    st.session_state.analyzed_mode = ""
    st.session_state.audit_skips = []
    st.session_state.hand_review_images = {}
    st.session_state.hand_model_status = "未執行"

# ==================== 【V9.1：六步驟設定精簡導覽；結果另以分頁呈現】 ====================
# 「設定模式」僅包含分析前六步。成功分析後直接進入「結果模式」。
# 此模式與 step index 獨立，不讓結果頁被迫沿用上一頁／下一頁。
V8_STEPS = [
    "1｜影片與基本設定", "2｜三鏡頭時間同步", "3｜自動辨識設定",
    "4｜人工條件", "5｜Hand 實驗設定", "6｜啟動分析",
]
if 'v8_page_index' not in st.session_state:
    st.session_state.v8_page_index = 0
# 相容同一工作階段從 v9 舊版升級的索引（原本 7～13 是結果頁）。
_previous_result_page = int(st.session_state.v8_page_index) >= len(V8_STEPS)
st.session_state.v8_page_index = max(0, min(len(V8_STEPS)-1, int(st.session_state.v8_page_index)))
st.session_state.setdefault(
    'v9_screen',
    'results' if _previous_result_page and st.session_state.analysis_done else 'setup',
)
if st.session_state.v9_screen not in ('setup', 'results'):
    st.session_state.v9_screen = 'setup'
if st.session_state.v9_screen == 'results' and not st.session_state.analysis_done:
    st.session_state.v9_screen = 'setup'
st.session_state.setdefault('v8_values',{})
st.session_state.setdefault('v8_uploads',{})
st.session_state.setdefault('v8_rejected_rows',[])
st.session_state.setdefault('v8_rejected_photos',{})
st.session_state.setdefault('v8_manual_rejected',{})
st.session_state.setdefault('v9_confirmed',False)
st.session_state.setdefault('v9_sync_mode','尚未同步')
st.session_state.setdefault('v9_confirmed_signature',None)
st.session_state.setdefault('v9_auto_suggestion',None)
st.session_state.setdefault('v9_sync_summary',None)
for _cam in ('Front','Right','Left'):
    st.session_state.v8_values.setdefault('sync_'+_cam,0.0)



def v8_save_value(name):
    st.session_state.v8_values[name]=st.session_state.get('_v8_field_'+name)


def v8_value(name, default):
    return st.session_state.v8_values.setdefault(name,default)


def v8_widget(fun, label, name, default, **kwargs):
    v8_value(name, default)
    wkey='_v8_field_'+name
    if wkey not in st.session_state:
        st.session_state[wkey]=st.session_state.v8_values[name]
    result=getattr(st,fun)(label,key=wkey,on_change=v8_save_value,args=(name,),**kwargs)
    return result


def v8_upload_on_change(name):
    st.session_state.v8_uploads[name]=st.session_state.get('_v8_uploader_'+name)
    cam={'front':'Front','right':'Right','left':'Left'}.get(name)
    if cam:
        st.session_state.v8_values['sync_'+cam]=0.0
        st.session_state.pop('_v9_anchor_'+cam,None)
    st.session_state.v9_confirmed=False
    st.session_state.v9_auto_suggestion=None


def v8_upload(label, name):
    uploaded=st.file_uploader(label,type=['mp4','mov','avi'],key='_v8_uploader_'+name,
                              on_change=v8_upload_on_change,args=(name,))
    saved=st.session_state.v8_uploads.get(name)
    if uploaded is not None and saved is not uploaded:
        saved=uploaded
        st.session_state.v8_uploads[name]=uploaded
    if saved is not None:
        st.caption('已暫存：'+str(saved.name)+'（切換步驟不會清除）')
        if st.button('清除這支影片',key='v8_clear_'+name):
            st.session_state.v8_uploads[name]=None
            cam={'front':'Front','right':'Right','left':'Left'}.get(name)
            if cam:
                st.session_state.v8_values['sync_'+cam]=0.0
                st.session_state.pop('_v9_anchor_'+cam,None)
            st.session_state.v9_confirmed=False
            st.session_state.v9_auto_suggestion=None
            st.rerun()
    return saved


def v8_go(delta):
    """只在分析前六個設定步驟之間導覽。"""
    st.session_state.v8_page_index=max(0,min(len(V8_STEPS)-1,st.session_state.v8_page_index+delta))


def v8_jump():
    st.session_state.v8_page_index=V8_STEPS.index(st.session_state['_v8_jump_select'])


def v9_open_results():
    if st.session_state.analysis_done:
        st.session_state.v9_screen = 'results'


def v9_back_to_setup():
    # 不刪除上次分析結果；若改動了條件需重新啟動分析才能更新報表。
    st.session_state.v9_screen = 'setup'
    st.session_state.v8_page_index = min(st.session_state.v8_page_index, len(V8_STEPS)-1)


with st.sidebar:
    st.markdown('### RULA 操作導覽')
    if st.session_state.v9_screen == 'setup':
        st.session_state['_v8_jump_select'] = V8_STEPS[st.session_state.v8_page_index]
        st.selectbox('快速跳轉設定步驟',V8_STEPS,key='_v8_jump_select',on_change=v8_jump)
        if st.session_state.analysis_done:
            st.button('📊 查看上次分析結果',key='v9_show_previous_results',
                      on_click=v9_open_results,use_container_width=True)
    else:
        st.success('✅ 分析完成，請在主畫面切換結果分頁。')
        st.button('⚙️ 返回設定／重新分析',key='v9_return_settings',
                  on_click=v9_back_to_setup,use_container_width=True)
    st.caption('上傳影片與設定僅保留在目前瀏覽器工作階段。')

# 頂部只保留藍色進度條與目前步驟；不在頂部重複顯示導覽按鈕。
if st.session_state.v9_screen == 'setup':
    st.progress((st.session_state.v8_page_index+1)/len(V8_STEPS))
    st.markdown(
        f"<div style='text-align:center;font-weight:650;margin:0.45rem 0 1rem 0'>"
        f"{V8_STEPS[st.session_state.v8_page_index]}</div>",
        unsafe_allow_html=True,
    )

# 重要：參數與上傳檔案無論目前顯示哪個步驟，皆從非 widget 的永久狀態讀取。
EVAL_OPTIONS=['右手側','左手側','左右側同步計算 (需上傳正面、右手側、左手側三個視角影片)']
MUSCLE_OPTIONS=['0: 非靜態且動作少於每分鐘 4 次','1: 姿勢維持超過 1 分鐘或動作每分鐘至少 4 次']
FORCE_OPTIONS=['0: < 2kg (間歇施力)','1: 2-10kg (間歇施力)','2: 2-10kg 靜態／重複施力','3: >10kg 負載、重複施力或衝擊力']
LEG_OPTIONS=['請先目視確認…','腿足受支撐且平衡（1分）','腿足未受支撐或不平衡（2分）']
TWIST_OPTIONS=['請先目視確認…','中間範圍（1分）','接近旋轉極限（2分）']
AUTO_FLAGS={
    'is_abducted':'上臂外展（正面肩肘相對位置）',
    'is_across_midline':'前臂跨越身體中線（正面向量投影）',
    'is_out_to_side':'前臂向外側工作（正面位置）',
    'is_side_bending':'軀幹側彎（肩髖中心線）',
    'is_neck_side_bending':'頸部側彎（頭頸相對軀幹）',
    'upper_arm_extended':'上臂向後伸展（側面朝向判斷；實驗性）',
}
front_labels={
    'is_abducted':'上臂外展（+1）',
    'is_across_midline':'前臂跨中線（+1）',
    'is_out_to_side':'前臂向外側工作（+1）',
    'is_side_bending':'軀幹側彎（+1）',
    'is_neck_side_bending':'頸部側彎（+1）',
}
for flag in AUTO_FLAGS:
    v8_value('auto_'+flag,flag!='upper_arm_extended')
for flag in front_labels:
    for side in ('右手側','左手側'):
        v8_value('manual_'+side+'_'+flag,False)
auto_selection={flag:v8_value('auto_'+flag,flag!='upper_arm_extended') for flag in AUTO_FLAGS}
manual_front={(side,flag):v8_value('manual_'+side+'_'+flag,False)
              for side in ('右手側','左手側') for flag in front_labels}

# Defaults for all settings, even when their page isn't active.
eval_mode=v8_value('eval_mode',EVAL_OPTIONS[0])
num_samples=v8_value('num_samples',25)
combo_muscle=v8_value('muscle',MUSCLE_OPTIONS[0])
combo_force=v8_value('force',FORCE_OPTIONS[0])
vid_front_file=st.session_state.v8_uploads.get('front')
vid_right_file=st.session_state.v8_uploads.get('right')
vid_left_file=st.session_state.v8_uploads.get('left')
leg_support_choice=v8_value('leg',LEG_OPTIONS[0])
neck_twisted_manual=v8_value('neck_twisted',False)
neck_extended_manual=v8_value('neck_extended',False)
trunk_twisted_manual=v8_value('trunk_twisted',False)
trunk_extended_manual=v8_value('trunk_extended',False)
right_wrist_twist_choice=v8_value('rw_twist',TWIST_OPTIONS[0])
left_wrist_twist_choice=v8_value('lw_twist',TWIST_OPTIONS[0])
right_wrist_deviated=v8_value('rw_deviated',False)
left_wrist_deviated=v8_value('lw_deviated',False)
right_shoulder_raised=v8_value('rs_raised',False)
left_shoulder_raised=v8_value('ls_raised',False)
right_arm_supported=v8_value('ra_supported',False)
left_arm_supported=v8_value('la_supported',False)
right_arm_extended=v8_value('ra_extended',False)
left_arm_extended=v8_value('la_extended',False)
v6_enable_hand=v8_value('hand_enabled',True)
v6_policy_label=v8_value('hand_policy','安全比較模式：RULA 維持 Pose v5 計分')
raw_image_export=v8_value('raw_export',True)

page=st.session_state.v8_page_index if st.session_state.v9_screen == "setup" else -1
if page == 0:
    st.subheader('上傳影片與作業設定')
    eval_mode=v8_widget('radio','評估模式','eval_mode',EVAL_OPTIONS[0],options=EVAL_OPTIONS,horizontal=True)
    a,b,c=st.columns(3)
    with a: vid_front_file=v8_upload('正面鏡頭（必要）','front')
    with b: vid_right_file=v8_upload('右側鏡頭','right')
    with c: vid_left_file=v8_upload('左側鏡頭','left')
    d,e,f=st.columns(3)
    with d: num_samples=v8_widget('slider','抽樣數量','num_samples',25,min_value=5,max_value=100)
    with e: combo_muscle=v8_widget('selectbox','肌肉使用加分','muscle',MUSCLE_OPTIONS[0],options=MUSCLE_OPTIONS)
    with f: combo_force=v8_widget('selectbox','負重施力加分','force',FORCE_OPTIONS[0],options=FORCE_OPTIONS)
    st.info('三台攝影機不需要同時開始或拍一樣長；下一步會校正共同時間，之後才正式分析。')

elif page == 1:
    st.subheader('三鏡頭時間同步｜只需校正一次')
    st.caption('三支影片不用同時按錄影，也不用片長一樣。選同一個動作瞬間，系統會自動換算後續每一張影格的時間。')
    active_cams = (['Front','Right','Left'] if eval_mode.startswith('左右側')
                   else ['Front', 'Right' if eval_mode == '右手側' else 'Left'])
    upload_map = {'Front': vid_front_file,'Right': vid_right_file,'Left': vid_left_file}
    available = all(upload_map[c] is not None for c in active_cams)
    if not available:
        st.warning('請先回到第 1 步，上傳此評估模式所需的正面及側面影片。')
    else:
        try:
            meta = {c:v9_video_metadata_from_bytes(upload_map[c].getvalue()) for c in active_cams}
        except Exception as exc:
            st.error(f'影片資訊讀取失敗：{exc}')
            meta = None
        if meta:
            sig = v9_sync_signature(upload_map, eval_mode)
            if st.session_state.v9_confirmed_signature != sig:
                st.session_state.v9_confirmed = False
            st.markdown('#### A. 快速自動建議（選用）')
            st.caption('比對三段影片的動作強度曲線；攝影角度差很多或反覆動作時可能建議錯誤，**不會自動套用**。')
            if st.button('✨ 分析動作並建議時間偏移', key='v9_find_motion', use_container_width=True):
                try:
                    with st.spinner('正在以低解析度計算三支影片的動作變化…'):
                        motion = {c:v9_motion_from_bytes(upload_map[c].getvalue()) for c in active_cams}
                        st.session_state.v9_auto_suggestion = v9_suggest_anchors(motion,meta,active_cams)
                except Exception as exc:
                    st.session_state.v9_auto_suggestion = {'usable':False,'reason':str(exc)}
            suggestion = st.session_state.get('v9_auto_suggestion')
            if suggestion:
                if suggestion.get('usable'):
                    st.dataframe(pd.DataFrame([{'鏡頭':cam,'建議同步事件秒數':suggestion['anchors'][cam],
                        '動作相關性':suggestion['pairs'][cam]['correlation'] if cam!='Front' else 1.0,
                        '建議可信度':suggestion['pairs'][cam]['confidence'] if cam!='Front' else '參考鏡頭'}
                        for cam in active_cams]),hide_index=True,use_container_width=True)
                    st.warning(suggestion['caution'])
                    st.button('套用建議秒數，再查看三畫面', key='v9_apply_motion',
                              on_click=v9_apply_auto_suggestion,use_container_width=True)
                else:
                    st.info('自動同步不夠可靠：'+suggestion.get('reason','無法估計')+'。請直接使用下方手動對齊。')
            st.markdown('#### B. 三畫面同步預覽與微調')
            st.caption('找三支影片共有的動作，例如「手第一次碰到物品」。下方三個時間都指向同一事件時，按「確認同步」。')
            cols = st.columns(len(active_cams))
            anchors_now = {}
            for col,cam in zip(cols,active_cams):
                with col:
                    title = {'Front':'正面','Right':'右側','Left':'左側'}[cam]
                    m=meta[cam]
                    st.markdown('**'+title+'鏡頭**')
                    st.caption(f"長 {m['duration']:.2f} 秒｜{m['fps']:.2f} FPS")
                    st.session_state['v9_max_'+cam]=float(m['last_time'])
                    widget_key='_v9_anchor_'+cam
                    start_value=float(st.session_state.v8_values.get('sync_'+cam,0.0))
                    start_value=min(max(0.0,start_value),float(m['last_time']))
                    st.session_state.v8_values['sync_'+cam]=start_value
                    if widget_key not in st.session_state:
                        st.session_state[widget_key]=start_value
                    st.number_input('共同事件位於影片第幾秒',min_value=0.0,
                        max_value=float(m['last_time']),step=round(1.0/m['fps'],4),
                        format='%.3f', key=widget_key,
                        on_change=v9_sync_input_changed,args=(cam,))
                    val=float(st.session_state[widget_key])
                    anchors_now[cam]=val
                    picture=v9_preview_jpeg(upload_map[cam].getvalue(),round(val,2))
                    if picture is not None:
                        v92_show_image(picture, f'{title}｜{val:.3f} 秒')
                    else:
                        st.error('預覽影格讀取失敗，請確認影片編碼')
                    x,y=st.columns(2)
                    with x:
                        st.button('← 前一幀',key='v9_prev_'+cam,on_click=v9_nudge,
                                  args=(cam,-1.0/m['fps']),use_container_width=True)
                    with y:
                        st.button('後一幀 →',key='v9_next_'+cam,on_click=v9_nudge,
                                  args=(cam,1.0/m['fps']),use_container_width=True)
            try:
                begin,end=v9_common_window(meta,anchors_now,active_cams)
                st.success(f'校正後共同可分析範圍：{begin:+.2f} ～ {end:+.2f} 秒（共 {end-begin:.2f} 秒）')
                st.caption('共同時間 0 秒是選定的動作事件；負秒代表事件之前。各影片採樣會自動換算。')
                st.markdown('#### C. 選擇真正要評估的動作區間')
                st.caption('照片若只看到天花板、受測者還沒進入畫面，請縮短區間；不必把整段影片都評分。')
                trim_sig = (tuple(round(float(anchors_now[c]),3) for c in active_cams),
                            round(begin,3), round(end,3))
                if st.session_state.get('_v92_trim_signature') != trim_sig:
                    st.session_state['_v92_trim_signature'] = trim_sig
                    st.session_state['_v92_trim_range'] = (float(begin),float(end))
                    st.session_state.v8_values['v92_analysis_window'] = (float(begin),float(end))
                    st.session_state.v9_confirmed = False
                chosen_range = st.slider('共同時間軸：動作分析起點～終點（秒）',
                    min_value=float(begin), max_value=float(end),
                    value=(float(begin),float(end)),
                    step=min(0.1, max(0.001, (end-begin)/20)),
                    key='_v92_trim_range',on_change=v92_trim_changed)
                st.session_state.v8_values['v92_analysis_window'] = tuple(map(float,chosen_range))
                st.info(f'本次將只分析共同時間 {chosen_range[0]:+.2f} ～ {chosen_range[1]:+.2f} 秒'
                        f'，共 {chosen_range[1]-chosen_range[0]:.2f} 秒。'
                        '輸出圖表的時間從分析第 0 秒開始；ZIP 仍保存各鏡頭原始時間。')
                if st.checkbox('查看分析起點的三鏡頭畫面（確認受測者已進入鏡頭）',
                               key='v92_preview_trim_start', value=False):
                    preview_cols=st.columns(len(active_cams))
                    for pcol,pcam in zip(preview_cols,active_cams):
                        with pcol:
                            source_second=float(anchors_now[pcam])+float(chosen_range[0])
                            shot=v9_preview_jpeg(upload_map[pcam].getvalue(),round(source_second,2))
                            if shot:
                                v92_show_image(shot,f'{pcam}｜影片 {source_second:.2f} 秒')
                            else:
                                st.warning(f'{pcam} 起點預覽失敗')
                if max(anchors_now.values()) > 0.7 * max(m['last_time'] for m in meta.values()):
                    st.warning('目前同步錨點靠近影片尾端：完整重疊區間將包含很多事件之前的畫面。'
                               '如果前段沒有實際作業，請使用上方滑桿裁切。')
                st.button('✅ 三畫面已對到同一動作，確認同步', key='v9_confirm',
                          on_click=v9_confirm_sync,args=('預覽確認',),type='primary',use_container_width=True)
                if st.session_state.v9_confirmed and st.session_state.v9_confirmed_signature==sig:
                    st.success('已確認同步，可以按下一步繼續。')
                with st.expander('沒有時間校正需求？可明確選擇 0 秒基準'):
                    st.warning('只有確定三台攝影機同時開始錄影，才建議跳過。')
                    st.button('使用全部 0 秒，並標記為未校正',key='v9_no_alignment',
                              on_click=v9_use_zero_anchors,args=(active_cams,))
            except ValueError as exc:
                st.error(f'同步時間沒有有效重疊：{exc}')

elif page == 2:
    st.subheader('自動辨識條件')
    st.caption('勾選表示使用影像向量代理判斷；取消後會顯示右側／左側的手動勾選。')
    cols=st.columns(2)
    for i,(flag,label) in enumerate(AUTO_FLAGS.items()):
        with cols[i%2]:
            auto_selection[flag]=v8_widget('checkbox',label,'auto_'+flag,flag!='upper_arm_extended')
    with st.expander('僅設定未自動判斷的項目',expanded=True):
        for flag,label in front_labels.items():
            if not auto_selection[flag]:
                a,b=st.columns(2)
                with a: manual_front[('右手側',flag)]=v8_widget('checkbox','右側｜'+label,'manual_右手側_'+flag,False)
                with b: manual_front[('左手側',flag)]=v8_widget('checkbox','左側｜'+label,'manual_左手側_'+flag,False)
    st.caption('影像估測值並不是臨床量測精度；正面鏡頭關鍵點不足時，啟用自動辨識的樣本會被略過。')

elif page == 3:
    st.subheader('人工確認條件（必要項）')
    st.caption('影像無法確認實際負重、支撐或前臂旋轉時，仍需輸入。以下會套用整段影片。')
    a,b,c=st.columns(3)
    with a:
        leg_support_choice=v8_widget('selectbox','腿足支撐／平衡','leg',LEG_OPTIONS[0],options=LEG_OPTIONS)
        neck_twisted_manual=v8_widget('checkbox','頸部扭轉（+1）','neck_twisted',False)
        neck_extended_manual=v8_widget('checkbox','頸部後仰','neck_extended',False)
        trunk_twisted_manual=v8_widget('checkbox','軀幹扭轉（+1）','trunk_twisted',False)
        trunk_extended_manual=v8_widget('checkbox','軀幹向後伸展','trunk_extended',False)
    with b:
        st.markdown('**右手側**')
        right_wrist_twist_choice=v8_widget('selectbox','右腕扭轉','rw_twist',TWIST_OPTIONS[0],options=TWIST_OPTIONS)
        right_wrist_deviated=v8_widget('checkbox','右腕尺／橈偏','rw_deviated',False)
        right_shoulder_raised=v8_widget('checkbox','右肩抬高','rs_raised',False)
        right_arm_supported=v8_widget('checkbox','右臂有支撐','ra_supported',False)
        right_arm_extended=v8_widget('checkbox','右臂向後伸展（手動備用）','ra_extended',False)
    with c:
        st.markdown('**左手側**')
        left_wrist_twist_choice=v8_widget('selectbox','左腕扭轉','lw_twist',TWIST_OPTIONS[0],options=TWIST_OPTIONS)
        left_wrist_deviated=v8_widget('checkbox','左腕尺／橈偏','lw_deviated',False)
        left_shoulder_raised=v8_widget('checkbox','左肩抬高','ls_raised',False)
        left_arm_supported=v8_widget('checkbox','左臂有支撐','la_supported',False)
        left_arm_extended=v8_widget('checkbox','左臂向後伸展（手動備用）','la_extended',False)

elif page == 4:
    st.subheader('Pose + Hand 實驗設定')
    v6_enable_hand=v8_widget('checkbox','啟用 Hand Landmarker（21 點／手）','hand_enabled',True)
    v6_policy_label=v8_widget('radio','手腕評分來源','hand_policy','安全比較模式：RULA 維持 Pose v5 計分',
        options=['安全比較模式：RULA 維持 Pose v5 計分','實驗計分模式：幾何品質達標才採 Hand 角度（研究用途）'],
        disabled=not v6_enable_hand)
    raw_image_export=v8_widget('checkbox','ZIP 保存未繪製骨架的原始影格','raw_export',True)
    st.info('建議選「安全比較」；Hand 2D 估測與 Pose 3D 不可直接當作已校準的共同 3D 角度。')

def build_manual_mods(side_name):
    is_right = side_name == "右手側"
    twist_choice = right_wrist_twist_choice if is_right else left_wrist_twist_choice
    return {
        "auto_selection": dict(auto_selection),
        "manual_front": {flag: manual_front.get((side_name, flag), False) for flag in front_labels},
        "leg_score": 1 if leg_support_choice == "腿足受支撐且平衡（1分）" else 2,
        "wrist_twist_score": 1 if twist_choice == "中間範圍（1分）" else 2,
        "wrist_deviated": right_wrist_deviated if is_right else left_wrist_deviated,
        "shoulder_raised": right_shoulder_raised if is_right else left_shoulder_raised,
        "arm_supported": right_arm_supported if is_right else left_arm_supported,
        "upper_arm_extended": right_arm_extended if is_right else left_arm_extended,
        "neck_twisted": neck_twisted_manual,
        "neck_extended": neck_extended_manual,
        "trunk_twisted": trunk_twisted_manual,
        "trunk_extended": trunk_extended_manual,
    }

# 當全部正面鏡頭自動補正都已關閉，可用手動設定繼續，避免因正面關鍵點不足而中斷。
def manual_front_fallback():
    front_keys = ("is_abducted", "is_across_midline", "is_out_to_side", "is_side_bending", "is_neck_side_bending")
    if any(auto_selection.get(key, True) for key in front_keys):
        return None
    return {key: False for key in front_keys}


# 在非設定頁也必須由保存的資料建立每次分析的選項。
v6_hand_policy='gated' if v6_enable_hand and v6_policy_label.startswith('實驗') else 'compare'
if page == 5:
    st.subheader('啟動分析')
    ready_files=[name for name,f in [('正面',vid_front_file),('右側',vid_right_file),('左側',vid_left_file)] if f]
    st.write('**已上傳：**'+('、'.join(ready_files) if ready_files else '尚未上傳影片'))
    st.write(f'**抽樣數：**{num_samples}　｜　**分析模式：**{eval_mode}　｜　**Hand：**'+('開啟' if v6_enable_hand else '關閉'))
    if leg_support_choice==LEG_OPTIONS[0] or any((right_wrist_twist_choice if side=='右手側' else left_wrist_twist_choice)==TWIST_OPTIONS[0]
         for side in (('右手側','左手側') if eval_mode.startswith('左右側') else (eval_mode,))):
        st.warning('請先返回「人工條件」填妥腿足支撐、所選側別的手腕扭轉，才能開始分析。')
    st.write('**時間同步：**'+('已確認（'+st.session_state.v9_sync_mode+'）' if st.session_state.v9_confirmed else '尚未確認'))
    if not st.session_state.v9_confirmed:
        st.warning('先到「三鏡頭時間同步」預覽畫面並確認，否則不會開始分析。')
    st.caption('執行後自動跳到結果總覽；原本的三視角分析及 ZIP 報表功能皆保留。')

# 執行分析按鈕
if page == 5 and st.button("🚀 啟動全關節生物力學分析", type="primary", use_container_width=True):
    is_dual_mode = eval_mode.startswith("左右側同步計算")

    # 嚴格檢查影片上傳完整性
    if is_dual_mode:
        if not (vid_front_file and vid_right_file and vid_left_file):
            st.error("❌ 您選擇了【左右側同步計算】，請務必同時上傳「1. 正面鏡頭」、「2. 右手側鏡頭」與「3. 左手側鏡頭」三個視角的影片！")
            st.stop()
    else:
        selected_side_vid = vid_right_file if eval_mode == "右手側" else vid_left_file
        if not vid_front_file:
            st.error("❌ 缺少【正面鏡頭影片】！正面鏡頭是用於校正水平面外展與中線穿越的必要基準。")
            st.stop()
        elif not selected_side_vid:
            st.error(f"❌ 缺少【{eval_mode}鏡頭影片】！您選擇評估「{eval_mode}」，必須提供該側面的拍攝視角進行主屈曲量測。")
            st.stop()

    # V9: prevent analyzing mismatched or stale video synchronization settings.
    sync_inputs={'Front':vid_front_file,'Right':vid_right_file,'Left':vid_left_file}
    active_cams=['Front','Right','Left'] if is_dual_mode else (['Front','Right'] if eval_mode=='右手側' else ['Front','Left'])
    if (not st.session_state.get('v9_confirmed') or
        st.session_state.get('v9_confirmed_signature') != v9_sync_signature(sync_inputs, eval_mode)):
        st.error('尚未確認這一組影片的同步錨點。請先到第 2 步預覽並按「確認同步」。')
        st.stop()
    sync_anchors={cam:float(st.session_state.v8_values.get('sync_'+cam,0.0)) for cam in active_cams}
    sync_mode=st.session_state.v9_sync_mode

    required_manual_sides = ["右手側", "左手側"] if is_dual_mode else [eval_mode]
    if leg_support_choice == "請先目視確認…":
        st.error("請先人工確認腿與足部是否受支撐且平衡，再啟動分析。")
        st.stop()
    for required_side in required_manual_sides:
        twist_choice = right_wrist_twist_choice if required_side == "右手側" else left_wrist_twist_choice
        if twist_choice == "請先目視確認…":
            st.error(f"請先人工確認{required_side}手腕扭轉狀態，再啟動分析。")
            st.stop()

    with st.spinner(f"系統正在進行 33 點姿態特徵擷取與 RULA 解算（模式：{eval_mode}）..."):
        st.session_state.v4_overrides = {}
        st.session_state.display_images_b64 = []
        st.session_state.display_right_b64 = []
        st.session_state.display_left_b64 = []
        st.session_state.zip_bytes = None
        st.session_state.hand_review_images = {}
        st.session_state.v8_rejected_rows = []
        st.session_state.v8_rejected_photos = {}
        st.session_state.v8_manual_rejected = {}
        st.session_state.hand_model_status = "未啟用"
        st.session_state.v9_sync_summary = None
        gc.collect()

        # 模型失敗只關閉 Hand 比較，不影響既有 Pose/RULA 分析。
        hand_task = None
        if v6_enable_hand:
            try:
                hand_asset_path = ensure_hand_model()
                hand_task = make_hand_landmarker(hand_asset_path)
                st.session_state.hand_model_status = "已啟用 Hand Landmarker"
            except Exception as exc:
                logging.exception("Hand model could not be initialized")
                st.session_state.hand_model_status = f"Hand 未啟用：{type(exc).__name__}: {exc}"
                st.warning("Hand 模型無法下載或初始化，本次仍使用 Pose 原版進行 RULA 分析。請查看模型路徑或將 hand_landmarker.task 放入 models/。")

        m_val = int(combo_muscle.split(":")[0])
        f_val = int(combo_force.split(":")[0])

        # ==================== 【模式 A：左右側同步計算 (三鏡頭分開計算、分析與擷取)】 ====================
        if is_dual_mode:
            t_front = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_front.write(vid_front_file.getvalue()); t_front.close()

            t_right = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_right.write(vid_right_file.getvalue()); t_right.close()

            t_left = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_left.write(vid_left_file.getvalue()); t_left.close()

            cap_front = cv2.VideoCapture(t_front.name)
            cap_right = cv2.VideoCapture(t_right.name)
            cap_left  = cv2.VideoCapture(t_left.name)

            fps_front = float(cap_front.get(cv2.CAP_PROP_FPS) or 30.0)
            fps_right = float(cap_right.get(cv2.CAP_PROP_FPS) or 30.0)
            fps_left = float(cap_left.get(cv2.CAP_PROP_FPS) or 30.0)
            count_front = int(cap_front.get(cv2.CAP_PROP_FRAME_COUNT))
            count_right = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))
            count_left = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_front = fps_front if np.isfinite(fps_front) and fps_front > 0 else 30.0
            fps_right = fps_right if np.isfinite(fps_right) and fps_right > 0 else 30.0
            fps_left = fps_left if np.isfinite(fps_left) and fps_left > 0 else 30.0
            sync_meta={'Front':v9_frame_meta(fps_front,count_front),
                       'Right':v9_frame_meta(fps_right,count_right),
                       'Left':v9_frame_meta(fps_left,count_left)}
            try:
                sample_times, sync_start, sync_end = v9_make_sample_times(
                    num_samples,sync_meta,sync_anchors,active_cams,
                    clip_range=st.session_state.v8_values.get('v92_analysis_window'))
            except ValueError as exc:
                for c in [cap_front,cap_right,cap_left]: c.release()
                for p in [t_front.name,t_right.name,t_left.name]:
                    try: os.remove(p)
                    except OSError: pass
                st.error('影片無法建立同步區間：'+str(exc))
                st.stop()
            sync_report=v9_sync_report(sync_meta,sync_anchors,active_cams,sync_start,sync_end,sync_mode)
            sync_samples={i+1:v9_sample_row(t,sync_meta,sync_anchors,active_cams,i+1,analysis_start=sync_start)
                          for i,t in enumerate(sample_times)}

            combined_records = []
            right_only_records = []
            left_only_records = []
            zip_right_imgs = []
            zip_left_imgs = []
            zip_hand_imgs = []
            zip_raw_imgs = []
            keypoint_records = []
            display_right_b64 = []
            display_left_b64 = []
            skipped_right = 0
            skipped_left = 0
            skip_diagnostics = []
            rejected_rows = []
            rejected_photos = {}

            progress_bar = st.progress(0)
            status_text = st.empty()

            with mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=POSE_MODEL_COMPLEXITY,
                min_detection_confidence=0.50
            ) as pose:
                total_samples = len(sample_times)
                for i, sample_time in enumerate(sample_times):
                    sec = round(float(sample_time - sync_start), 2)
                    idx_front = sync_samples[i+1]['Front影格索引']
                    idx_right = sync_samples[i+1]['Right影格索引']
                    idx_left = sync_samples[i+1]['Left影格索引']
                    cap_front.set(cv2.CAP_PROP_POS_FRAMES, idx_front)
                    cap_right.set(cv2.CAP_PROP_POS_FRAMES, idx_right)
                    cap_left.set(cv2.CAP_PROP_POS_FRAMES, idx_left)

                    s_f, f_frame = cap_front.read()
                    s_r, r_frame = cap_right.read()
                    s_l, l_frame = cap_left.read()

                    if s_f and s_r and s_l:
                        if raw_image_export:
                            for label, raw in (("Front", f_frame), ("Right", r_frame), ("Left", l_frame)):
                                payload = _encode_preview(raw)
                                if payload:
                                    zip_raw_imgs.append((f"Raw_Frames/{label}/Sample_{i+1:03d}_{sec}s.jpg", payload))
                        # 1. 正面鏡頭分別提取右手與左手的冠狀面外展/交叉補正
                        front_ok_r, front_mods_r = analyze_front_camera(f_frame, pose, "右手側")
                        front_ok_l, front_mods_l = analyze_front_camera(f_frame, pose, "左手側")
                        if not front_ok_r:
                            front_mods_r = manual_front_fallback()
                        if not front_ok_l:
                            front_mods_l = manual_front_fallback()

                        # 2. 右側鏡頭只評估右側關鍵點；低品質正面補正會使該樣本略過，不默認為「否」。
                        succ_r, data_r, img_r, conf_r = analyze_side_camera(
                            r_frame, pose, "右手側", front_mods_r, m_val, f_val,
                            manual_mods=build_manual_mods("右手側"),
                            hand_task=hand_task, hand_scoring_policy=v6_hand_policy
                        )

                        # 3. 左側鏡頭只評估左側關鍵點。
                        succ_l, data_l, img_l, conf_l = analyze_side_camera(
                            l_frame, pose, "左手側", front_mods_l, m_val, f_val,
                            manual_mods=build_manual_mods("左手側"),
                            hand_task=hand_task, hand_scoring_policy=v6_hand_policy
                        )
                        if not succ_r or data_r is None:
                            skipped_right += 1
                            skip_diagnostics.append({"樣本編號": i+1, "時間(秒)": sec, "評估側": "右手側", "可能原因": "正面關鍵點不足" if not front_ok_r else "側面關鍵點不足或角度無效"})
                            v8_add_rejected_sample(rejected_rows,rejected_photos,i+1,sec,"右手側",
                                "正面關鍵點不足" if not front_ok_r else "側面關鍵點不足或角度無效",
                                {"Front":f_frame,"Right":r_frame,"Left":l_frame},pose)
                        if not succ_l or data_l is None:
                            skipped_left += 1
                            skip_diagnostics.append({"樣本編號": i+1, "時間(秒)": sec, "評估側": "左手側", "可能原因": "正面關鍵點不足" if not front_ok_l else "側面關鍵點不足或角度無效"})
                            v8_add_rejected_sample(rejected_rows,rejected_photos,i+1,sec,"左手側",
                                "正面關鍵點不足" if not front_ok_l else "側面關鍵點不足或角度無效",
                                {"Front":f_frame,"Right":r_frame,"Left":l_frame},pose)

                        if succ_r and data_r is not None:
                            hand_points_r = data_r.pop("_v6_hand_points", None)
                            hand_angle_r = data_r.pop("_v6_hand_angle", float("nan"))
                            snap_r = data_r.pop("_v7_keypoints", None)
                            if snap_r is not None:
                                keypoint_records.append({"樣本編號": i+1, "時間(秒)": sec, **snap_r})
                            rec_r = {"樣本編號": i + 1, "時間(秒)": sec, "評估側": "右手側", "右側信賴度(%)": round(conf_r, 1)}
                            rec_r.update(data_r)
                            right_only_records.append(rec_r)

                            img_r = put_chinese_text(img_r, f"時間: {sec}s | 獨立擷取: 右手側", (15, 15), (0, 255, 255), 20)
                            img_r = put_chinese_text(img_r, f"右手 Grand Score: {data_r['右手 Grand Score']} 分 ({data_r['右手 AL']})", (15, 45), (0, 0, 255) if data_r['右手 Grand Score'] > 4 else (0, 255, 0), 20)
                            img_r = put_chinese_text(img_r, f"外展:{data_r['右手上臂外展補正']} | 交叉:{data_r['右手橫越中線補正']}", (15, 75), (255, 180, 0), 16)

                            hand_img_r = draw_hand_overlay(img_r, hand_points_r, hand_angle_r)
                            _save_hand_photo("右手側", i + 1, sec, _encode_preview(img_r), _encode_preview(hand_img_r), zip_hand_imgs)
                            oh, ow = img_r.shape[:2]
                            out_scale = min(720 / max(oh, ow), 1.0)
                            if out_scale < 1.0:
                                img_r = cv2.resize(img_r, (int(ow * out_scale), int(oh * out_scale)))
                            ok_r, buf_r = cv2.imencode(".jpg", img_r, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                            if ok_r:
                                b_r = buf_r.tobytes()
                                zip_right_imgs.append((f"Sample_{i+1:03d}_{sec}s_Right.jpg", b_r))
                                cap_str_r = f"右手樣本 #{i+1} ({sec}s) | 總分: {data_r['右手 Grand Score']} ({data_r['右手 AL']}) | 手腕: {data_r['右手手腕角度']}°"
                                display_right_b64.append((cap_str_r, base64.b64encode(b_r).decode("utf-8")))

                        if succ_l and data_l is not None:
                            hand_points_l = data_l.pop("_v6_hand_points", None)
                            hand_angle_l = data_l.pop("_v6_hand_angle", float("nan"))
                            snap_l = data_l.pop("_v7_keypoints", None)
                            if snap_l is not None:
                                keypoint_records.append({"樣本編號": i+1, "時間(秒)": sec, **snap_l})
                            rec_l = {"樣本編號": i + 1, "時間(秒)": sec, "評估側": "左手側", "左側信賴度(%)": round(conf_l, 1)}
                            rec_l.update(data_l)
                            left_only_records.append(rec_l)

                            img_l = put_chinese_text(img_l, f"時間: {sec}s | 獨立擷取: 左手側", (15, 15), (0, 255, 255), 20)
                            img_l = put_chinese_text(img_l, f"左手 Grand Score: {data_l['左手 Grand Score']} 分 ({data_l['左手 AL']})", (15, 45), (0, 0, 255) if data_l['左手 Grand Score'] > 4 else (0, 255, 0), 20)
                            img_l = put_chinese_text(img_l, f"外展:{data_l['左手上臂外展補正']} | 交叉:{data_l['左手橫越中線補正']}", (15, 75), (255, 180, 0), 16)

                            hand_img_l = draw_hand_overlay(img_l, hand_points_l, hand_angle_l)
                            _save_hand_photo("左手側", i + 1, sec, _encode_preview(img_l), _encode_preview(hand_img_l), zip_hand_imgs)
                            oh, ow = img_l.shape[:2]
                            out_scale = min(720 / max(oh, ow), 1.0)
                            if out_scale < 1.0:
                                img_l = cv2.resize(img_l, (int(ow * out_scale), int(oh * out_scale)))
                            ok_l, buf_l = cv2.imencode(".jpg", img_l, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                            if ok_l:
                                b_l = buf_l.tobytes()
                                zip_left_imgs.append((f"Sample_{i+1:03d}_{sec}s_Left.jpg", b_l))
                                cap_str_l = f"左手樣本 #{i+1} ({sec}s) | 總分: {data_l['左手 Grand Score']} ({data_l['左手 AL']}) | 手腕: {data_l['左手手腕角度']}°"
                                display_left_b64.append((cap_str_l, base64.b64encode(b_l).decode("utf-8")))

                        # 若該秒左右手皆成功獨立解算，建立雙側對照總表紀錄
                        if succ_r and succ_l and data_r is not None and data_l is not None:
                            r_g = data_r["右手 Grand Score"]
                            l_g = data_l["左手 Grand Score"]
                            worst_side = "雙側同分" if r_g == l_g else ("右手側" if r_g > l_g else "左手側")
                            final_grand = max(r_g, l_g)
                            final_al = data_r["右手 AL"] if r_g >= l_g else data_l["左手 AL"]
                            overall_conf = round((conf_r + conf_l) / 2.0, 1)
                            neck_fused,neck_gap,neck_reason = v8_neck_cross_view(data_r,data_l)

                            comb_row = {
                                "樣本編號": i + 1,
                                "時間(秒)": sec,
                                "頸部角度": round((data_r["頸部角度"] + data_l["頸部角度"]) / 2.0, 1),
                                "頸部_v7_舊版平均(僅展示)": round((data_r["頸部角度"]+data_l["頸部角度"])/2,1),
                                "右側頸部原角度": data_r["頸部角度"],
                                "左側頸部原角度": data_l["頸部角度"],
                                "頸部_v8_右側候選": data_r["頸部_v8_候選角度"],
                                "頸部_v8_左側候選": data_l["頸部_v8_候選角度"],
                                "頸部_v8_右側品質": data_r["頸部_v8_候選品質"],
                                "頸部_v8_左側品質": data_l["頸部_v8_候選品質"],
                                "頸部_v8_候選差值": neck_gap,
                                "頸部_v8_候選融合角": neck_fused,
                                "頸部_v8_融合說明": neck_reason,
                                "軀幹角度": round((data_r["軀幹角度"] + data_l["軀幹角度"]) / 2.0, 1),
                                "右上臂角度": data_r["右手上臂角度"],
                                "右前臂角度": data_r["右手前臂角度"],
                                "右手腕角度": data_r["右手手腕角度"],
                                "右Pose舊版手腕角度": data_r["右手Pose舊版手腕角度"],
                                "右PoseHand實驗手腕角度": data_r["右手PoseHand實驗手腕角度"],
                                "右Hand偵測耗時(ms)": data_r["右手Hand偵測耗時(ms)"],
                                "右手腕扭轉": data_r["右手手腕扭轉"],
                                "左上臂角度": data_l["左手上臂角度"],
                                "左前臂角度": data_l["左手前臂角度"],
                                "左手腕角度": data_l["左手手腕角度"],
                                "左Pose舊版手腕角度": data_l["左手Pose舊版手腕角度"],
                                "左PoseHand實驗手腕角度": data_l["左手PoseHand實驗手腕角度"],
                                "左Hand偵測耗時(ms)": data_l["左手Hand偵測耗時(ms)"],
                                "左手腕扭轉": data_l["左手手腕扭轉"],
                                "右手 Score A": data_r["右手 Score A"],
                                "右手 Score D": data_r["頸軀腿 Score D"],
                                "右手 Score C": data_r["右手 Score C"],
                                "右手 Grand Score": r_g,
                                "右手 AL": data_r["右手 AL"],
                                "左手 Score A": data_l["左手 Score A"],
                                "左手 Score D": data_l["頸軀腿 Score D"],
                                "左手 Score C": data_l["左手 Score C"],
                                "左手 Grand Score": l_g,
                                "左手 AL": data_l["左手 AL"],
                                "最危害側": worst_side,
                                "最終最高分": final_grand,
                                "最終 AL": final_al,
                                "Overall_Conf": overall_conf
                            }
                            combined_records.append(comb_row)
                    else:
                        skipped_right += 1
                        skipped_left += 1
                        skip_diagnostics.extend([{"樣本編號": i+1, "時間(秒)": sec, "評估側": side, "可能原因": "影片影格讀取失敗"} for side in ("右手側", "左手側")])
                        for side in ("右手側","左手側"):
                            v8_add_rejected_sample(rejected_rows,rejected_photos,i+1,sec,side,
                                "影片影格讀取失敗",{"Front":f_frame if s_f else None,
                                "Right":r_frame if s_r else None,"Left":l_frame if s_l else None},pose)

                    progress_bar.progress((i + 1) / total_samples)
                    status_text.text(f"左右側獨立擷取與姿態解算進度: {i+1} / {total_samples} 筆完成")

            for c in [cap_front, cap_right, cap_left]: c.release()
            for p in [t_front.name, t_right.name, t_left.name]:
                try: os.remove(p)
                except Exception: pass

            v9_decorate_records(combined_records,sync_samples)
            v9_decorate_records(right_only_records,sync_samples)
            v9_decorate_records(left_only_records,sync_samples)
            v9_decorate_records(skip_diagnostics,sync_samples)
            v9_decorate_records(rejected_rows,sync_samples)
            df_comb = pd.DataFrame(combined_records) if combined_records else pd.DataFrame()
            df_r = v92_pose_wrist_qc(add_hand_temporal_notes(pd.DataFrame(right_only_records), "右手側"), "右手側") if right_only_records else pd.DataFrame()
            df_l = v92_pose_wrist_qc(add_hand_temporal_notes(pd.DataFrame(left_only_records), "左手側"), "左手側") if left_only_records else pd.DataFrame()

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('Synchronization/Synchronization_Report.csv',sync_report.to_csv(index=False).encode('utf-8-sig'))
                zf.writestr('Synchronization/Frame_Time_Map.csv',pd.DataFrame(sync_samples.values()).to_csv(index=False).encode('utf-8-sig'))
                zf.writestr('Synchronization/Settings.json',json.dumps({'mode':sync_mode,'anchors':sync_anchors,
                    'common_start':sync_start,'common_end':sync_end,'cameras':active_cams,
                    'time_display':'analysis_elapsed_from_zero','source_timestamp_mapping':'Frame_Time_Map.csv'},
                    ensure_ascii=False,indent=2).encode('utf-8'))
                if not df_comb.empty:
                    zf.writestr("RULA_左右同步彙整報表.csv", df_comb.to_csv(index=False).encode('utf-8-sig'))
                    cols=[c for c in df_comb.columns if c.startswith('頸部_v8_') or c in
                          ('樣本編號','時間(秒)','右側頸部原角度','左側頸部原角度')]
                    zf.writestr('Diagnostics/Neck_Comparison.csv',df_comb[cols].to_csv(index=False).encode('utf-8-sig'))
                if not df_r.empty:
                    zf.writestr("RULA_右手側獨立報表.csv", df_r.to_csv(index=False).encode('utf-8-sig'))
                if not df_l.empty:
                    zf.writestr("RULA_左手側獨立報表.csv", df_l.to_csv(index=False).encode('utf-8-sig'))
                zf.writestr("Diagnostics/Skipped_Samples.csv", pd.DataFrame(skip_diagnostics if skip_diagnostics else [{"樣本編號":"","時間(秒)":"","評估側":"","可能原因":""}]).to_csv(index=False).encode("utf-8-sig"))
                zf.writestr("Diagnostics/Rejected_Samples_Detail.csv",pd.DataFrame(rejected_rows).to_csv(index=False).encode('utf-8-sig'))
                v8_zip_rejected_photos(zf,rejected_photos)
                for fname, fbytes in zip_right_imgs:
                    zf.writestr(f"右側鏡頭擷取照片/{fname}", fbytes)
                for fname, fbytes in zip_left_imgs:
                    zf.writestr(f"左側鏡頭擷取照片/{fname}", fbytes)
                for fname, fbytes in zip_hand_imgs:
                    zf.writestr(fname, fbytes)
                for fname, fbytes in zip_raw_imgs:
                    zf.writestr(fname, fbytes)
                if keypoint_records:
                    zf.writestr("Diagnostics/Keypoint_Coordinates.jsonl",
                        "\n".join(json.dumps(clean_json(v), ensure_ascii=False, allow_nan=False)
                                  for v in keypoint_records).encode("utf-8"))
                zf.writestr("Diagnostics/V7_Angle_Definitions.txt", ANGLE_DEFINITIONS.encode("utf-8"))
                if not df_r.empty or not df_l.empty:
                    comparison = pd.concat([df.assign(評估側=label) for label, df in (("右手側", df_r), ("左手側", df_l)) if not df.empty], ignore_index=True)
                    zf.writestr("Hand_Comparison/Comparison_Data.csv", comparison.to_csv(index=False).encode("utf-8-sig"))

            fixed_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.session_state.analysis_done = True
            st.session_state.df_records = df_comb
            st.session_state.df_right_only = df_r
            st.session_state.df_left_only = df_l
            st.session_state.display_right_b64 = display_right_b64
            st.session_state.display_left_b64 = display_left_b64
            st.session_state.zip_bytes = zip_buffer.getvalue()
            st.session_state.v8_rejected_rows = rejected_rows
            st.session_state.v8_rejected_photos = rejected_photos
            st.session_state.zip_filename = f"RULA_DualSide_Separate_Report_{fixed_time_str}.zip"
            st.session_state.analyzed_mode = "左右側同步計算"
            st.session_state.skipped_info = {"expected": total_samples, "right": skipped_right, "left": skipped_left}
            st.session_state.v9_sync_summary=sync_report
            gc.collect()

        # ==================== 【模式 B：單側評估 (右手側 或 左手側)】 ====================
        else:
            target_side = eval_mode
            selected_side_vid = vid_right_file if target_side == "右手側" else vid_left_file

            t_front = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_front.write(vid_front_file.getvalue()); t_front.close()
            
            t_side = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_side.write(selected_side_vid.getvalue()); t_side.close()

            cap_front = cv2.VideoCapture(t_front.name)
            cap_side = cv2.VideoCapture(t_side.name)

            fps_front = float(cap_front.get(cv2.CAP_PROP_FPS) or 30.0)
            fps_side = float(cap_side.get(cv2.CAP_PROP_FPS) or 30.0)
            count_front = int(cap_front.get(cv2.CAP_PROP_FRAME_COUNT))
            count_side = int(cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_front = fps_front if np.isfinite(fps_front) and fps_front > 0 else 30.0
            fps_side = fps_side if np.isfinite(fps_side) and fps_side > 0 else 30.0
            side_camera='Right' if target_side=='右手側' else 'Left'
            sync_meta={'Front':v9_frame_meta(fps_front,count_front),
                       side_camera:v9_frame_meta(fps_side,count_side)}
            try:
                sample_times,sync_start,sync_end=v9_make_sample_times(
                    num_samples,sync_meta,sync_anchors,active_cams,
                    clip_range=st.session_state.v8_values.get('v92_analysis_window'))
            except ValueError as exc:
                cap_front.release();cap_side.release()
                for p in [t_front.name,t_side.name]:
                    try:os.remove(p)
                    except OSError:pass
                st.error('影片無法建立同步區間：'+str(exc))
                st.stop()
            sync_report=v9_sync_report(sync_meta,sync_anchors,active_cams,sync_start,sync_end,sync_mode)
            sync_samples={i+1:v9_sample_row(t,sync_meta,sync_anchors,active_cams,i+1,analysis_start=sync_start)
                          for i,t in enumerate(sample_times)}

            records = []
            zip_images = []
            zip_hand_imgs = []
            zip_raw_imgs = []
            keypoint_records = []
            display_images_b64 = []
            skipped_count = 0
            skip_diagnostics = []
            rejected_rows = []
            rejected_photos = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            with mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=POSE_MODEL_COMPLEXITY,
                min_detection_confidence=0.50
            ) as pose:
                total_samples = len(sample_times)
                for i, sample_time in enumerate(sample_times):
                    sec = round(float(sample_time - sync_start), 2)
                    idx_front = sync_samples[i+1]['Front影格索引']
                    idx_side = sync_samples[i+1][side_camera+'影格索引']
                    cap_front.set(cv2.CAP_PROP_POS_FRAMES, idx_front)
                    cap_side.set(cv2.CAP_PROP_POS_FRAMES, idx_side)
                    s_f, f_frame = cap_front.read()
                    s_s, s_frame = cap_side.read()

                    if s_f and s_s:
                        if raw_image_export:
                            for label, raw in (("Front", f_frame), ("Right" if target_side == "右手側" else "Left", s_frame)):
                                payload = _encode_preview(raw)
                                if payload:
                                    zip_raw_imgs.append((f"Raw_Frames/{label}/Sample_{i+1:03d}_{sec}s.jpg", payload))
                        front_ok, front_mods = analyze_front_camera(f_frame, pose, target_side)
                        if not front_ok:
                            front_mods = manual_front_fallback()
                        succ_s, data, side_annotated, conf = analyze_side_camera(
                            s_frame, pose, target_side, front_mods, m_val, f_val,
                            manual_mods=build_manual_mods(target_side),
                            hand_task=hand_task, hand_scoring_policy=v6_hand_policy
                        )
                        if not succ_s or data is None:
                            skipped_count += 1
                            skip_diagnostics.append({"樣本編號": i+1, "時間(秒)": sec, "評估側": target_side, "可能原因": "正面關鍵點不足" if not front_ok else "側面關鍵點不足或角度無效"})
                            v8_add_rejected_sample(rejected_rows,rejected_photos,i+1,sec,target_side,
                                "正面關鍵點不足" if not front_ok else "側面關鍵點不足或角度無效",
                                {"Front":f_frame,"Right" if target_side=="右手側" else "Left":s_frame},pose)

                        if succ_s and data is not None:
                            hand_points_s = data.pop("_v6_hand_points", None)
                            hand_angle_s = data.pop("_v6_hand_angle", float("nan"))
                            snap_s = data.pop("_v7_keypoints", None)
                            if snap_s is not None:
                                keypoint_records.append({"樣本編號": i+1, "時間(秒)": sec, **snap_s})
                            record = {
                                "樣本編號": i + 1,
                                "時間(秒)": sec,
                                "主要評估側": target_side,
                                "視角信賴度(%)": round(conf, 1)
                            }
                            record.update(data)
                            records.append(record)

                            side_annotated = put_chinese_text(side_annotated, f"時間: {sec}s | 評估側: {target_side}", (15, 15), (0, 255, 255), 20)
                            side_annotated = put_chinese_text(side_annotated, f"Grand Score: {data['最終最高分']} 分 ({data['最終 AL']})", (15, 45), (0, 0, 255) if data['最終最高分'] > 4 else (0, 255, 0), 20)
                            side_annotated = put_chinese_text(side_annotated, f"外展:{data[f'{target_side[:2]}上臂外展補正']} | 交叉:{data[f'{target_side[:2]}橫越中線補正']}", (15, 75), (255, 180, 0), 16)

                            hand_img_s = draw_hand_overlay(side_annotated, hand_points_s, hand_angle_s)
                            _save_hand_photo(target_side, i + 1, sec, _encode_preview(side_annotated), _encode_preview(hand_img_s), zip_hand_imgs)
                            oh, ow = side_annotated.shape[:2]
                            out_scale = min(720 / max(oh, ow), 1.0)
                            if out_scale < 1.0:
                                side_annotated = cv2.resize(side_annotated, (int(ow * out_scale), int(oh * out_scale)))

                            is_ok, buf = cv2.imencode(".jpg", side_annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                            if is_ok:
                                jpg_bytes = buf.tobytes()
                                side_en = "Right" if target_side == "右手側" else "Left"
                                zip_images.append((f"Sample_{i+1:03d}_{sec}s_{side_en}.jpg", jpg_bytes))
                                
                                b64_img = base64.b64encode(jpg_bytes).decode("utf-8")
                                caption_str = f"樣本 #{i+1} ({sec}s) | 總分: {data['最終最高分']} ({data['最終 AL']}) | 手腕: {data[f'{target_side[:2]}手腕角度']}°"
                                display_images_b64.append((caption_str, b64_img))
                    else:
                        skipped_count += 1
                        skip_diagnostics.append({"樣本編號": i+1, "時間(秒)": sec, "評估側": target_side, "可能原因": "影片影格讀取失敗"})
                        v8_add_rejected_sample(rejected_rows,rejected_photos,i+1,sec,target_side,
                            "影片影格讀取失敗",{"Front":f_frame if s_f else None,
                             "Right" if target_side=="右手側" else "Left":s_frame if s_s else None},pose)

                    progress_bar.progress((i + 1) / total_samples)
                    status_text.text(f"全關節姿態解算進度: {i+1} / {total_samples} 筆完成")

            cap_front.release(); cap_side.release()
            try:
                os.remove(t_front.name); os.remove(t_side.name)
            except Exception:
                pass

            v9_decorate_records(records,sync_samples)
            v9_decorate_records(skip_diagnostics,sync_samples)
            v9_decorate_records(rejected_rows,sync_samples)
            df_result = v92_pose_wrist_qc(add_hand_temporal_notes(pd.DataFrame(records), target_side), target_side) if records else pd.DataFrame()
            
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('Synchronization/Synchronization_Report.csv',sync_report.to_csv(index=False).encode('utf-8-sig'))
                zf.writestr('Synchronization/Frame_Time_Map.csv',pd.DataFrame(sync_samples.values()).to_csv(index=False).encode('utf-8-sig'))
                zf.writestr('Synchronization/Settings.json',json.dumps({'mode':sync_mode,'anchors':sync_anchors,
                    'common_start':sync_start,'common_end':sync_end,'cameras':active_cams,
                    'time_display':'analysis_elapsed_from_zero','source_timestamp_mapping':'Frame_Time_Map.csv'},
                    ensure_ascii=False,indent=2).encode('utf-8'))
                if not df_result.empty:
                    zf.writestr("RULA_Report.csv", df_result.to_csv(index=False).encode('utf-8-sig'))
                zf.writestr("Diagnostics/Skipped_Samples.csv", pd.DataFrame(skip_diagnostics if skip_diagnostics else [{"樣本編號":"","時間(秒)":"","評估側":"","可能原因":""}]).to_csv(index=False).encode("utf-8-sig"))
                zf.writestr("Diagnostics/Rejected_Samples_Detail.csv",pd.DataFrame(rejected_rows).to_csv(index=False).encode('utf-8-sig'))
                v8_zip_rejected_photos(zf,rejected_photos)
                for fname, fbytes in zip_images:
                    zf.writestr(f"Photos/{fname}", fbytes)
                for fname, fbytes in zip_hand_imgs:
                    zf.writestr(fname, fbytes)
                for fname, fbytes in zip_raw_imgs:
                    zf.writestr(fname, fbytes)
                if keypoint_records:
                    zf.writestr("Diagnostics/Keypoint_Coordinates.jsonl",
                        "\n".join(json.dumps(clean_json(v), ensure_ascii=False, allow_nan=False)
                                  for v in keypoint_records).encode("utf-8"))
                zf.writestr("Diagnostics/V7_Angle_Definitions.txt", ANGLE_DEFINITIONS.encode("utf-8"))
                if not df_result.empty:
                    zf.writestr("Hand_Comparison/Comparison_Data.csv", df_result.to_csv(index=False).encode("utf-8-sig"))

            side_en_tag = "Right" if target_side == "右手側" else "Left"
            fixed_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            st.session_state.analysis_done = True
            st.session_state.df_records = df_result
            st.session_state.display_images_b64 = display_images_b64
            st.session_state.zip_bytes = zip_buffer.getvalue()
            st.session_state.v8_rejected_rows = rejected_rows
            st.session_state.v8_rejected_photos = rejected_photos
            st.session_state.zip_filename = f"RULA_{side_en_tag}_Report_{fixed_time_str}.zip"
            st.session_state.analyzed_mode = target_side
            st.session_state.skipped_info = {"expected": total_samples, "single": skipped_count}
            st.session_state.v9_sync_summary=sync_report
            gc.collect()

    if hand_task is not None:
        try:
            hand_task.close()
        except Exception:
            logging.exception("Hand model close failed")
    st.session_state.v9_screen='results'
    st.rerun()



# ==================== 【V4：時間異常檢測／逐影格人工複核】 ====================
# 警示閾值是工程篩檢參數，不是已驗證的人體角度準確性標準。
# V4.1: 保守異常篩檢。正常的姿勢轉換不再因相鄰角度差大而被列入異常。
# 所有數值均為工程篩檢門檻，尚未經人工標準影像驗證。
V41_WRIST_EXTREME_DEG = 60.0
V41_ELBOW_EXTREME_DEG = 170.0
V41_ISOLATED_SPIKE_DEG = 85.0
V41_NEIGHBOR_SIMILAR_DEG = 25.0


def v4_mark_anomalies(df, side):
    """極保守的嚴重異常清單：極端腕/肘角度 + 極明顯孤立尖峰。

    單次真實動作變化、腕 2D/3D 差異、或原始資料的寬鬆複核旗標
    仍保留在 CSV 供研究，但不單獨觸發使用者警示。
    """
    if df is None or df.empty:
        return pd.DataFrame()
    data = df.copy().sort_values("時間(秒)").reset_index(drop=True)
    prefix = "右手" if side == "右手側" else "左手"
    checks = [("頸部角度", "頸部"), ("軀幹角度", "軀幹"),
              (f"{prefix}上臂角度", "上臂"), (f"{prefix}前臂角度", "前臂"),
              (f"{prefix}手腕角度", "手腕")]
    reasons = [[] for _ in range(len(data))]
    for col, label in checks:
        if col not in data.columns:
            continue
        vals = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
        for i, value in enumerate(vals):
            if not np.isfinite(value):
                reasons[i].append(f"{label}角度無效，無法計分")
                continue
            if label == "手腕" and value > V41_WRIST_EXTREME_DEG:
                reasons[i].append(f"手腕極端角度 {value:.1f}°（建議確認遮擋）")
            if label == "前臂" and value >= V41_ELBOW_EXTREME_DEG:
                reasons[i].append(f"前臂屈曲角度極端 {value:.1f}°（建議檢查關節定位）")
            # 僅標記前後影格彼此接近、中央影格卻突然大幅偏離的孤立尖峰。
            # 這能排除大部分真實的連續彎腰、起立、舉手等運動。
            if 0 < i < len(vals) - 1:
                prev, nxt = vals[i - 1], vals[i + 1]
                if (np.isfinite(prev) and np.isfinite(nxt)
                        and abs(prev - nxt) <= V41_NEIGHBOR_SIMILAR_DEG
                        and abs(value - prev) >= V41_ISOLATED_SPIKE_DEG
                        and abs(value - nxt) >= V41_ISOLATED_SPIKE_DEG):
                    reasons[i].append(f"{label}孤立尖峰 {value:.1f}°（前後值 {prev:.1f}°／{nxt:.1f}°）")
    data["自動複核警示"] = ["；".join(dict.fromkeys(rs)) for rs in reasons]
    data["需要複核"] = data["自動複核警示"].str.len() > 0
    data["警示等級"] = np.where(data["需要複核"], "嚴重待確認", "無嚴重警示")
    data["評估側"] = side
    return data



def v4_recalculate_from_scores(row, updates, side):
    """使用人工指定的各部位 RULA 分數重新查表；不更動原始角度。"""
    prefix = "右手" if side == "右手側" else "左手"
    vals = {
        "ua": int(updates["上臂"]), "la": int(updates["前臂"]),
        "w": int(updates["手腕"]), "wt": int(updates["腕扭轉"]),
        "nk": int(updates["頸部"]), "tk": int(updates["軀幹"]),
        "lg": int(updates["腿部"]), "muscle": int(updates["肌肉"]),
        "force": int(updates["負重"]),
    }
    a, b, c, d, total, al = compute_rula_full(vals["ua"], vals["la"], vals["w"], vals["wt"],
                                                  vals["nk"], vals["tk"], vals["lg"], vals["muscle"], vals["force"])
    return {"樣本編號": int(row["樣本編號"]), "時間(秒)": float(row["時間(秒)"]), "評估側": side,
            "原始Grand Score": int(row[f"{prefix} Grand Score"]), "人工Grand Score": total,
            "原始AL": row[f"{prefix} AL"], "人工AL": al,
            "人工Score A": a, "人工Score B": b, "人工Score C": c, "人工Score D": d,
            **{f"人工{key}分數": v for key, v in updates.items()},
            "異動說明": "逐影格分數覆核（未修改原始骨架角度）"}


def v4_make_zip(original_zip, reviews, anomaly_frames):
    """將原始報告與人工覆核結果一起下載，保留原始資料供比對。"""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original_zip), "r") as original, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in original.infolist():
            dst.writestr(info, original.read(info.filename))
        dst.writestr("Diagnostics/V4_Anomaly_Review.csv", anomaly_frames.to_csv(index=False).encode("utf-8-sig"))
        dst.writestr("Diagnostics/V4_Manual_Overrides.csv", pd.DataFrame(list(reviews.values())).to_csv(index=False).encode("utf-8-sig"))
        dst.writestr("Diagnostics/V4_README.txt", "V4.1 使用保守篩檢：相鄰正常運動不算異常；只警示極端手腕/前臂角度或孤立尖峰。警示不等於辨識錯誤；沒有警示也不等於正確。人工修正修改的是逐影格 RULA 分數，原始角度、原始影像與原始 CSV 均未覆蓋。請另行使用標準工作表核對 RULA 查表內容。".encode("utf-8"))
    return out.getvalue()


def v4_show_review_center():
    st.subheader("異常影格審核／逐張人工修正／風險時間軸")
    mode = st.session_state.analyzed_mode
    both = mode == "左右側同步計算"
    available = {}
    if both:
        available = {"右手側": st.session_state.df_right_only, "左手側": st.session_state.df_left_only}
    else:
        available = {mode: st.session_state.df_records}
    available = {k: v for k, v in available.items() if v is not None and not v.empty}
    if not available:
        st.info("目前沒有可供複核的有效樣本。")
        return

    all_checked = pd.concat([v4_mark_anomalies(df, side) for side, df in available.items()], ignore_index=True)
    tool = st.radio('目前審核功能', ['🚩 異常影格', '✍️ 人工重新計分', '📈 覆核時間軸'],
                    horizontal=True, key='v8_review_tool')
    if tool == '🚩 異常影格':
        flagged = all_checked[all_checked["需要複核"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("有效評估筆數", len(all_checked))
        c2.metric("待複核筆數", len(flagged))
        c3.metric("警示占比", f"{len(flagged) / len(all_checked):.1%}")
        st.caption("V4.1 採保守篩檢：只提示極端腕/肘角度、或前後相近但中央極端偏離的孤立尖峰。正常連續動作不因角度變化快而警示；原始 2D/3D 差異仍保留在 CSV。")
        if flagged.empty:
            st.success("沒有觸發目前的工程警示閾值；仍建議抽樣人工核對。")
        else:
            st.dataframe(flagged[["評估側", "樣本編號", "時間(秒)", "最終最高分", "自動複核警示"]], use_container_width=True, hide_index=True)
        side = st.selectbox("選擇要查看照片的評估側", list(available), key="v4_review_side")
        side_data = v4_mark_anomalies(available[side], side)
        only_flagged = st.checkbox("只列出有警示的影格", value=True, key="v4_only_flags")
        candidates = side_data[side_data["需要複核"]] if only_flagged else side_data
        if len(candidates):
            ids = candidates["樣本編號"].astype(int).tolist()
            chosen = st.selectbox("樣本編號", ids, key="v4_photo_id")
            found = side_data[side_data["樣本編號"] == chosen].iloc[0]
            st.write(f"{found['時間(秒)']} 秒 ｜ 原始 RULA {found['最終最高分']} ｜ {found['自動複核警示'] or '無警示'}")
            images = (st.session_state.display_right_b64 if side == "右手側" else st.session_state.display_left_b64) if both else st.session_state.display_images_b64
            import re
            photo_found = False
            for caption, payload in images:
                match = re.search(r"#(\d+)", caption)
                if match and int(match.group(1)) == int(chosen):
                    # /media 圖片路徑在部分 Streamlit Cloud/瀏覽器環境無法存取；
                    # 直接在 iframe 用 data URI 內嵌，不向 /media 發請求。
                    import html as html_lib
                    v92_show_image(payload, caption, encoded=True)
                    photo_found = True
                    break
            if not photo_found:
                st.warning("此樣本沒有可用的預覽照片；請至原始 ZIP 的 Photos 資料夾檢查。")
        else:
            st.info("此評估側沒有符合條件的影格，可取消勾選警示篩選。")

    if tool == '✍️ 人工重新計分':
        st.caption("這是可選的人工覆核工具，不影響原始自動分析。僅修改指定樣本的各部位分數，重新查表並保留原始資料。")
        side = st.selectbox("評估側", list(available), key="v4_edit_side")
        df = available[side].sort_values("樣本編號")
        ids = df["樣本編號"].astype(int).tolist()
        sample = st.selectbox("選擇欲修正的樣本", ids, key="v4_edit_sample")
        row = df[df["樣本編號"].astype(int) == sample].iloc[0]
        prefix = "右手" if side == "右手側" else "左手"
        defaults = {"上臂": (f"{prefix}上臂分數", 1, 6), "前臂": (f"{prefix}前臂分數", 1, 3),
                    "手腕": (f"{prefix}手腕分數", 1, 4), "腕扭轉": (f"{prefix}手腕扭轉分數", 1, 2),
                    "頸部": ("頸部分數", 1, 6), "軀幹": ("軀幹分數", 1, 6),
                    "腿部": ("腿部分數", 1, 2), "肌肉": ("肌肉使用加分", 0, 1),
                    "負重": ("負重施力加分", 0, 3)}
        key = (side, int(sample))
        previous = st.session_state.v4_overrides.get(key, {})
        with st.form(f"v4_override_{side}_{sample}"):
            fields = st.columns(3)
            values = {}
            for i, (label, (col, low, high)) in enumerate(defaults.items()):
                value = int(previous.get(f"人工{label}分數", row[col]))
                values[label] = fields[i % 3].number_input(f"{label}分數", min_value=low, max_value=high, value=max(low, min(high, value)), step=1)
            if st.form_submit_button("重新計分並儲存本影格覆核結果", type="primary"):
                result = v4_recalculate_from_scores(row, values, side)
                st.session_state.v4_overrides[key] = result
                st.success(f"已儲存：原始 {result['原始Grand Score']} 分 → 複核後 {result['人工Grand Score']} 分（{result['人工AL']}）")
        if key in st.session_state.v4_overrides:
            result = st.session_state.v4_overrides[key]
            st.metric("本影格複核後分數", result["人工Grand Score"], delta=result["人工Grand Score"] - result["原始Grand Score"])
            if st.button("撤銷這一筆人工修正", key=f"v4_undo_{side}_{sample}"):
                del st.session_state.v4_overrides[key]
                st.rerun()
        st.subheader("已保存的逐影格修正")
        if st.session_state.v4_overrides:
            st.dataframe(pd.DataFrame(st.session_state.v4_overrides.values()), use_container_width=True, hide_index=True)
        else:
            st.info("目前沒有人工修正，原始分數維持不變。")

    if tool == '📈 覆核時間軸':
        st.caption("以每個影格的原始 RULA Grand Score 繪製；如有人工修正，另以修正後分數作比較。時間間距依影片取樣而定。")
        side = st.selectbox("時間軸評估側", list(available), key="v4_chart_side")
        series = available[side].sort_values("時間(秒)").copy()
        prefix = "右手" if side == "右手側" else "左手"
        series["原始RULA"] = pd.to_numeric(series[f"{prefix} Grand Score"], errors="coerce")
        series["修正後RULA"] = series["原始RULA"].copy()
        for idx, row in series.iterrows():
            edit = st.session_state.v4_overrides.get((side, int(row["樣本編號"])))
            if edit:
                series.loc[idx, "修正後RULA"] = edit["人工Grand Score"]
        chart = series[["時間(秒)", "原始RULA", "修正後RULA"]].set_index("時間(秒)")
        st.line_chart(chart, y_label="RULA 分數（1～7）", x_label="影片時間（秒）")
        st.dataframe(series[["樣本編號", "時間(秒)", "原始RULA", "修正後RULA"]], use_container_width=True, hide_index=True)

    st.subheader("📦 下載含異常清單與人工覆核紀錄的完整 ZIP")
    enhanced = v4_make_zip(st.session_state.zip_bytes, st.session_state.v4_overrides, all_checked)
    st.caption("使用瀏覽器本地 Blob 下載，避開部分 Streamlit Cloud /media 下載路徑失效的問題。")
    render_blob_download_button(enhanced, "RULA_V7_Review_Report.zip", "📥 下載審核 ZIP（原始資料＋人工修正紀錄）")


# ==================== 【V6：Pose / Pose+Hand 手腕演算法比較中心】 ====================
def v6_show_hand_comparison():
    st.subheader("手腕演算法比較：Pose 2D／3D／混合 vs Hand 2D")
    st.caption("公平對照：Pose 2D vs Hand 2D。原版 Pose 混合角度與 Pose 3D 額外保留；Hand World 為手掌局部資訊，不與 Pose World 混用。時序尖峰僅作診斷，不會自動判錯或改分。")
    st.info("模型狀態：" + str(st.session_state.get("hand_model_status", "未知")))
    both = st.session_state.analyzed_mode == "左右側同步計算"
    available = ({"右手側": st.session_state.df_right_only, "左手側": st.session_state.df_left_only}
                 if both else {st.session_state.analyzed_mode: st.session_state.df_records})
    available = {side: df for side, df in available.items() if df is not None and not df.empty}
    if not available:
        return
    side = st.selectbox("比較評估側", list(available), key="v6_review_side")
    df = available[side].copy().sort_values("樣本編號")
    prefix = "右手" if side == "右手側" else "左手"
    old_col = f"{prefix}Pose舊版手腕角度"
    new_col = f"{prefix}PoseHand實驗手腕角度"
    diff_col = f"{prefix}Pose2D與Hand2D差異"
    legacy_diff_col = f"{prefix}新舊手腕差異"
    pose2d_col = f"{prefix}手腕角度_2D"
    pose3d_col = f"{prefix}手腕角度_3D"
    quality_col = f"{prefix}Hand品質原因"
    time_col = f"{prefix}Hand偵測耗時(ms)"
    pose_ms_col = f"{prefix}側面Pose偵測耗時(ms)"
    if old_col not in df.columns:
        st.warning("本批資料尚無 Hand 比較欄位，請用 v6 重新執行影片分析。")
        return
    good = df[new_col].notna()
    ok_ratio = 100.0 * float(good.mean())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("成功匹配手部", f"{int(good.sum())}/{len(df)}")
    c2.metric("手部匹配率", f"{ok_ratio:.1f}%")
    c3.metric("平均絕對角度差", f"{df.loc[good, diff_col].mean():.1f}°" if good.any() else "無資料")
    c4.metric("側面 Pose 中位耗時", f"{df[pose_ms_col].median():.0f} ms" if df[pose_ms_col].notna().any() else "未執行")
    c5.metric("Hand 中位耗時", f"{df[time_col].median():.0f} ms" if df[time_col].notna().any() else "未執行")
    st.caption("耗時數值是在本次 Streamlit 執行環境實測，並非模型官方效能；Pose 欄只計側面 Pose 偵測（含重試），不包含前鏡頭偵測／骨架繪圖。")
    with st.expander('Hand 匹配失敗與品質問題統計（研究診斷）',expanded=False):
        reason_column=f"{prefix}Hand品質原因"
        matched_column=f"{prefix}Hand匹配成功"
        pass_column=f"{prefix}Hand品質通過"
        a1,a2,a3=st.columns(3)
        a1.metric('成功匹配',int(df[matched_column].fillna(False).sum()))
        a2.metric('品質通過',int(df[pass_column].fillna(False).sum()))
        a3.metric('待檢查',int((~df[pass_column].fillna(False).astype(bool)).sum()))
        if reason_column in df.columns:
            causes=df[reason_column].fillna('未記錄').astype(str).value_counts().rename_axis('品質說明').reset_index(name='筆數')
            st.dataframe(causes,use_container_width=True,hide_index=True)
        st.caption('這些是幾何品質判斷與模型匹配情況，不是 Hand 或 Pose 的量測準確率。')

    ids = df["樣本編號"].astype(int).tolist()
    chosen = st.selectbox("選取影格查看兩版結果（可選 9、25、42 等）", ids,
                          index=ids.index(9) if 9 in ids else 0, key="v6_hand_sample")
    row = df[df["樣本編號"].astype(int) == int(chosen)].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Pose 2D", f"{row[pose2d_col]:.1f}°" if pd.notna(row[pose2d_col]) else "—")
    b.metric("Pose 3D", f"{row[pose3d_col]:.1f}°" if pd.notna(row[pose3d_col]) else "—")
    c.metric("Pose 混合 v5", f"{row[old_col]:.1f}°" if pd.notna(row[old_col]) else "—")
    d.metric("Hand 2D", f"{row[new_col]:.1f}°" if pd.notna(row[new_col]) else "未辨識")
    st.write(f"**Pose2D／Hand2D 比較差：** {row[diff_col]:.1f}°" if pd.notna(row[diff_col]) else "Hand 無有效 2D 對照")
    st.write(f"樣本 #{int(chosen)}｜{row['時間(秒)']} 秒｜實際計分演算法：{row[f'{prefix}手腕計分演算法']}")
    st.caption(f"同條件情境比較：Pose 基準 RULA {row[f'{prefix}Pose基準GrandScore']}；若只將手腕屈伸改用 Hand，RULA {row[f'{prefix}Hand情境GrandScore']}（無 Hand 時為空值）。此為敏感度對照，不代表 Hand 方案更準確。")
    st.caption(f"匹配：{row[f'{prefix}Hand匹配成功']}；幾何品質篩檢：{row[f'{prefix}Hand品質通過']}；原因：{row[quality_col]}")
    st.caption(f"Hand World 手掌局部長度（模型單位）：{row[f'{prefix}HandWorld掌心長度(模型單位)']}；掌指張角：{row[f'{prefix}HandWorld掌指張角(度)']}°（僅供比較，不是手腕屈曲角度）")
    pair = st.session_state.get("hand_review_images", {}).get((side, int(chosen)))
    if pair:
        import html as html_lib
        before_b64, after_b64 = pair
        col_pose, col_hand = st.columns(2)
        with col_pose:
            st.markdown('**原始 Pose**')
            v92_show_image(before_b64, encoded=True)
        with col_hand:
            st.markdown('**Pose + Hand 21 點**')
            v92_show_image(after_b64, encoded=True)
    else:
        st.warning("找不到這個樣本的比較照片；請檢查 ZIP 的 Hand_Comparison 資料夾。")
    table_cols = ["樣本編號", "時間(秒)", pose2d_col, pose3d_col, old_col, new_col, diff_col, quality_col,
                  f"{prefix}Pose基準GrandScore", f"{prefix}Hand情境GrandScore",
                  f"{prefix}Hand匹配成功", f"{prefix}Hand品質通過", f"{prefix}Hand用於計分",
                  pose_ms_col, time_col]
    st.dataframe(df[table_cols], use_container_width=True, hide_index=True)
    graph = df[["時間(秒)", pose2d_col, old_col, new_col]].set_index("時間(秒)")
    st.line_chart(graph, x_label="影片時間（秒）", y_label="手腕角度代理值（度）")
    flagged_count = int(df['Pose手腕需複核（保守）'].fillna(False).sum()) if 'Pose手腕需複核（保守）' in df.columns else 0
    st.info(f'Pose 手腕保守品質診斷：{flagged_count} 筆可能存在遮擋或孤立尖峰。'
            '請檢視原始影格；此項只提醒，不會自動改動 RULA 分數。')
    st.caption("公平比較以 Pose 2D 與 Hand 2D 為主；Pose 混合保留供回歸測試。角度差異變小不等於真值更準。")

# ==================== 【V9.1：結果採按需載入的真正分頁】 ====================
# Streamlit >= 1.55 支援 on_change="rerun" 和 TabContainer.open。
# 避免顯示「風險總覽」時，就同時產生 ZIP、Base64 照片及其餘所有結果頁。
if st.session_state.v9_screen == 'results':
    if not st.session_state.analysis_done:
        st.warning('尚無分析結果，請返回設定並在最後一步啟動分析。')
    else:
        mode_now = st.session_state.analyzed_mode
        df_show = st.session_state.df_records
        st.subheader('📊 RULA 分析結果')
        st.caption('點選下方分頁直接檢視資料，不必再按下一步；如需修改分析設定，可從側邊欄返回。')
        (tab_overview, tab_review, tab_rejected, tab_angles,
         tab_gallery, tab_reports, tab_download) = st.tabs([
            '📈 風險總覽', '🔎 異常／人工覆核', '🧩 剔除樣本',
            '📐 頸部／手腕比較', '🖼️ 影格照片', '📋 完整報表', '📦 下載',
        ], key='v9_result_tabs', on_change='rerun')
        if tab_overview.open:
            with tab_overview:
                st.subheader('RULA 風險總覽')
                info=st.session_state.get('skipped_info',{})
                st.caption(f'評估模式：{mode_now}｜跳過影格仍保留在「被剔除樣本」步驟中。')
                if st.session_state.get('v9_sync_summary') is not None:
                    with st.expander('本次三鏡頭時間同步結果',expanded=False):
                        st.dataframe(st.session_state.v9_sync_summary,use_container_width=True,hide_index=True)
                if df_show is None or df_show.empty:
                    st.warning('沒有共同有效樣本。可在剔除影格頁檢查失敗原因，單側資料可能仍在完整報表中。')
                else:
                    scores=pd.to_numeric(df_show['最終最高分'],errors='coerce')
                    a,b,c,d=st.columns(4)
                    a.metric('有效樣本數',len(df_show))
                    b.metric('平均 Grand Score',f'{scores.mean():.2f}')
                    c.metric('最高分',int(scores.max()))
                    d.metric('AL3／AL4 比例',f'{(scores.ge(5).mean()*100):.1f}%')
                    series=df_show[['時間(秒)','最終最高分']].copy().sort_values('時間(秒)')
                    st.line_chart(series.set_index('時間(秒)'),x_label='時間（秒）',y_label='RULA Grand Score')
                    if '最危害側' in df_show.columns and mode_now=='左右側同步計算':
                        st.write('**最危害側統計（含雙側同分）**')
                        st.dataframe(df_show['最危害側'].value_counts().rename_axis('分類').reset_index(name='筆數'),hide_index=True)
                    st.caption('平均數僅用於描述性統計，正式 RULA 風險仍需關注最危險姿勢與工作條件。')
                st.info('先檢查時間同步結果，再看被剔除影格、頸部、手腕與人工修正。')
        if tab_review.open:
            with tab_review:
                v4_show_review_center()
        if tab_rejected.open:
            with tab_rejected:
                v8_show_rejected_center()
        if tab_angles.open:
            with tab_angles:
                neck_tab, wrist_tab = st.tabs(
                    ['🧠 頸部品質', '🤚 手腕演算法'],
                    key='v9_angle_tabs', on_change='rerun',
                )
                if neck_tab.open:
                    with neck_tab:
                        v8_show_neck_center()
                if wrist_tab.open:
                    with wrist_tab:
                        v6_show_hand_comparison()
        if tab_gallery.open:
            with tab_gallery:
                st.subheader('逐影格骨架照片')
                if mode_now=='左右側同步計算':
                    which=st.radio('選擇側別',['右手側','左手側'],horizontal=True,key='v8_gallery_side')
                    images=st.session_state.display_right_b64 if which=='右手側' else st.session_state.display_left_b64
                else:
                    images=st.session_state.display_images_b64
                if images:
                    idx=st.selectbox('選取影格',range(len(images)),format_func=lambda i:images[i][0],key='v8_gallery_select')
                    import html as html_lib
                    cap,payload=images[idx]
                    v92_show_image(payload, cap, encoded=True)
                    if st.checkbox('展開所有縮圖（耗用較多記憶體）',key='v8_all_photo_thumbs'):
                        cols=st.columns(3)
                        for i,(caption,payload) in enumerate(images):
                            with cols[i%3]:
                                v92_show_image(v92_thumbnail_b64(payload), caption, encoded=True)
                else:
                    st.info('沒有可顯示的有效樣本照片。剔除樣本請到「🧩 剔除樣本」分頁檢查。')
        if tab_reports.open:
            with tab_reports:
                st.subheader('完整報表與下載')
                if df_show is not None and not df_show.empty:
                    if mode_now=='左右側同步計算':
                        table_choice=st.radio('報表類型',['左右共同','右側完整','左側完整'],horizontal=True,key='v8_report_choice')
                        table={'左右共同':df_show,'右側完整':st.session_state.df_right_only,
                               '左側完整':st.session_state.df_left_only}[table_choice]
                        st.dataframe(table,use_container_width=True,hide_index=True)
                    else:
                        st.dataframe(df_show,use_container_width=True,hide_index=True)
                else:
                    st.warning('沒有可用的共同有效資料，但仍可下載診斷 ZIP。')
        if tab_download.open:
            with tab_download:
                st.subheader('匯出報表及診斷 ZIP')
                if st.session_state.zip_bytes:
                    initial_zip=st.session_state.zip_bytes
                    if st.session_state.v4_overrides:
                        if mode_now=='左右側同步計算':
                            source_dfs={'右手側':st.session_state.df_right_only,'左手側':st.session_state.df_left_only}
                        else:
                            source_dfs={mode_now:st.session_state.df_records}
                        anomaly=pd.concat([v4_mark_anomalies(data,side) for side,data in source_dfs.items()
                                      if data is not None and not data.empty],ignore_index=True)
                        initial_zip=v4_make_zip(initial_zip,st.session_state.v4_overrides,anomaly)
                    finalized=v8_download_report_zip(initial_zip,st.session_state.get('v8_manual_rejected',{}))
                    render_blob_download_button(finalized,st.session_state.zip_filename,
                                                f'📥 下載 {mode_now} 完整分析＋失敗影格 ZIP')
                st.caption('ZIP 包含原始 RULA 報表、被剔除樣本清單與雙照片、頸部候選比較及人工註記，不會覆蓋原始分數。')
# 只有分析前六步才顯示底部「上一步／下一步」；結果頁使用分頁，不再有步驟按鈕。
if st.session_state.v9_screen == 'setup':
    st.divider()
    bottom_a, bottom_b, bottom_c = st.columns([1,3,1])
    with bottom_a:
        st.button('← 上一步',disabled=st.session_state.v8_page_index==0,
                  on_click=v8_go,args=(-1,),use_container_width=True,key='v8_prev_bottom')
    with bottom_b:
        if st.session_state.v8_page_index == len(V8_STEPS)-1:
            st.caption('最後一步：按「啟動分析」後，會直接開啟可點選的結果分頁。')
        else:
            st.caption('切換步驟不會自動開始分析；完成設定後再啟動。')
    with bottom_c:
        if st.session_state.v8_page_index < len(V8_STEPS)-1:
            st.button('下一步 →',on_click=v8_go,args=(1,),
                      use_container_width=True,key='v8_next_bottom')
