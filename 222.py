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
# 舊版 Pose API 必須搭配支援 mp.solutions 的 MediaPipe；新版移除時明確提示。
if not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'pose'):
    st.set_page_config(page_title='RULA｜MediaPipe 版本不相容', layout='centered')
    st.error('目前安裝的 MediaPipe 已不支援 mp.solutions.pose。')
    st.info('請使用本專案 requirements.txt（mediapipe==0.10.21，建議 Python 3.11），不要直接升級到 0.10.30 以上；或另行將完整專案遷移至 Tasks Pose API。')
    st.stop()
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
from datetime import datetime, date
from PIL import Image, ImageDraw, ImageFont

# Streamlit Cloud: keep MediaPipe legacy Pose at complexity 1.
# Complexity 2 asks MediaPipe to download pose_landmark_heavy.tflite into site-packages,
# which is typically read-only on hosted deployments.
POSE_MODEL_COMPLEXITY = 1
APP_BUILD_TAG = "RULA_USER_UI_REVIEW_CENTER_2026-10-30"
# 既有計分矩陣沿用 RULA_VERIFIED_TABLES_FRONT_REVIEW_2026-10-30，便於追溯驗證基準。

# RULA 1993/2016 標準工作表 (使用者提供論文 表3-8, 表3-9, 表3-10)
# 三個互相獨立的矩陣，在 tests/test_independent_rula_tables.py 逐格比對固定人工轉錄值。
RULA_TABLE_A_ROWS = [
    [1, 2, 2, 2, 2, 3, 3, 3],
    [2, 2, 2, 2, 3, 3, 3, 3],
    [2, 3, 3, 3, 3, 3, 4, 4],
    [2, 3, 3, 3, 3, 4, 4, 4],
    [3, 3, 3, 3, 3, 4, 4, 4],
    [3, 4, 4, 4, 4, 4, 5, 5],
    [3, 3, 4, 4, 4, 4, 5, 5],
    [3, 4, 4, 4, 4, 4, 5, 5],
    [4, 4, 4, 4, 4, 5, 5, 5],
    [4, 4, 4, 4, 4, 5, 5, 5],
    [4, 4, 4, 4, 4, 5, 5, 5],
    [4, 4, 4, 5, 5, 5, 6, 6],
    [5, 5, 5, 5, 5, 6, 6, 7],
    [5, 6, 6, 6, 6, 6, 7, 7],
    [6, 6, 6, 7, 7, 7, 7, 8],
    [7, 7, 7, 7, 7, 8, 8, 9],
    [8, 8, 8, 8, 8, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9],
]
RULA_TABLE_B_ROWS = [
    [1, 3, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7],
    [2, 3, 2, 3, 4, 5, 5, 5, 6, 7, 7, 7],
    [3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7],
    [5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8],
    [7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9],
]
RULA_TABLE_C_ROWS = [
    [1, 2, 3, 3, 4, 5, 5, 5, 5],
    [2, 2, 3, 4, 4, 5, 5, 5, 5],
    [3, 3, 3, 4, 4, 5, 6, 6, 6],
    [3, 3, 3, 4, 5, 6, 6, 6, 6],
    [4, 4, 4, 5, 6, 7, 7, 7, 7],
    [4, 4, 5, 6, 6, 7, 7, 7, 7],
    [5, 5, 6, 6, 7, 7, 7, 7, 7],
    [5, 5, 6, 7, 7, 7, 7, 7, 7],
    [5, 5, 6, 7, 7, 7, 7, 7, 7],
]
TABLE_A = {1: {1: {1: {1: 1, 2: 2}, 2: {1: 2, 2: 2}, 3: {1: 2, 2: 3}, 4: {1: 3, 2: 3}}, 2: {1: {1: 2, 2: 2}, 2: {1: 2, 2: 2}, 3: {1: 3, 2: 3}, 4: {1: 3, 2: 3}}, 3: {1: {1: 2, 2: 3}, 2: {1: 3, 2: 3}, 3: {1: 3, 2: 3}, 4: {1: 4, 2: 4}}}, 2: {1: {1: {1: 2, 2: 3}, 2: {1: 3, 2: 3}, 3: {1: 3, 2: 4}, 4: {1: 4, 2: 4}}, 2: {1: {1: 3, 2: 3}, 2: {1: 3, 2: 3}, 3: {1: 3, 2: 4}, 4: {1: 4, 2: 4}}, 3: {1: {1: 3, 2: 4}, 2: {1: 4, 2: 4}, 3: {1: 4, 2: 4}, 4: {1: 5, 2: 5}}}, 3: {1: {1: {1: 3, 2: 3}, 2: {1: 4, 2: 4}, 3: {1: 4, 2: 4}, 4: {1: 5, 2: 5}}, 2: {1: {1: 3, 2: 4}, 2: {1: 4, 2: 4}, 3: {1: 4, 2: 4}, 4: {1: 5, 2: 5}}, 3: {1: {1: 4, 2: 4}, 2: {1: 4, 2: 4}, 3: {1: 4, 2: 5}, 4: {1: 5, 2: 5}}}, 4: {1: {1: {1: 4, 2: 4}, 2: {1: 4, 2: 4}, 3: {1: 4, 2: 5}, 4: {1: 5, 2: 5}}, 2: {1: {1: 4, 2: 4}, 2: {1: 4, 2: 4}, 3: {1: 4, 2: 5}, 4: {1: 5, 2: 5}}, 3: {1: {1: 4, 2: 4}, 2: {1: 4, 2: 5}, 3: {1: 5, 2: 5}, 4: {1: 6, 2: 6}}}, 5: {1: {1: {1: 5, 2: 5}, 2: {1: 5, 2: 5}, 3: {1: 5, 2: 6}, 4: {1: 6, 2: 7}}, 2: {1: {1: 5, 2: 6}, 2: {1: 6, 2: 6}, 3: {1: 6, 2: 6}, 4: {1: 7, 2: 7}}, 3: {1: {1: 6, 2: 6}, 2: {1: 6, 2: 7}, 3: {1: 7, 2: 7}, 4: {1: 7, 2: 8}}}, 6: {1: {1: {1: 7, 2: 7}, 2: {1: 7, 2: 7}, 3: {1: 7, 2: 8}, 4: {1: 8, 2: 9}}, 2: {1: {1: 8, 2: 8}, 2: {1: 8, 2: 8}, 3: {1: 8, 2: 9}, 4: {1: 9, 2: 9}}, 3: {1: {1: 9, 2: 9}, 2: {1: 9, 2: 9}, 3: {1: 9, 2: 9}, 4: {1: 9, 2: 9}}}}
TABLE_B = {1: {1: {1: 1, 2: 3}, 2: {1: 2, 2: 3}, 3: {1: 3, 2: 4}, 4: {1: 5, 2: 5}, 5: {1: 6, 2: 6}, 6: {1: 7, 2: 7}}, 2: {1: {1: 2, 2: 3}, 2: {1: 2, 2: 3}, 3: {1: 4, 2: 5}, 4: {1: 5, 2: 5}, 5: {1: 6, 2: 7}, 6: {1: 7, 2: 7}}, 3: {1: {1: 3, 2: 3}, 2: {1: 3, 2: 4}, 3: {1: 4, 2: 5}, 4: {1: 5, 2: 6}, 5: {1: 6, 2: 7}, 6: {1: 7, 2: 7}}, 4: {1: {1: 5, 2: 5}, 2: {1: 5, 2: 6}, 3: {1: 6, 2: 7}, 4: {1: 7, 2: 7}, 5: {1: 7, 2: 7}, 6: {1: 8, 2: 8}}, 5: {1: {1: 7, 2: 7}, 2: {1: 7, 2: 7}, 3: {1: 7, 2: 8}, 4: {1: 8, 2: 8}, 5: {1: 8, 2: 8}, 6: {1: 8, 2: 8}}, 6: {1: {1: 8, 2: 8}, 2: {1: 8, 2: 8}, 3: {1: 8, 2: 8}, 4: {1: 8, 2: 9}, 5: {1: 9, 2: 9}, 6: {1: 9, 2: 9}}}
TABLE_C = [[1, 2, 3, 3, 4, 5, 5, 5, 5], [2, 2, 3, 4, 4, 5, 5, 5, 5], [3, 3, 3, 4, 4, 5, 6, 6, 6], [3, 3, 3, 4, 5, 6, 6, 6, 6], [4, 4, 4, 5, 6, 7, 7, 7, 7], [4, 4, 5, 6, 6, 7, 7, 7, 7], [5, 5, 6, 6, 7, 7, 7, 7, 7], [5, 5, 6, 7, 7, 7, 7, 7, 7], [5, 5, 6, 7, 7, 7, 7, 7, 7]]
# 注意，Table A/B 的部分數值和個別後來改編的 Excel 工作表可能不同；
# 本系統採用固定的論文原始表及獨立工作表做可重現的對照，不隨套件版本改變。

# ===== v10.1: 影像內嵌顯示（保留原始比例、不依賴 /media） =====
def v10_image_html(image_data, caption="", encoded=False, max_width="100%"):
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


def v10_show_image(image_data, caption="", encoded=False):
    """Avoid both Streamlit media URLs and components.html iframe clipping."""
    st.markdown(v10_image_html(image_data, caption=caption, encoded=encoded),
                unsafe_allow_html=True)


def v10_thumbnail_b64(image_b64, max_width=400):
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



# ==================== 【V11：分關節骨架品質檢查與密集時間重查】 ====================
# 本區所有門檻是「需複核篩檢」，不是經實驗確認的精確度或人體解剖閾值。
# 只有跨條件的強訊號／孤立錯位才標記「需人工複核」，不因正常動作幅度大而直接剔除。
V11_QC_GUIDE = ("V11.2 骨架品質是工程性遮擋／定位錯誤篩檢，非量測準確率或遮擋的絕對判定。\n"
                "RULA 系統分數若標為需人工複核，不能直接作為正式風險結論。\n"
                "P2 跨鏡頭只提供可見度候選，不搬用未標定鏡頭的 2D 角度。\n"
                "正常高速移動不可只憑角度差判成異常；手臂靠近頭部也不直接代表關鍵點一定錯誤。\n")
V11_PROBE_SECONDS = 0.12
V11_QC_MODES = ("保守（推薦：只列嚴重疑慮）", "審慎（包含需留意的影格）")


def v11_pt(snap, i):
    if not isinstance(snap, dict):
        return None
    p = snap.get('Pose影像像素座標', {}).get(str(i))
    if not isinstance(p, dict):
        return None
    x, y, v = (float(p.get(k, float('nan'))) for k in ('x_px', 'y_px', 'visibility'))
    if not np.isfinite([x,y,v]).all():
        return None
    return np.array([x,y], dtype=float), v



# ==================== 【V11.1：頭頸／手臂近距遮擋風險、局部品質證據】 ====================
def v11_1_head_arm_proximity(snap, side, data):
    """2D 遮擋風險代理值，不等同已證明遮擋或關節錯位。

    當目標手腕投影到鼻尖與同側肩點之間、前臂屈曲投影極端，
    才增加品質證據。只用來篩檢，不修改任何角度或 RULA 查表。
    """
    out = {'頭頸手臂重疊風險': False, '頭頸手腕距離比': np.nan,
           '頭頸遮擋證據':'未發現明顯幾何重疊', '局部可能受影響':'',
           '頭頸遮擋強訊號':False}
    right = side == '右手側'
    ids = (12,14,16,24) if right else (11,13,15,23)
    points=[v11_pt(snap, i) for i in (0,*ids)]
    if any(p is None for p in points):
        out['頭頸遮擋證據']='頭部或手臂定位不足，無法進行相對幾何檢查'
        return out
    nose, shoulder, elbow, wrist, hip=[p[0] for p in points]
    torso=float(np.linalg.norm(shoulder-hip))
    if torso < 22:
        out['頭頸遮擋證據']='軀幹投影尺度過小，無法判斷'
        return out
    # 使用頭部到同側肩的中間區域，對應可能交疊的頭頸/上臂投影。
    zone=(nose + shoulder)/2.0
    dist=float(np.linalg.norm(wrist-zone)/torso)
    prefix='右手' if right else '左手'
    try: flex=float(data.get(prefix+'前臂角度',np.nan))
    except (TypeError,ValueError): flex=np.nan
    try: neck=float(data.get('頸部角度',np.nan))
    except (TypeError,ValueError): neck=np.nan
    # 需要幾何接近 + 屈肘大角度 + 明顯前傾三個條件才成立。
    # 0.24 / 0.36 是暫定影像尺度門檻，不是人體工學或解剖學常數。
    near = bool(dist < 0.36 and np.isfinite(flex) and flex >= 130
                and np.isfinite(neck) and neck >= 20)
    strong = bool(near and dist < 0.24 and flex >= 140 and neck >= 24)
    out['頭頸手腕距離比']=round(dist,3)
    out['頭頸手臂重疊風險']=near
    out['頭頸遮擋強訊號']=strong
    if near:
        out['局部可能受影響']='手腕、手肘、頸部'
        out['頭頸遮擋證據']=(f'手腕靠近頭頸投影區（{dist:.2f} 個軀幹長）'
            f'；前臂角度 {flex:.1f}°；頸部角度 {neck:.1f}°'
            '；須確認是否手臂／物品交疊')
    return out

def v11_1_neck_crosscheck(qc, data, peer):
    """另一側鏡頭僅提供頸部評分差異警示，不直接融合/替換 2D 角度。"""
    qc.update({'頸部跨鏡頭角度差':np.nan,'頸部跨鏡頭基礎分級差':np.nan,
               '頸部跨鏡頭診斷':'未比較'})
    if not isinstance(peer,dict):
        qc['頸部跨鏡頭診斷']='缺少同步另一側有效資料'
        return
    try:
        a=float(data.get('頸部獨立估測角度',data.get('頸部角度',np.nan)))
        b=float(peer.get('頸部獨立估測角度',peer.get('頸部角度',np.nan)))
    except (TypeError,ValueError):return
    if not (np.isfinite(a) and np.isfinite(b)):return
    def neck_base(angle):return 1 if angle<=10 else 2 if angle<=20 else 3
    angle_gap=abs(a-b); grade_gap=abs(neck_base(a)-neck_base(b))
    qc['頸部跨鏡頭角度差']=round(angle_gap,2)
    qc['頸部跨鏡頭基礎分級差']=grade_gap
    if angle_gap >= 15 and grade_gap >= 1:
        qc['頸部跨鏡頭診斷']='兩視角頸部基礎分級不一致，可能是遮擋、視角或同步問題'
        qc['品質原因'] += f'；左右頸部估計差 {angle_gap:.1f}° 且跨分級'
        parts=set(filter(None,qc.get('品質嫌疑關節','').split(',')))
        parts.add('neck')
        qc['品質嫌疑關節']=','.join(sorted(parts))
        if qc['品質狀態']=='通過':qc['品質狀態']='留意'
        # 跨鏡頭頸部分級分歧由 shared_neck_decide_pair 依行動等級影響處理；
        # 不能僅因角度差大＋手臂靠近頭部，就要求重評整張影格。
    elif angle_gap >= 15:
        qc['頸部跨鏡頭診斷']='兩視角角度差大，但頸部基礎分級一致；僅保留診斷'
    else:
        qc['頸部跨鏡頭診斷']='未發現顯著跨鏡頭分級差異'

def v11_quality_from_snapshot(snap, side, data=None):
    """幾何與可見度是不同證據；只有多證據疑慮才擱置系統評分。"""
    data = data or {}
    right = side == '右手側'
    ids = {'sh':12 if right else 11, 'el':14 if right else 13,
           'wr':16 if right else 15, 'hp':24 if right else 23,
           'other_wr':15 if right else 16}
    coords = {k:v11_pt(snap, idx) for k,idx in ids.items()}
    hard, soft, affected = [], [], []
    vis = {}
    h = float(snap.get('影像高', 0) or 0) if isinstance(snap,dict) else 0
    w = float(snap.get('影像寬', 0) or 0) if isinstance(snap,dict) else 0
    for joint in ('sh','el','wr','hp'):
        pair=coords.get(joint)
        if pair is None:
            hard.append(joint+' 座標無效')
            affected.append(joint)
            vis[joint]=float('nan')
            continue
        xy, v=pair
        vis[joint]=float(v)
        if v < 0.35:
            soft.append(joint+f' 可見度偏低 {v:.2f}')
            affected.append(joint)
        if w>0 and h>0 and (xy[0] < -0.05*w or xy[0] > 1.05*w or xy[1] < -0.05*h or xy[1] > 1.05*h):
            hard.append(joint+' 超出影像範圍')
            affected.append(joint)
    ratio=float('nan')
    for j in ('sh','el','wr','hp'):
        if coords[j] is None:
            return {'品質狀態':'需人工複核','品質原因':'；'.join(hard+soft),
                    '品質可直接採信':False,'品質嫌疑關節':','.join(sorted(set(affected))),
                    '品質前臂上臂比':ratio, '品質最低可見度':np.nan,
                    '品質需時序複查':False,'品質硬訊號數':len(hard),'品質軟訊號數':len(soft)}
    shoulder, elbow, wrist, hip=(coords[x][0] for x in ('sh','el','wr','hp'))
    upper=float(np.linalg.norm(elbow-shoulder))
    lower=float(np.linalg.norm(wrist-elbow))
    torso=float(np.linalg.norm(hip-shoulder))
    ratio=lower/upper if upper>1e-5 else float('nan')
    if upper<8 or lower<8 or torso<14:
        hard.append('肢段投影尺度過小，幾何定位可能失真')
        affected.extend(['el','wr'])
    if np.isfinite(ratio) and (ratio<0.22 or ratio>3.8):
        soft.append(f'前臂／上臂投影比值偏離一般範圍 {ratio:.2f}')
        affected.extend(['el','wr'])
    # 只能稱「影像投影比例異常」；不將透視縮短直接判成解剖不可能。
    if lower < 0.13 * max(1,torso):
        soft.append('前臂在影像中的投影非常短（可能朝向鏡頭／被遮擋）')
        affected.append('wr')
    if upper < 0.13 * max(1,torso):
        soft.append('上臂在影像中的投影非常短（可能朝向鏡頭／被遮擋）')
        affected.append('el')
    if coords['other_wr'] is not None:
        other_wrist=coords['other_wr'][0]
        if np.linalg.norm(other_wrist-wrist) < 0.09*max(1,torso):
            soft.append('兩側手腕投影接近，可能重疊')
            affected.append('wr')
    # 大幅角度不直接算錯，僅觸發前後影格檢查（可避免原本只憑跳動就剔除）。
    prefix='右手' if right else '左手'
    elbow_flex=float(data.get(prefix+'前臂角度', float('nan')))
    wrist_flex=float(data.get(prefix+'手腕角度', float('nan')))
    if np.isfinite(elbow_flex) and elbow_flex>=165:
        soft.append(f'前臂角度達 {elbow_flex:.1f}°，建議時序重查')
        affected.append('el')
    if np.isfinite(wrist_flex) and wrist_flex>=65:
        soft.append(f'手腕角度達 {wrist_flex:.1f}°，建議時序重查')
        affected.append('wr')
    # 保留 Hand 與 Pose 角度差，不將「與另一個模型不一致」當成直接否決條件。
    if not bool(data.get(prefix+'Hand用於計分',False)) and vis.get('wr',1) < 0.5:
        soft.append('Hand 無可用角度且 Pose 腕點可靠度偏低')
        affected.append('wr')
    overlap=v11_1_head_arm_proximity(snap,side,data)
    if overlap['頭頸手臂重疊風險']:
        soft.append(overlap['頭頸遮擋證據'])
        affected.extend(['el','wr','neck'])
    # 僅高度重疊、極端屈肘與頭頸前傾共同出現時先列局部複核；
    # 這是遮擋風險而非已確定錯點，仍保留系統評分與雙照片。
    if overlap['頭頸遮擋強訊號']:
        hard.append('頭頸與手腕投影高度重疊且前臂屈曲極端，定位證據不足')
    minimum=min(x for x in vis.values() if np.isfinite(x)) if any(np.isfinite(x) for x in vis.values()) else np.nan
    # 低可見度只有一項時不直接擱置；有兩個獨立品質疑慮才進入複核。
    is_hard=bool(hard) or (len(soft)>=2 and min(vis.get('el',1),vis.get('wr',1)) < 0.42)
    state='需人工複核' if is_hard else ('留意' if soft else '通過')
    return {'品質狀態':state,'品質原因':'；'.join(hard+soft) if (hard or soft) else '主要關節品質檢查通過',
            '品質可直接採信':state!='需人工複核',
            '品質嫌疑關節':','.join(sorted(set(affected))),
            '品質前臂上臂比':round(ratio,3) if np.isfinite(ratio) else np.nan,
            '品質最低可見度':round(minimum,3) if np.isfinite(minimum) else np.nan,
            '品質需時序複查':bool(soft or hard),
            '品質硬訊號數':len(hard),'品質軟訊號數':len(soft), **overlap}


def v11_normalized_points_from_landmarks(landmarks, side):
    """所有點在同一影像正規化，避免因人物在鏡頭內平移產生假尖峰。"""
    right=side=='右手側'
    sh=12 if right else 11; el=14 if right else 13; wr=16 if right else 15
    hp=24 if right else 23
    ids=(sh,el,wr,hp)
    try:
        pts=np.array([[float(landmarks[i].x),float(landmarks[i].y)] for i in ids])
        vis=np.array([float(landmarks[i].visibility) for i in ids])
        if not np.all(np.isfinite(pts)) or not np.all(np.isfinite(vis)):
            return None
        unit=float(np.linalg.norm(pts[0]-pts[3]))
        if unit<0.035:
            return None
        return {'el':(pts[1]-pts[0])/unit,'wr':(pts[2]-pts[0])/unit,
                'visibility':{'el':float(vis[1]),'wr':float(vis[2])}}
    except (IndexError,AttributeError,TypeError,ValueError):
        return None


def v11_probe_neighbor_frame(cap, index, pose_model, side):
    """只重讀附近影格作診斷，絕不把鄰近姿勢直接充作當前真值。"""
    cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,int(index)))
    ok, frame=cap.read()
    if not ok or frame is None or frame.size==0:
        return None
    h,w=frame.shape[:2]
    scale=min(850.0/max(h,w),1.0)
    if scale<1:
        frame=cv2.resize(frame,(int(w*scale),int(h*scale)))
    try:
        results=pose_model.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        if not getattr(results,'pose_landmarks',None):
            return None
        return v11_normalized_points_from_landmarks(results.pose_landmarks.landmark,side)
    except Exception:
        logging.exception('V11 neighbor pose probe failed')
        return None


def v11_temporal_recheck(cap, current_frame_idx, fps, pose_model, side, snap):
    """前後影格彼此相近且中心孤立偏移，才警示；否則記錄無法確定。"""
    out={'時間重查':'未啟動','時間重查判定':'資訊不足','時間錯位關節':'','重查間隔(秒)':V11_PROBE_SECONDS}
    base_points=snap.get('Pose影像像素座標',{}) if isinstance(snap,dict) else {}
    if not base_points: return out
    # 用原始 0~1 座標重建中心幀的正規化關節位置。
    hh=float(snap.get('影像高',0) or 0); ww=float(snap.get('影像寬',0) or 0)
    if hh<=0 or ww<=0: return out
    def _mk(idx):
        v=base_points.get(str(idx));
        if not isinstance(v,dict): return None
        return np.array([float(v['x_px'])/ww,float(v['y_px'])/hh])
    r=side=='右手側'; sh=_mk(12 if r else 11); hp=_mk(24 if r else 23)
    el=_mk(14 if r else 13); wr=_mk(16 if r else 15)
    if any(x is None for x in (sh,hp,el,wr)): return out
    scale=float(np.linalg.norm(sh-hp))
    if scale<0.035: return out
    middle={'el':(el-sh)/scale,'wr':(wr-sh)/scale}
    frames=max(1,round(float(fps)*V11_PROBE_SECONDS))
    before=max(0,int(current_frame_idx)-frames)
    after=int(current_frame_idx)+frames
    old=int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    try:
        a=v11_probe_neighbor_frame(cap,before,pose_model,side)
        b=v11_probe_neighbor_frame(cap,after,pose_model,side)
    finally:
        cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,old))
    out['時間重查']='已檢查前後影格'
    if a is None or b is None:
        out['時間重查判定']='鄰近關鍵點不足，無法確認'
        return out
    suspected=[]
    for joint in ('el','wr'):
        if min(a['visibility'][joint],b['visibility'][joint])<0.5: continue
        neighbor_distance=float(np.linalg.norm(a[joint]-b[joint]))
        deviation=float(np.linalg.norm(middle[joint]-(a[joint]+b[joint])/2))
        # 中心影格孤立跳動，不能只因相鄰兩張自然動作差很多而報警。
        if neighbor_distance<0.18 and deviation>0.38:
            suspected.append(f'{joint}:前後一致但中心偏離{deviation:.2f}個軀幹長')
    if suspected:
        out['時間重查判定']='疑似孤立關鍵點錯位'
        out['時間錯位關節']='；'.join(suspected)
    else:
        out['時間重查判定']='未發現孤立跳點（不代表位置一定正確）'
    return out


def v11_cross_view_hint(other_frame, pose_model, side, current_visibility):
    """P2 僅比較同一隻手在其他鏡頭的可見度，不直接挪用角度。"""
    result={'跨視角診斷':'未檢查','其他鏡頭腕肘最低可見度':np.nan}
    if other_frame is None or getattr(other_frame,'size',0)==0:
        return result
    h,w=other_frame.shape[:2]
    scale=min(760.0/max(h,w),1.0)
    smaller=cv2.resize(other_frame,(int(w*scale),int(h*scale))) if scale<1 else other_frame
    try:
        out=pose_model.process(cv2.cvtColor(smaller,cv2.COLOR_BGR2RGB))
        if not getattr(out,'pose_landmarks',None):
            result['跨視角診斷']='另一鏡頭未偵測到人體'
            return result
        lm=out.pose_landmarks.landmark
        ids=(14,16) if side=='右手側' else (13,15)
        v=float(min(lm[ids[0]].visibility,lm[ids[1]].visibility))
        result['其他鏡頭腕肘最低可見度']=round(v,3)
        result['跨視角診斷']=('另一鏡頭的腕肘可能較清楚；可人工比對原圖' if v>current_visibility+0.18 and v>=0.60
                           else '另一鏡頭未顯著改善腕肘可見度')
    except Exception:
        result['跨視角診斷']='其他鏡頭品質檢查失敗'
    return result


def v11_update_qc(data, snap, side, side_cap=None, side_frame_idx=None, side_fps=None,
                  pose_model=None, other_frame=None, policy='保守（推薦：只列嚴重疑慮）', neck_peer=None):
    """P0-P2 關節品質標記；保存系統評分，不以可疑值作為可採信結論。"""
    qc=v11_quality_from_snapshot(snap,side,data)
    qc.update({'時間重查':'未觸發','時間重查判定':'未觸發','時間錯位關節':'',
               '跨視角診斷':'未觸發','其他鏡頭腕肘最低可見度':np.nan})
    if qc['品質需時序複查'] and side_cap is not None and pose_model is not None:
        t=v11_temporal_recheck(side_cap,side_frame_idx,side_fps,pose_model,side,snap)
        qc.update(t)
        if t['時間重查判定']=='疑似孤立關鍵點錯位':
            qc['品質狀態']='需人工複核'
            qc['品質原因']+='；'+t['時間錯位關節']
            qc['品質可直接採信']=False
    if qc['品質狀態']=='需人工複核' and other_frame is not None and pose_model is not None:
        cur=qc.get('品質最低可見度',np.nan)
        qc.update(v11_cross_view_hint(other_frame,pose_model,side,float(cur) if np.isfinite(cur) else 0.0))
    v11_1_neck_crosscheck(qc,data,neck_peer)
    # 不確定 ≠ 必須逐張人工：先用原 RULA Table 測試外側工作有/無是否改變總分。
    if data.get('外側工作判定狀態') == '待複核':
        impact=rula_outside_score_sensitivity(data,side)
        data['外側工作分數敏感度']=impact['note']
        if impact['same_grand']:
            qc['品質原因']=str(qc.get('品質原因',''))+'；前臂向外工作不確定，但兩情境 Grand Score 相同（保留診斷、不強制逐張確認）'
            if qc['品質狀態']=='通過':qc['品質狀態']='留意'
        else:
            qc['品質狀態'] = '需人工複核'
            qc['品質原因'] = str(qc.get('品質原因', '')) + '；前臂向外側工作不確定：'+impact['note']
            qc['品質嫌疑關節'] = ','.join(sorted(set(filter(None,
                str(qc.get('品質嫌疑關節', '')).split(',') + ['el', 'wr']))))
    qc['品質可直接採信']=(qc['品質狀態']!='需人工複核')
    qc['品質是否列入複核中心']=(qc['品質狀態']=='需人工複核' or
                        (policy.startswith('審慎') and qc['品質狀態']=='留意'))
    qc['品質計分說明']=('已輸出系統評分，需人工確認後才可採信' if qc['品質狀態']=='需人工複核'
                    else '工程品質檢查通過；仍非角度準確率證明')
    data.update(qc)
    return qc


def v11_front_fallback_note(qc, data, front_ok, auto_settings):
    """正面不可用改採作業預設；不把缺少正面當成側面骨架嚴重失敗。

    若側面原本有嚴重 QC 則保留原狀。未觀測的正面補正仍需依作業資訊或
    局部照片批次修正；但不能自動使全部缺正面樣本進入強制人工複核。
    """
    front_flags=('is_abducted','is_across_midline','is_out_to_side','is_side_bending','is_neck_side_bending')
    if front_ok or not any(bool(auto_settings.get(x,False)) for x in front_flags):
        return qc
    if qc.get('品質狀態') == '通過':
        qc['品質狀態']='留意'
    # 保留骨架自身的品質可直接採信與列入複核判斷，不因 Front 影格品質而覆蓋。
    qc['品質原因']=str(qc.get('品質原因',''))+'；正面補正無法影像判斷，沿用分析前的作業設定'
    qc['品質計分說明']='側面骨架結果照常採用；正面補正來自作業設定，例外可批次修改'
    # 有 Front、但該影格因轉向等因素失去外側工作辨識能力時，必須列入局部複核。
    # 「全段未上傳 Front」則依使用者事先作業設定，不使每一張自動排入複核中心。
    if (bool(auto_settings.get('is_out_to_side', False)) and
        data.get('外側工作判定狀態') == '人工設定（正面不可用）'):
        impact=rula_outside_score_sensitivity(data, '右手側' if '右手上臂分數' in data else '左手側')
        data['外側工作分數敏感度']=impact['note']
        if impact['same_grand']:
            qc['品質原因'] += '；此張 Front 無法估算前臂外側工作，但不影響 RULA Grand Score，僅保留診斷'
            qc['品質計分說明'] = ('前臂向外側工作仍未確認；有／無兩情境的 RULA 總分相同。'
                              '若另有骨架或頸部品質問題仍須複核。')
        else:
            qc['品質狀態'] = '需人工複核'
            qc['品質可直接採信'] = False
            qc['品質是否列入複核中心'] = True
            qc['品質原因'] += '；此張 Front 視角不可靠，外側工作補正可能影響分數：'+impact['note']
            qc['品質計分說明'] = '前臂向外側工作沿用暫算值；需局部人工複核後才能採信'
    data.update(qc)
    return qc


def v11_mark_suspected_joints(overlay,snap,side,qc):
    """只用當次辨識到的關節位置畫紅圈，絕不把猜測點畫成新真值。"""
    if overlay is None or not isinstance(overlay,np.ndarray):return overlay
    result=overlay.copy()
    orig_h=float(snap.get('影像高',0) or 0) if isinstance(snap,dict) else 0
    orig_w=float(snap.get('影像寬',0) or 0) if isinstance(snap,dict) else 0
    if orig_w<=0 or orig_h<=0:return result
    target={'sh':12,'el':14,'wr':16,'hp':24} if side=='右手側' else {'sh':11,'el':13,'wr':15,'hp':23}
    oh,ow=result.shape[:2]
    suspicious=set(str(qc.get('品質嫌疑關節','')).split(','))
    if 'neck' in suspicious:
        point=v11_pt(snap,0)  # 只圈原模型實際預測的鼻尖點，不創建新關節。
        if point is not None:
            xy,_=point
            px=int(xy[0]*ow/orig_w); py=int(xy[1]*oh/orig_h)
            if 0<=px<ow and 0<=py<oh:
                cv2.circle(result,(px,py),24,(0,0,255),3,cv2.LINE_AA)
                cv2.putText(result,'QC:head',(max(0,px-34),max(16,py-27)),
                            cv2.FONT_HERSHEY_SIMPLEX,.50,(0,0,255),2,cv2.LINE_AA)
    for key in suspicious:
        if key not in target:continue
        pt=v11_pt(snap,target[key])
        if pt is None:continue
        xy,_=pt
        x=int(xy[0]*ow/orig_w);y=int(xy[1]*oh/orig_h)
        if 0<=x<ow and 0<=y<oh:
            cv2.circle(result,(x,y),19,(0,0,255),3,cv2.LINE_AA)
            cv2.putText(result,'QC:'+key,(max(0,x-22),max(16,y-24)),cv2.FONT_HERSHEY_SIMPLEX,.50,(0,0,255),2,cv2.LINE_AA)
    return result


def v11_add_qc_review(items, photos, sample_id, sec, side, qc, frames, pose_model, overlay=None, snap=None):
    """新增高風險骨架疑慮於原有雙圖片中心（原始圖＋當次骨架圖）。"""
    before=len(items)
    v8_add_rejected_sample(items,photos,sample_id,sec,side,
                           '骨架品質：'+qc['品質原因'],frames,pose_model)
    if len(items)<=before: return
    item=items[-1]
    only_notice = qc.get('品質狀態')=='留意'
    item['紀錄類型']=('骨架品質提醒（仍可採信）' if only_notice else '骨架品質待複核（有RULA 系統分數）')
    item['評分狀態']=('品質提醒：原分數維持有效，無須強制人工複核' if only_notice
                    else '僅系統評分，未經複核不應作為正式結論')
    item['局部可能受影響']=qc.get('局部可能受影響','')
    item['頭頸遮擋證據']=qc.get('頭頸遮擋證據','')
    item['頸部跨鏡頭診斷']=qc.get('頸部跨鏡頭診斷','')
    item['可疑關節']=qc.get('品質嫌疑關節','')
    item['時間重查判定']=qc.get('時間重查判定','')
    item['跨視角診斷']=qc.get('跨視角診斷','')
    side_cam='Right' if side=='右手側' else 'Left'
    key=f'Sample_{int(sample_id):03d}_{side}_{side_cam}'
    if overlay is not None and key in photos:
        image=_encode_preview(v11_mark_suspected_joints(overlay,snap,side,qc))
        if image:
            photos[key]['overlay']=image
            photos[key]['note']='本次正式分析的骨架圖；'+qc['品質原因']


def v112_is_informational_notice(item):
    """A QC notice is not a failed sample and does not require confirmation.

    Only explicitly recognized informational entries are filtered; unknown/legacy
    record types stay in the actionable group to prevent hiding failures.
    """
    return str(item.get('紀錄類型', '')).strip() == '骨架品質提醒（仍可採信）'


def v112_split_review_items(items):
    """Return (actionable, informational), preserving every original record."""
    actionable, informational = [], []
    for item in items or []:
        (informational if v112_is_informational_notice(item) else actionable).append(item)
    return actionable, informational


def v11_qc_df(df):
    if df is None or df.empty or '品質狀態' not in df.columns:
        return pd.DataFrame(columns=['樣本編號','時間(秒)','評估側','品質狀態','品質原因'])
    return df[df['品質狀態']!='通過'].copy()

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
    # 與 Pose 2D 數值不同並不是拒絕 Hand 的理由；僅由獨立的幾何指標篩選。
    # Pose visibility 仍另存診斷，但不作為 Hand 對 Pose 分歧的否決條件。
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



# 側面有向角度：畫面 X 正負不直接表示前後，必須先確認頭部朝向。
# 輸出是 2D 投影「候選姿勢」，非實際 3D 頸椎或肩關節量角。
def side_directional_posture(nose, ear, eye, shoulder_center, hip_center,
                             shoulder, elbow, nose_vis=1., ear_vis=1., eye_vis=1.):
    def p(x):
        a=np.asarray(x,dtype=float)
        return a if a.shape==(2,) and np.all(np.isfinite(a)) else None
    pts=[p(v) for v in (nose,ear,eye,shoulder_center,hip_center,shoulder,elbow)]
    out={'face_side':'不明', 'trunk_signed_deg':np.nan, 'upper_arm_signed_deg':np.nan,
         'neck_proxy_deg':np.nan, 'trunk_extended':None, 'upper_arm_extended':None,
         'neck_extended':None,'diagnostic':'頭部朝向或關節品質不足，維持人工初始設定'}
    if any(v is None for v in pts): return out
    no,ea,ey,sc,hc,sh,el=pts
    torso=sc-hc; arm=el-sh
    torso_len=float(np.linalg.norm(torso)); arm_len=float(np.linalg.norm(arm))
    face_x=float(no[0]-ea[0]); face_dist=float(np.linalg.norm(no-ea))
    # 鼻與眼相對同一耳朵都要指向同側；例如低頭遮擋時鼻點會突然跑到耳朵另一邊。
    eye_x=float(ey[0]-ea[0])
    if (nose_vis<0.55 or ear_vis<0.5 or eye_vis<0.5 or torso_len<28 or arm_len<16 or
        face_dist<10 or abs(face_x)<max(7.,0.075*torso_len) or
        abs(eye_x)<max(6.,0.04*torso_len) or eye_x*face_x<=0):
        return out
    sign=1. if face_x>0 else -1.
    out['face_side']='面向畫面右' if sign>0 else '面向畫面左'
    # 軀幹由髖指向肩；向臉部前方的偏移為正，向背部為負。
    signed_tk=float(np.degrees(np.arctan2(sign*torso[0],-torso[1])))
    out['trunk_signed_deg']=round(signed_tk,2)
    # 用軀幹自身下向量及其前向法線表示上臂屈伸，避免直接看肘部在肩左或右。
    down=-torso/torso_len
    front=sign*np.array([down[1],-down[0]])
    signed_arm=float(np.degrees(np.arctan2(np.dot(arm,front),np.dot(arm,down))))
    out['upper_arm_signed_deg']=round(signed_arm,2)
    if -65.<=signed_tk<=-10.:out['trunk_extended']=True
    elif signed_tk>=-4.:out['trunk_extended']=False
    # -10 至 -4 度屬於接近閾值的模糊區，沿用人工設定。
    if signed_arm < -23.:out['upper_arm_extended']=True
    elif signed_arm > -17.:out['upper_arm_extended']=False
    # 頭部候選：耳→眼代表臉部朝向，扣除軀幹的投影傾斜；非頸椎量測。
    # 僅對非常明顯的仰頭採用自動判讀，其餘接近邊界時維持人工設定。
    if eye_vis>=0.58:
        eye_vec=ey-ea
        head_forward=float(sign*eye_vec[0]);head_len=float(np.linalg.norm(eye_vec))
        if head_len>10. and head_forward>=max(6.,0.18*head_len) and abs(signed_tk)<80.:
            head_pitch=float(np.degrees(np.arctan2(-eye_vec[1],head_forward)))
            relative_pitch=head_pitch+signed_tk
            out['neck_proxy_deg']=round(relative_pitch,2)
            if relative_pitch>=40.:out['neck_extended']=True
            elif relative_pitch<=18.:out['neck_extended']=False
    out['diagnostic']='側面有向角度為候選估測；邊界／遮擋沿用人工初始設定'
    return out

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
    grand_score = TABLE_C[min(max(score_c, 1), 9) - 1][min(max(score_d, 1), 9) - 1]
    al_num = 1 if grand_score <= 2 else (2 if grand_score <= 4 else (3 if grand_score <= 6 else 4))
    return score_a, score_b, score_c, score_d, grand_score, f"AL{al_num}"


# ===== 對不確定的前臂外側工作，先測試「有／無 +1」是否實際改變 RULA 分數 =====
def rula_outside_score_sensitivity(row, side, profile=None):
    """在不猜測姿勢真值下，測試 RULA 有／無外側工作補正的兩種結果。

    若兩種情境的 Grand Score 完全相同，該姿勢條件仍未知，但當次總分
    不因它而變；不需要為此單一未知條件強制人工逐張審核。
    注意：這不能代替人體姿勢正確率驗證，其他品質或頸部疑慮仍照常複核。
    """
    pre='右手' if side=='右手側' else '左手'
    try:
        candidates=[]
        for value in (False, True):
            if profile is not None:
                scenario=dict(profile)
                scenario['out_side']=value
                out=batch_recalculate_side(row, side, scenario)
                result=(int(out[pre+' Grand Score']),str(out[pre+' AL']))
            else:
                across=str(row.get(pre+'橫越中線補正','否')).startswith('是')
                la=get_lower_arm_score(float(row[pre+'前臂角度']),across,value)
                out=compute_rula_full(
                    int(row[pre+'上臂分數']), la,
                    int(row[pre+'手腕分數']),int(row[pre+'手腕扭轉分數']),
                    int(row['頸部分數']),int(row['軀幹分數']),
                    int(row['腿部分數']),int(row['肌肉使用加分']),
                    int(row['負重施力加分']))
                result=(int(out[4]),str(out[5]))
            candidates.append(result)
        same_grand=(candidates[0][0]==candidates[1][0])
        same_level=(candidates[0][1]==candidates[1][1])
        return {'known':True,'same_grand':same_grand,'same_level':same_level,
                'no_outside':candidates[0], 'yes_outside':candidates[1],
                'note':('有／無外側工作補正的 RULA 總分完全相同' if same_grand else
                        ('總分不同、行動等級相同' if same_level else '行動等級會改變'))}
    except (ValueError,TypeError,KeyError,OverflowError,IndexError) as exc:
        return {'known':False,'same_grand':False,'same_level':False,
                'note':'無法評估外側補正分數敏感度：'+str(exc)}


# ===== 結案版：同一時間點共用頸部分數，只有可能改變行動等級才局部複核 =====
# 注意：左右影像 2D 頸角不等於經校準的頸椎角度；本邏輯不選「信心值高的那邊」。
def shared_neck_rula_with_score(data, side, neck_score, manual_override=None):
    """固定其他部位，只把頸部分數改成同一個共用值後重新執行 RULA 查表。"""
    prefix = '右手' if side == '右手側' else '左手'
    manual_override = manual_override or {}
    def val(label, key):
        raw=manual_override.get('人工'+label+'分數',data[key])
        return int(raw)
    return compute_rula_full(
        val('上臂',prefix+'上臂分數'), val('前臂',prefix+'前臂分數'),
        val('手腕',prefix+'手腕分數'), val('腕扭轉',prefix+'手腕扭轉分數'),
        int(neck_score), val('軀幹','軀幹分數'), val('腿部','腿部分數'),
        val('肌肉','肌肉使用加分'), val('負重','負重施力加分')
    )


def shared_neck_apply_to_side(data, side, neck_score, status, source):
    """覆寫正式頸部分數及所有依賴它的 RULA 表格與總分；保留原始角度於技術欄位。"""
    prefix='右手' if side=='右手側' else '左手'
    data.setdefault('頸部獨立估測分數',int(data['頸部分數']))
    data.setdefault('頸部獨立估測角度',data.get('頸部角度'))
    result=shared_neck_rula_with_score(data,side,neck_score)
    a,b,c,d,total,al=result
    data.update({'頸部分數':int(neck_score),'頸部共用分數':int(neck_score),
        '頸部共用狀態':status,'頸部共用來源':source,
        prefix+' Score A':a,'頸軀腿 Score B':b,prefix+' Score C':c,
        '頸軀腿 Score D':d,prefix+' Grand Score':total,
        prefix+' AL':al,'最終最高分':total,'最終 AL':al})
    return result


def shared_neck_decide_pair(right, left, sample_id, second):
    """不從較高 visibility 猜真值。若兩種頸分可導致不同完整雙側 AL，改人工局部確認。"""
    if right is None or left is None:
        return None
    nr=int(right['頸部分數']); nl=int(left['頸部分數'])
    options=sorted({nr,nl})
    scenario={}
    for n in options:
        r=shared_neck_rula_with_score(right,'右手側',n)
        l=shared_neck_rula_with_score(left,'左手側',n)
        worst=max(r[4],l[4]);al='AL'+str(1 if worst<=2 else 2 if worst<=4 else 3 if worst<=6 else 4)
        scenario[n]={'右手分數':r[4],'左手分數':l[4],'雙側最高分':worst,'雙側AL':al}
    impactful=len({v['雙側AL'] for v in scenario.values()})>1
    chosen=max(options)  # 若 AL 不變，仍保守採用較高的同一個頸部分數；不是角度真值
    if nr==nl:
        state='一致，自動採用';source='左右頸部分數一致'
    elif impactful:
        state='待頸部人工確認';source='左右不同且可能改變雙側行動等級'
    else:
        state='自動保守共用';source='左右不同，但完整雙側行動等級不變'
    result={'樣本編號':int(sample_id),'時間(秒)':float(second),
        '右側原頸部分數':nr,'左側原頸部分數':nl,
        '右側原頸角':right.get('頸部角度'), '左側原頸角':left.get('頸部角度'),
        '選定共用頸部分數':chosen,'狀態':state,'需要人工確認':impactful,
        '選擇說明':source,'各候選情境':scenario}
    shared_neck_apply_to_side(right,'右手側',chosen,state,source)
    shared_neck_apply_to_side(left,'左手側',chosen,state,source)
    return result


def shared_neck_reviewed_score(source_row, side, neck_score, manual=None):
    """讓人工頸分真正反映在複核後雙側分數；未取代手動輸入的其他部位。"""
    values=shared_neck_rula_with_score(source_row,side,neck_score,manual)
    return int(values[4]),str(values[5])


def shared_neck_case_table(cases, decisions):
    rows=[]
    for sid,case in sorted((cases or {}).items()):
        chosen=(decisions or {}).get(int(sid))
        rows.append({'樣本編號':sid,'系統狀態':case['狀態'],
          '目前共用頸部分數':int(chosen) if chosen is not None else (
              None if case['需要人工確認'] else int(case['選定共用頸部分數'])),
          '是否仍需人工確認':bool(case['需要人工確認'] and chosen is None),
          '說明':('已人工確認頸部分數' if chosen is not None else case['選擇說明'])})
    return pd.DataFrame(rows)

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

# ===== 結案候選：前臂向外側工作，雙軀幹參考＋前臂方向（僅影像代理值） =====
# 注意：以下門檻是工程測試值，並非原始 RULA 的角度／距離標準。
# 在單張正面投影無法確認時回傳「待複核」，不可以直接當成 RULA 正式真值。
RULA_OUTSIDE_SHOULDER_YES = 0.30    # 手腕超出同側肩點的肩寬比例
RULA_OUTSIDE_HIP_YES = 0.28         # 手腕超出同側髖點的肩寬比例
RULA_OUTSIDE_FOREARM_YES = 0.16     # 腕相對肘的側向伸出肩寬比例
RULA_OUTSIDE_DIRECTION_YES = 0.26   # 前臂側向投影佔前臂投影長比例


def rula_outside_work_geometry(sh_left, sh_right, hp_left, hp_right, elbow, wrist, target_side):
    """純 2D 投影的三態分類：自動是／自動否／待複核。

    依*人體*肩線定義左右方向、以肩寬正規化，分別比較同側肩與髖；
    要求前臂肘→腕確實向外，防止把自然下垂或只有上臂外展誤判為向外工作。
    本函式不分析工作物／施力方向；正面大幅轉向應由呼叫端先拒絕。
    """
    v = [np.asarray(p, dtype=float) for p in (sh_left, sh_right, hp_left, hp_right, elbow, wrist)]
    if any(p.shape != (2,) or not np.isfinite(p).all() for p in v):
        return {'status': '待複核', 'reason': '肩、髖、肘或腕座標缺失／不合理'}
    sl, sr, hl, hr, el, wr = v
    shoulder_axis = sr - sl
    shoulder_width = float(np.linalg.norm(shoulder_axis))
    if shoulder_width < 12.0:
        return {'status': '待複核', 'reason': '肩寬投影過短，無法判定外側工作'}
    lateral_axis = shoulder_axis / shoulder_width
    is_right = target_side == '右手側'
    if target_side not in ('右手側', '左手側'):
        return {'status': '待複核', 'reason': '未指定有效評估側別'}
    side_sign = 1.0 if is_right else -1.0
    own_sh = sr if is_right else sl
    own_hp = hr if is_right else hl
    shoulder_margin = side_sign * float(np.dot(wr - own_sh, lateral_axis)) / shoulder_width
    hip_margin = side_sign * float(np.dot(wr - own_hp, lateral_axis)) / shoulder_width
    forearm_lateral = side_sign * float(np.dot(wr - el, lateral_axis)) / shoulder_width
    forearm_length = float(np.linalg.norm(wr - el))
    if forearm_length < max(9.0, 0.16 * shoulder_width):
        return {'status': '待複核', 'reason': '前臂投影過短或遮擋，方向不可靠',
                'wrist_shoulder_margin': round(shoulder_margin, 3),
                'wrist_hip_margin': round(hip_margin, 3)}
    direction_ratio = forearm_lateral * shoulder_width / forearm_length
    metrics = {'wrist_shoulder_margin': round(shoulder_margin, 3),
               'wrist_hip_margin': round(hip_margin, 3),
               'forearm_lateral_margin': round(forearm_lateral, 3),
               'forearm_direction_ratio': round(direction_ratio, 3)}
    if (shoulder_margin >= RULA_OUTSIDE_SHOULDER_YES and
        hip_margin >= RULA_OUTSIDE_HIP_YES and
        forearm_lateral >= RULA_OUTSIDE_FOREARM_YES and
        direction_ratio >= RULA_OUTSIDE_DIRECTION_YES):
        return {'status': '自動是', 'reason': '腕點超出同側肩髖＋肘至腕朝人體外側延伸', **metrics}
    if (shoulder_margin <= 0.10 or hip_margin <= 0.10 or
        forearm_lateral <= 0.045 or direction_ratio <= 0.10):
        return {'status': '自動否', 'reason': '手腕未明顯離開軀幹兩參考點或前臂未朝外側延伸', **metrics}
    return {'status': '待複核', 'reason': '位置／前臂方向介於可靠判定範圍，待人工確認', **metrics}


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
    # 正面旋轉程度以 World 深度差與投影幾何檢查；只要正面視角可疑，
    # 就不使用該影格的正面自動補正，改用使用者的作業初始設定。
    # 此檢查不等於已驗證的解剖量角準確性。
    world = getattr(results, "pose_world_landmarks", None)
    rotation_warning = False
    if world is not None:
        wlm = world.landmark
        shoulder_depth_delta = abs(float(wlm[12].z - wlm[11].z))
        shoulder_world_width = float(np.linalg.norm([wlm[12].x-wlm[11].x, wlm[12].y-wlm[11].y, wlm[12].z-wlm[11].z]))
        rotation_warning = shoulder_depth_delta / max(shoulder_world_width, 1e-6) > 0.55
    # 在明顯轉身、肩線接近重疊、彎腰造成正面投影失真時禁止自動補正。
    hip_width = float(np.linalg.norm(hp_right - hp_left))
    shoulder_tilt = math.degrees(math.atan2(abs(shoulder_vec[1]), max(abs(shoulder_vec[0]), 1e-6)))
    foreshortened = (shoulder_width < 0.48 * torso_length or
                     (hip_width >= 15 and shoulder_width < 0.65 * hip_width))
    unreliable_projection = bool(rotation_warning or foreshortened or shoulder_tilt > 22.0)
    if unreliable_projection:
        return False, {"front_auto_paused": True,
                       "front_rotation_warning": bool(rotation_warning),
                       "front_screening_reason": ("姿勢轉身或正面投影不可靠："
                           + ("肩深度旋轉；" if rotation_warning else "")
                           + ("肩線縮短；" if foreshortened else "")
                           + ("肩線傾斜；" if shoulder_tilt > 22 else "")),
                       "front_keypoint_visibility": float(np.mean(_visibility_values(results, required))*100.0)}
    # 目標側肩膀位於軸向哪一邊，避免鏡像影像造成左右符號反轉。
    target_sh = xy(sh_idx)
    elbow = xy(el_idx)
    wrist = xy(wr_idx)
    side_sign = float(np.sign(np.dot(target_sh - shoulder_mid, horiz)))
    if side_sign == 0:
        return False, {}
    elbow_out = side_sign * float(np.dot(elbow - target_sh, horiz)) / shoulder_width
    wrist_midline = side_sign * float(np.dot(wrist - shoulder_mid, horiz)) / shoulder_width
    # 只把具有明確幾何證據的情況視為 True，設中性帶避免邊界抖動。
    is_abducted = bool(elbow_out > 0.33)
    is_across_midline = bool(wrist_midline < -0.12)
    outside = rula_outside_work_geometry(
        sh_left, sh_right, hp_left, hp_right, elbow, wrist, target_side)
    # 兩個條件屬同一前臂 RULA +1；跨中線成立時不重複判向外工作。
    if is_across_midline:
        outside = {**outside, 'status': '自動否', 'reason': '已歸類為跨越中線（不重複判向外）'}
    # 單張影像無法判斷＝「待複核」；此處僅輸出暫算的 False，正式採信必須人工確認。
    is_out_to_side = bool(outside['status'] == '自動是')
    # 前臂與同側髖的可見度必須比其他正面條件更嚴格；否則不能正面自動加分。
    outside_vis = min(float(lm[i].visibility) for i in (sh_idx, el_idx, wr_idx, 23, 24))
    if outside_vis < 0.55 and not is_across_midline:
        outside['status'] = '待複核'
        outside['reason'] = f'肘／腕／肩／髖可見度不足（最低 {outside_vis:.2f}）'
        is_out_to_side = False
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
        "outside_work_status": outside['status'],
        "outside_work_reason": outside['reason'],
        "outside_wrist_shoulder_margin": outside.get('wrist_shoulder_margin', np.nan),
        "outside_wrist_hip_margin": outside.get('wrist_hip_margin', np.nan),
        "outside_forearm_lateral_margin": outside.get('forearm_lateral_margin', np.nan),
        "outside_forearm_direction_ratio": outside.get('forearm_direction_ratio', np.nan),
        "is_side_bending": bool(trunk_lateral_angle > 12.0),
        # 已由本批 9 張正面影片核對發現大量假陽性，暫停此項自動加分。
        # 保留代理角度診斷，但正式分數只採人工事先確認/批次修改。
        "is_neck_side_bending": False,
        "front_keypoint_visibility": float(np.mean(_visibility_values(results, required)) * 100.0),
        "front_rotation_warning": rotation_warning,
        "front_auto_paused": False,
        "front_screening_reason": "正面幾何品質初步通過（非準確率保證）",
        "cross_midline_normalized": round(wrist_midline, 3),
        "trunk_lateral_angle_proxy": round(trunk_lateral_angle, 1),
        "neck_lateral_angle_proxy": round(neck_relative_angle, 1),
    }
    return True, mods

# ===== v11.4: 側視角同側「耳－肩－髖」頸部向量（候選影像投影角） =====
V114_NECK_METHOD = (
    "RULA v11.4 頸部同側參考線：目標側耳 E / 肩 S / 髖 H。\n"
    "同側頭頸角 = abs(180 - angle(H,S,E))。新舊角度都是 2D 視角投影代理值，\n"
    "不代表經量角器或標定的頸椎解剖角度。正面或側面視角不佳、頭部旋轉、\n"
    "肩耳髖遮擋都可能造成失真；僅有可見度、比例與幾何品質達標才允許採用。\n"
    "優先使用同側角計分，三點品質不足時以原基準備援並保留診斷原因。\n"
    "RULA Table A/B/C 採固定參考工作表並附逐格測試；角度仍需人工驗證。\n"
)
V114_NECK_MODES = (
    "僅比較（維持原本頸角）",
    "採用同側耳－肩－髖；品質不佳時保留原本角度",
)


def v114_same_side_neck(landmarks, width, height, target_side):
    """同側耳(7/8)、肩(11/12)、髖(23/24) 的 2D 相對角度。

    頭部轉向、側視投影、姿態遮擋均可能讓幾何角不等於頸椎屈曲。
    只有三點 visibility、相對距離和幾何均可接受才回傳 valid=True。
    不自動使用對側耳／肩來補缺，以免再次引入錯誤基準。
    """
    side = "右手側" if target_side == "右手側" else "左手側"
    ids = (8, 12, 24) if side == "右手側" else (7, 11, 23)
    base = {"angle": float("nan"), "quality": 0.0, "valid": False,
            "note": "", "landmarks": f"同側耳{ids[0]}-肩{ids[1]}-髖{ids[2]}",
            "ear": None, "shoulder": None, "hip": None,
            "ratio": float("nan"), "ear_visibility": 0.0,
            "shoulder_visibility": 0.0, "hip_visibility": 0.0}
    try:
        pts = [np.array([float(landmarks[i].x) * float(width),
                         float(landmarks[i].y) * float(height)], dtype=float) for i in ids]
        visibility = [float(getattr(landmarks[i], "visibility", 0.0) or 0.0) for i in ids]
        if not all(np.all(np.isfinite(pt)) for pt in pts):
            base["note"] = "同側關鍵點座標無效"
            return base
        ear, shoulder, hip = pts
        base.update(ear=ear, shoulder=shoulder, hip=hip,
                    ear_visibility=round(visibility[0], 3),
                    shoulder_visibility=round(visibility[1], 3),
                    hip_visibility=round(visibility[2], 3),
                    quality=round(min(visibility), 3))
        torso_len = float(np.linalg.norm(shoulder - hip))
        head_len = float(np.linalg.norm(ear - shoulder))
        if torso_len < 28.0 or head_len < 8.0:
            base["note"] = "同側肩髖或肩耳距離太短，無法穩定量角"
            return base
        ratio = head_len / torso_len
        base["ratio"] = round(ratio, 3)
        if not 0.08 <= ratio <= 0.92:
            base["note"] = "同側耳肩距離與軀幹比例不合理"
            return base
        theta = calculate_2d_angle(hip, shoulder, ear)
        if not np.isfinite(theta):
            base["note"] = "同側向量無效"
            return base
        angle = float(abs(180.0 - theta))
        base["angle"] = round(angle, 2)
        if min(visibility) < 0.55:
            base["note"] = "同側耳／肩／髖可見度不足（只保留候選角）"
        elif angle > 85.0:
            base["note"] = "投影相對角極端，請檢查關鍵點及拍攝方向"
        else:
            base["valid"] = True
            base["note"] = "同側耳－肩－髖向量可用；仍為 2D 投影代理角"
        return base
    except (IndexError, ValueError, TypeError, AttributeError, FloatingPointError) as exc:
        base["note"] = f"同側幾何計算失敗（{type(exc).__name__}）"
        return base


def v114_overlay_neck_lines(image, candidate, legacy_angle=None, in_use=False):
    """只在有真實關鍵點時畫線；虛線延伸軀幹軸，讓測量基準可檢查。"""
    if image is None:
        return image
    out = image.copy()
    ear, shoulder, hip = (candidate.get(k) for k in ("ear", "shoulder", "hip"))
    if any(pt is None for pt in (ear, shoulder, hip)):
        return out
    def ip(pt):
        return tuple(int(round(float(v))) for v in pt)
    col = (30, 220, 100) if candidate.get("valid") else (40, 140, 255)
    # 青色：肩到髖的軀幹方向；黃色：肩到耳的頭頸方向。
    cv2.line(out, ip(hip), ip(shoulder), (255, 200, 70), 3, cv2.LINE_AA)
    cv2.line(out, ip(shoulder), ip(ear), (40, 235, 235), 3, cv2.LINE_AA)
    vec = np.asarray(shoulder) - np.asarray(hip)
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        ref = np.asarray(shoulder) + (vec/norm) * max(20.0, min(85.0, norm*0.4))
        # 虛線表示軀幹延長線，避免與真實骨架混淆。
        for v in np.linspace(0.0, 0.9, 6):
            p1 = np.asarray(shoulder) + (ref - shoulder) * v
            p2 = np.asarray(shoulder) + (ref - shoulder) * min(1.0, v+0.09)
            cv2.line(out, ip(p1), ip(p2), (255, 200, 70), 2, cv2.LINE_AA)
    for pt, label in ((hip, "H"), (shoulder, "S"), (ear, "E")):
        cv2.circle(out, ip(pt), 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, ip(pt), 4, col, -1, cv2.LINE_AA)
        cv2.putText(out, label, (ip(pt)[0] + 7, ip(pt)[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2, cv2.LINE_AA)
    angle = candidate.get("angle", float("nan"))
    if np.isfinite(angle):
        status = "USED" if in_use else ("COMPARE" if candidate.get("valid") else "LOW-QC")
        text = f"Neck E-S-H {angle:.1f} deg [{status}]"
    else:
        text = "Neck E-S-H unavailable"
    if legacy_angle is not None and np.isfinite(legacy_angle):
        text += f" | baseline {legacy_angle:.1f}"
    x = max(5, min(out.shape[1] - 150, ip(shoulder)[0] + 10))
    y = max(22, min(out.shape[0] - 6, ip(shoulder)[1] + 30))
    cv2.putText(out, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, col, 1, cv2.LINE_AA)
    return out


def v114_neck_report(*dataframes):
    columns = ['樣本編號', '時間(秒)', '評估側', '頸部_v7_原角度',
               '頸部_v114_同側角度', '頸部_v114_品質', '頸部_v114_品質通過',
               '頸部_v114_關鍵點', '頸部_v114_距離比例', '頸部_v114_原新差值',
               '頸部_v114_診斷', '頸部_v114_計分模式', '頸部_v114_計分來源',
               '頸部_v114_舊版GrandScore', '頸部_v114_同側情境GrandScore',
               '頸部_v114_情境總分差', '頸部角度', '頸部分數']
    nonempty = [df for df in dataframes if df is not None and not df.empty]
    if not nonempty:
        return pd.DataFrame(columns=columns)
    combined = pd.concat(nonempty, ignore_index=True)
    return combined[[c for c in columns if c in combined.columns]].copy()


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
    # 2026 結案修正：本批驗證 9 張正面推估側彎均無明顯目視依據。
    # 頸部側彎不再無條件自動 +1，僅依分析前明確設定與之後批次修正。
    front_mods['is_neck_side_bending'] = bool(manual_front.get('is_neck_side_bending', False))
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
    # 拇指／食指／小指和遠側骨架易遮擋：品質低時使用 Hand 或單側軀幹幾何備援，
    # 不因為某個手指或世界座標缺失就整張照片剔除。
    required = [t_sh, t_el, t_wr, t_hp]
    side_pose_start = time.perf_counter()
    results, _, _ = detect_pose_with_fallback(frame_work, pose_model, required)
    side_pose_ms = round((time.perf_counter() - side_pose_start) * 1000.0, 1)
    if not results or not getattr(results, "pose_landmarks", None):
        return False, None, frame_work, 0.0

    n_lm = results.pose_landmarks.landmark
    w_lm = results.pose_world_landmarks.landmark if getattr(results, 'pose_world_landmarks', None) else None
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
            if w_lm is None:
                return np.array([float('nan')]*3, dtype=float)
            return np.array([w_lm[idx].x, w_lm[idx].y, w_lm[idx].z], dtype=float)

        key_vis = _visibility_values(results, required)
        conf_side = float(np.mean(key_vis) * 100.0) if key_vis else 0.0

        # 肩關節與前臂角度使用「評估側真實肩點」，避免雙肩加權造成關節頂點偏移。
        # 軀幹與頭頸則用肩／髖中心估算影像投影；不能當作解剖學脊椎角度。
        sh, hp = pt2d(t_sh), pt2d(t_hp)
        # 遠側肩／髖被遮擋時，直接使用清晰的目標側關節；避免錯誤中點拉偏角度。
        sh_center = ((pt2d(t_sh) + pt2d(o_sh)) * 0.5
                     if float(n_lm[o_sh].visibility) >= 0.35 else pt2d(t_sh))
        hp_center = ((pt2d(t_hp) + pt2d(o_hp)) * 0.5
                     if float(n_lm[o_hp].visibility) >= 0.35 else pt2d(t_hp))
        ear = pt2d(t_ear) if n_lm[t_ear].visibility >= n_lm[o_ear].visibility else pt2d(o_ear)
        eye_idx = 2 if n_lm[2].visibility >= n_lm[5].visibility else 5
        visible_eye = pt2d(eye_idx)
        # 只在耳與眼屬同側時估測頸部後仰，避免左右臉部點混用。
        face_pair_valid = bool((ear is not None) and
             ((eye_idx == 2 and float(n_lm[t_ear].visibility) >= float(n_lm[o_ear].visibility) and t_ear == 8) or
              (eye_idx == 5 and float(n_lm[t_ear].visibility) >= float(n_lm[o_ear].visibility) and t_ear == 7) or
              (eye_idx == 2 and float(n_lm[o_ear].visibility) > float(n_lm[t_ear].visibility) and o_ear == 8) or
              (eye_idx == 5 and float(n_lm[o_ear].visibility) > float(n_lm[t_ear].visibility) and o_ear == 7)))

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
        raw_wrist_3d = (abs(180.0 - calculate_3d_angle(w3_el, w3_wr, palm_mid_3d))
                        if w_lm is not None else float('nan'))
        # v10：舊版混合角只為研究／回歸比較；絕不再參與正式計分。
        pose_legacy_mixed_angle = (
            float(raw_wrist_2d * 0.6 + raw_wrist_3d * 0.4)
            if np.isfinite(raw_wrist_2d) and np.isfinite(raw_wrist_3d) else float("nan")
        )
        # 若 Hand 失敗，僅允許 Pose 2D 備援，不允許 Pose World 3D 補上無效值。
        ang_wrist = float(raw_wrist_2d)
        wrist_raw_angle = ang_wrist
        wrist_angle_flag = bool(np.isfinite(ang_wrist) and ang_wrist > 60.0)

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
        pose_wrist_angle = float(raw_wrist_2d)  # v10 正式的 Pose 備援值
        hand_angle = float(hand_comp["angle_2d"])
        hand_diff = abs(hand_angle - pose_legacy_mixed_angle) if np.isfinite(hand_angle) and np.isfinite(pose_legacy_mixed_angle) else float("nan")
        pose_2d_hand_diff = abs(hand_angle - raw_wrist_2d) if np.isfinite(hand_angle) and np.isfinite(raw_wrist_2d) else float("nan")
        hand_quality_ok, hand_quality_problems = evaluate_hand_geometry(
            hand_comp, raw_wrist_2d, float(n_lm[t_wr].visibility), float(n_lm[t_el].visibility)
        )
        hand_comp["quality_notes"] = "；".join(hand_quality_problems) if hand_quality_problems else "Hand 幾何檢查通過（仍非角度準確率證明）"
        # v10 強制 Hand 優先，不再提供比較／實驗模式切換。
        hand_used_for_score = bool(hand_quality_ok and np.isfinite(hand_angle))
        if hand_used_for_score:
            ang_wrist = hand_angle
        elif np.isfinite(raw_wrist_2d):
            ang_wrist = float(raw_wrist_2d)
        else:
            # 兩種影像角度都不足時跳過本影格，避免把缺值當低風險。
            return False, None, frame_work, conf_side
        wrist_angle_flag = bool((np.isfinite(ang_wrist) and ang_wrist > 60.0)
                                or (np.isfinite(pose_2d_hand_diff) and pose_2d_hand_diff > 25.0)
                                or (not hand_used_for_score))

        # MediaPipe Pose 的拇指／手腕三維夾角不是前臂旋前旋後角，因此腕扭轉與偏移改由使用者目視確認。
        wt_score = int(manual_mods.get("wrist_twist_score", 1))
        is_wrist_deviated = bool(manual_mods.get("wrist_deviated", False))

        vertical_up = np.array([hp_center[0], hp_center[1] - 100.0])
        ang_trunk = calculate_2d_angle(vertical_up, hp_center, sh_center)

        head_center = ear * 0.6 + visible_eye * 0.4
        neck_legacy = abs(180.0 - calculate_2d_angle(hp_center, sh_center, head_center))
        neck_v8 = v8_neck_candidate(n_lm, work_w, work_h)
        neck_v114 = v114_same_side_neck(n_lm, work_w, work_h, target_side)
        neck_mode = manual_mods.get("neck_v114_mode", V114_NECK_MODES[1])
        use_new_neck = bool(neck_mode == V114_NECK_MODES[1] and neck_v114["valid"])
        ang_neck = neck_v114["angle"] if use_new_neck else neck_legacy
        neck_score_source = ("同側耳肩髖（品質通過）" if use_new_neck else
                             "舊版（新角度品質不足，未採用）" if neck_mode == V114_NECK_MODES[1] else
                             "舊版（比較模式）")

        # RULA 腿足支撐與平衡需依實際作業情境確認，不再用兩腳高度差猜測。
        lg_score = int(manual_mods.get("leg_score", 2))
        if not np.isfinite(ang_lower_arm) or not np.isfinite(ang_upper_arm) or not np.isfinite(ang_trunk) or not np.isfinite(ang_neck) or not np.isfinite(ang_wrist):
            return False, None, frame_work, conf_side

        # 有向姿勢候選：鏡頭面向不同／身體左右轉向時，不直接拿畫面 X 差值當前後。
        directional = side_directional_posture(
            pt2d(0), ear, visible_eye, sh_center, hp_center, sh, el,
            float(n_lm[0].visibility),
            max(float(n_lm[t_ear].visibility),float(n_lm[o_ear].visibility)),
            float(n_lm[eye_idx].visibility) if face_pair_valid else 0.0)
        is_extended = bool(manual_mods.get("upper_arm_extended",False))
        is_trunk_extended = bool(manual_mods.get("trunk_extended",False))
        is_neck_extended = bool(manual_mods.get("neck_extended",False))
        auto_extension_used = False
        auto_extension_warning = ""
        if auto_selection.get("upper_arm_extended",True):
            candidate=directional['upper_arm_extended']
            if candidate is not None:
                is_extended=bool(candidate)
                auto_extension_used=True
            else:
                auto_extension_warning="側面朝向／20度附近不確定，沿用人工初始設定"
        if auto_selection.get("trunk_extended",True):
            candidate=directional['trunk_extended']
            if candidate is not None:is_trunk_extended=bool(candidate)
        if auto_selection.get("neck_extended",True):
            candidate=directional['neck_extended']
            if candidate is not None:is_neck_extended=bool(candidate)

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
            is_extended=is_trunk_extended,
        )
        s_nk = get_neck_score(
            ang_neck,
            is_twisted=manual_mods.get("neck_twisted", False),
            is_side_bending=front_mods.get("is_neck_side_bending", False),
            is_extended=is_neck_extended,
        )
        
        sc_a, sc_b, sc_c, sc_d, grand, al = compute_rula_full(
            s_ua, s_la, s_wr, wt_score, s_nk, s_tk, lg_score, m_val, f_val
        )
        # 同一組手腕與其他部位固定，只比較新舊頸部基準對最終分數的影響。
        legacy_neck_rula = compute_rula_full(
            s_ua, s_la, s_wr, wt_score,
            get_neck_score(neck_legacy,
                           is_twisted=manual_mods.get("neck_twisted", False),
                           is_side_bending=front_mods.get("is_neck_side_bending", False),
                           is_extended=is_neck_extended),
            s_tk, lg_score, m_val, f_val
        )
        new_neck_rula = (compute_rula_full(
            s_ua, s_la, s_wr, wt_score,
            get_neck_score(neck_v114["angle"],
                           is_twisted=manual_mods.get("neck_twisted", False),
                           is_side_bending=front_mods.get("is_neck_side_bending", False),
                           is_extended=is_neck_extended),
            s_tk, lg_score, m_val, f_val
        ) if neck_v114["valid"] else None)
        # 同一影格相同的其他關節分數，只替換手腕「屈曲/伸展」部位分數做情境對照。
        # 在 Hand 優先模式下，兩種情境仍可報表比較，原本 RULA 分數不被覆蓋。
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
        frame_annotated = v114_overlay_neck_lines(
            frame_annotated, neck_v114, in_use=use_new_neck)
        
        side_prefix = "右手" if target_side == "右手側" else "左手"
        data = {
            "頸部角度": round(ang_neck, 1),
            "頸部側彎補正": "是 (+1分)" if front_mods.get("is_neck_side_bending") else "否",
            "頸部_v7_原角度": round(neck_legacy, 1),
            "頸部_v114_同側角度": neck_v114["angle"],
            "頸部_v114_品質": neck_v114["quality"],
            "頸部_v114_品質通過": neck_v114["valid"],
            "頸部_v114_關鍵點": neck_v114["landmarks"],
            "頸部_v114_距離比例": neck_v114["ratio"],
            "頸部_v114_原新差值": (round(abs(neck_legacy - neck_v114["angle"]), 1)
                                    if np.isfinite(neck_v114["angle"]) else np.nan),
            "頸部_v114_診斷": neck_v114["note"],
            "頸部_v114_計分模式": neck_mode,
            "頸部_v114_計分來源": neck_score_source,
            "頸部_v114_舊版GrandScore": legacy_neck_rula[4],
            "頸部_v114_同側情境GrandScore": new_neck_rula[4] if new_neck_rula else np.nan,
            "頸部_v114_情境總分差": (new_neck_rula[4] - legacy_neck_rula[4]
                                      if new_neck_rula else np.nan),
            "頸部_v8_候選角度": neck_v8['angle'],
            "頸部_v8_候選品質": neck_v8['quality'],
            "頸部_v8_關鍵點": neck_v8['landmarks'],
            "頸部_v8_診斷": neck_v8['note'],
            "軀幹角度": round(ang_trunk, 1),
            f"{side_prefix}上臂角度": round(ang_upper_arm, 1),
            f"{side_prefix}前臂角度": round(ang_lower_arm, 1),
            f"{side_prefix}手腕角度": round(ang_wrist, 1),
            f"{side_prefix}Pose舊版手腕角度": round(pose_legacy_mixed_angle, 1) if np.isfinite(pose_legacy_mixed_angle) else np.nan,
            f"{side_prefix}PoseHand實驗手腕角度": round(hand_angle, 1) if np.isfinite(hand_angle) else np.nan,
            f"{side_prefix}新舊手腕差異": round(hand_diff, 1) if np.isfinite(hand_diff) else np.nan,
            f"{side_prefix}Pose2D與Hand2D差異": round(pose_2d_hand_diff, 1) if np.isfinite(pose_2d_hand_diff) else np.nan,
            f"{side_prefix}Hand品質原因": hand_comp["quality_notes"],
            f"{side_prefix}手腕計分品質註記": ("Hand 與 Pose 差異大：僅診斷，不否決 Hand" if hand_used_for_score and np.isfinite(pose_2d_hand_diff) and pose_2d_hand_diff > 25 else ("Pose 2D 備援；需留意 Hand 偵測品質" if not hand_used_for_score else "Hand 幾何檢查通過")),
            f"{side_prefix}Hand掌長前臂比": hand_comp["palm_forearm_ratio"],
            f"{side_prefix}Hand對側手匹配差距比": hand_comp["other_hand_margin_ratio"],
            f"{side_prefix}Hand匹配成功": bool(hand_comp["matched"]),
            f"{side_prefix}Pose2D備援GrandScore": score_pose_ref[4],
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
            f"{side_prefix}手腕計分演算法": "Hand 2D（優先）" if hand_used_for_score else "Pose 2D（備援）",
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
            f"{side_prefix}前臂向外側工作補正": (
                "是 (+1分)" if front_mods.get("is_out_to_side") else
                ("待複核（暫按 0 補正；不代表確定沒有）" if front_mods.get('outside_work_status') == '待複核'
                 else "否（影像代理判斷／人工設定）")),
            "外側工作判定狀態": ("人工設定" if not auto_selection.get('is_out_to_side', True)
                                     else front_mods.get('outside_work_status', '來源未記錄')),
            "外側工作判定說明": front_mods.get('outside_work_reason', ''),
            "外側工作_手腕超肩比例": front_mods.get('outside_wrist_shoulder_margin', np.nan),
            "外側工作_手腕超髖比例": front_mods.get('outside_wrist_hip_margin', np.nan),
            "外側工作_前臂外伸比例": front_mods.get('outside_forearm_lateral_margin', np.nan),
            "外側工作_前臂方向比例": front_mods.get('outside_forearm_direction_ratio', np.nan),
            "軀幹側彎補正": "是 (+1分)" if front_mods.get("is_side_bending") else "否（影像代理判斷）",
            "軀幹扭轉補正": "是 (+1分)" if manual_mods.get("trunk_twisted") else "否（人工確認）",
            "軀幹向後伸展": "是（軀幹基礎分 2）" if is_trunk_extended else "否",
            "頸部扭轉補正": "是 (+1分)" if manual_mods.get("neck_twisted") else "否（人工確認）",
            "頸部後仰／伸展": "是（頸部基礎分 4）" if is_neck_extended else "否",
            "自動辨識設定": ", ".join(k for k, v in auto_selection.items() if v) or "未啟用",
            "正面鏡頭轉向提醒": bool(front_mods.get("front_rotation_warning", False)),
            "正面補正來源": ("沿用分析前人工設定（正面不可靠／缺席）" if front_mods.get("front_auto_paused")
                              else "正面投影估測＋頸側彎人工設定"),
            "正面補正品質說明": front_mods.get("front_screening_reason", ""),
            "跨越中線標準化位移": front_mods.get("cross_midline_normalized", np.nan),
            "軀幹側彎代理角度": front_mods.get("trunk_lateral_angle_proxy", np.nan),
            "頸部側彎代理角度": front_mods.get("neck_lateral_angle_proxy", np.nan),
            "上臂後伸自動判定成功": auto_extension_used,
            "上臂後伸判定備註": auto_extension_warning,
            "有向姿勢參考_面朝":directional["face_side"],
            "有向姿勢參考_軀幹角度":directional["trunk_signed_deg"],
            "有向姿勢參考_上臂角度":directional["upper_arm_signed_deg"],
            "有向姿勢參考_頸部候選角度":directional["neck_proxy_deg"],
            "有向姿勢參考_診斷":directional["diagnostic"],
            "軀幹後伸自動判定":directional["trunk_extended"] if auto_selection.get("trunk_extended",True) else None,
            "頸部後仰自動判定":directional["neck_extended"] if auto_selection.get("neck_extended",True) else None,
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
                         "pose混合": pose_legacy_mixed_angle, "hand2d": hand_angle,
                         "上臂": ang_upper_arm, "前臂屈曲": ang_lower_arm,
                         "軀幹2d投影": ang_trunk, "頸部2d代理": ang_neck,
                         "頸部v7舊角": neck_legacy, "頸部同側E-S-H": neck_v114["angle"]},
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



# =============== 影片原始秒數：肌肉使用／負重區間設定 ===============
# 用側面參考鏡頭的原始影片秒數，和同步事件(正負秒)無關。
# 此處只設定使用者掌握的作業條件，不從單張骨架臆測重量或重複頻率。
def rula_timeline_rules(values, duration=None):
    if not bool(values.get('timeline_enabled',False)):
        return []
    try:
        count=int(values.get('timeline_count',1))
        if not 1<=count<=24:
            raise ValueError('時段數量必須介於 1～24 組')
        rules=[]
        for i in range(count):
            start=float(values.get(f'timeline_{i}_start',0.0))
            end=float(values.get(f'timeline_{i}_end',5.0 if i==0 else 0.0))
            m=int(values.get(f'timeline_{i}_muscle',0))
            f=int(values.get(f'timeline_{i}_force',0))
            title=str(values.get(f'timeline_{i}_title',f'時段{i+1}') or f'時段{i+1}').strip()[:48]
            if not all(np.isfinite(x) for x in (start,end)):
                raise ValueError(f'第 {i+1} 組時間不是有效數字')
            if start<0 or end<=start:
                raise ValueError(f'第 {i+1} 組必須是「結束秒數大於開始秒數」')
            if duration is not None and end>float(duration)+0.08:
                raise ValueError(f'第 {i+1} 組結束時間 {end:g} 秒超過參考側面影片長度 {float(duration):.2f} 秒')
            if m not in (0,1) or f not in (0,1,2,3):
                raise ValueError(f'第 {i+1} 組 RULA 分數範圍不正確')
            rules.append({'name':title,'start':start,'end':end,'muscle':m,'force':f})
        rules.sort(key=lambda x:x['start'])
        for a,b in zip(rules,rules[1:]):
            if b['start']<a['end']-1e-8:
                raise ValueError(f'「{a["name"]}」和「{b["name"]}」時間重疊，請修改成不重複的範圍')
        return rules
    except (TypeError,OverflowError) as exc:
        raise ValueError('時段資料不正確，請重新輸入秒數和分數') from exc


def rula_timeline_at(video_second, rules, default_muscle, default_force):
    """[start, end)；最後一組在實際尾端仍使用 [start,end]，無重疊歧義。"""
    second=float(video_second)
    for i,rule in enumerate(rules):
        if rule['start']-1e-6<=second<rule['end']-1e-6 or (i==len(rules)-1 and abs(second-rule['end'])<=1e-3):
            return int(rule['muscle']),int(rule['force']),f"影片時段：{rule['name']}（{rule['start']:g}～{rule['end']:g}秒）"
    return int(default_muscle),int(default_force),'分析前的預設作業條件'


def rula_timeline_video_fields(video_second, rules, default_muscle, default_force):
    m,f,source=rula_timeline_at(video_second,rules,default_muscle,default_force)
    return {'參考側影片時間(秒)':round(float(video_second),3),'作業條件設定來源':source,
            '肌肉使用加分':m,'負重施力加分':f}

# =============== 影片時間軸：雙端點拖曳＋多區間，自訂互動元件 ===============
# 採用 Streamlit Components v1 的雙向通訊（純 HTML/JS，無 CDN 或額外 pip 套件）。
# 上傳影片內容僅在使用者工作階段傳給瀏覽器，不寫入公開靜態資料夾。
_TIMELINE_COMPONENT = None
_TIMELINE_MAX_VIDEO_BYTES = 24 * 1024 * 1024


def rula_interactive_timeline_component():
    global _TIMELINE_COMPONENT
    if _TIMELINE_COMPONENT is None:
        component_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'timeline_component')
        if not os.path.isfile(os.path.join(component_dir, 'index.html')):
            raise FileNotFoundError('找不到 timeline_component/index.html，請部署完整 ZIP，而非只上傳 222.py。')
        _TIMELINE_COMPONENT = components.declare_component('rula_drag_timeline_20261030', path=component_dir)
    return _TIMELINE_COMPONENT


@st.cache_data(show_spinner=False, max_entries=6, ttl=3600)
def rula_timeline_thumbnails(data, n=12):
    """縮圖一次讀取，避免每次 Streamlit rerun 都重做影片分析。"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        cap = cv2.VideoCapture(path)
        thumbs = []
        try:
            if not cap.isOpened():
                return []
            cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if cnt < 1:
                return []
            for i in range(n):
                index = int(round((i + 0.5) * (cnt - 1) / n))
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    thumbs.append('')
                    continue
                h, w = frame.shape[:2]
                sw = min(w, 200)
                if w > sw:
                    frame = cv2.resize(frame, (sw, max(1,int(h*sw/w))))
                ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 57])
                thumbs.append(base64.b64encode(encoded.tobytes()).decode('ascii') if ok else '')
            return thumbs
        finally:
            cap.release()
    finally:
        os.unlink(path)


def rula_timeline_segments_from_values(values):
    """從原有時段欄位讀取，讓舊版設定仍能出現在新版拖曳介面。"""
    if not values.get('timeline_enabled', False):
        return []
    segments = []
    for i in range(max(0, min(24, int(values.get('timeline_count', 0))))):
        segments.append({
            'title': str(values.get(f'timeline_{i}_title', f'時段{i+1}')),
            'start': float(values.get(f'timeline_{i}_start', 0)),
            'end': float(values.get(f'timeline_{i}_end', 0)),
            'muscle': int(values.get(f'timeline_{i}_muscle', 0)),
            'force': int(values.get(f'timeline_{i}_force', 0)),
        })
    return segments


def rula_timeline_apply_interactive(values, segments, duration):
    """先驗證再修改 session 資料；不允許不完整時段覆蓋先前有效設定。"""
    if not isinstance(segments, list) or len(segments) > 24:
        raise ValueError('時段應是最多 24 組的清單')
    cleaned = []
    for i, x in enumerate(segments):
        if not isinstance(x, dict):
            raise ValueError(f'第 {i+1} 組不是有效的時段')
        title = str(x.get('title', x.get('name', f'時段{i+1}'))).strip()[:48]
        a, b = float(x['start']), float(x['end'])
        m, f = int(x['muscle']), int(x['force'])
        if not (np.isfinite(a) and np.isfinite(b)) or a<0 or b-a<0.099:
            raise ValueError(f'第 {i+1} 組的開始或結束秒數不正確')
        cleaned.append({'title': title or f'時段{i+1}', 'start': a, 'end': b, 'muscle': m, 'force': f})
    # 使用既有計分函式完整驗證，包括影片長度、分數、重疊與秒數邊界。
    check = {'timeline_enabled': bool(cleaned), 'timeline_count': len(cleaned)}
    for i, x in enumerate(cleaned):
        for field in ('title', 'start', 'end', 'muscle', 'force'):
            check[f'timeline_{i}_{field}'] = x[field]
    if cleaned:
        rula_timeline_rules(check, duration)
    # 驗證成功後才一次提交。
    values.update(check)
    return rula_timeline_segments_from_values(values)


def rula_timeline_editor(uploaded, duration, values, camera_name):
    """影像可播時使用內嵌播放器；大影片使用 st.video，縮圖時間軸仍可調整。"""
    import hashlib
    data = uploaded.getvalue()
    signature = f'{camera_name}:'+hashlib.sha256(data).hexdigest()+f':{duration:.3f}'
    # 更換影片時不繼承另一支影片的時段（清空邏輯也由 uploader callback 執行）。
    if values.get('timeline_video_signature') not in (None, signature):
        values['timeline_enabled'] = False
        values['timeline_count'] = 0
    values['timeline_video_signature'] = signature
    segments = rula_timeline_segments_from_values(values)
    try:
        thumbs = rula_timeline_thumbnails(data, 12)
    except Exception:
        thumbs = []
    video_name = str(getattr(uploaded, 'name', 'video.mp4')).lower()
    playable_inline = (video_name.endswith(('.mp4','.m4v')) and len(data) <= _TIMELINE_MAX_VIDEO_BYTES)
    video_b64 = base64.b64encode(data).decode('ascii') if playable_inline else ''
    if not playable_inline:
        st.caption('此影片格式或檔案大小不適合直接嵌入拖曳元件：先用下方影片播放器觀看，縮圖時間軸仍可建立多個區間。')
        st.video(uploaded)
    result = rula_interactive_timeline_component()(
        duration=float(duration),
        segments=segments,
        thumbs=thumbs,
        signature=signature,
        video_b64=video_b64,
        mime='video/mp4',
        key='rula_video_timeline_editor_'+camera_name+'_'+signature[:14],
        default=None,
    )
    # 元件只在明確按「儲存全部區間」後回傳，不會拖一半就改寫正式分數。
    if isinstance(result, dict) and result.get('signature') == signature:
        revision = result.get('revision')
        if revision != st.session_state.get('timeline_last_saved_revision'):
            try:
                new_segments = rula_timeline_apply_interactive(values, result.get('segments'), duration)
                st.session_state['timeline_last_saved_revision'] = revision
                st.success(f'已儲存 {len(new_segments)} 組影片時段；分析時會依照片對應的影片秒數自動套用。')
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                st.error('時段尚未儲存：' + str(exc))
    if segments or values.get('timeline_enabled'):
        try:
            rules = rula_timeline_rules(values, duration)
            st.caption(f'目前已儲存 {len(rules)} 組區間；沒有被區間覆蓋的照片會使用上方預設肌肉使用／負重分數。')
        except ValueError as exc:
            st.warning('已儲存的時段需要重新確認：' + str(exc))


# ==================== 最終驗證版：照片級補正與批次確認（不重跑 MediaPipe） ====================
# 原始 DataFrame 只讀；人工勾選僅寫進獨立覆核資料，不覆寫角度或原始系統評分。
BATCH_BOOL_FIELDS = [
 ('abducted','上臂外展'),('across','前臂跨中線'),('out_side','前臂向外工作'),
 ('trunk_side','軀幹側彎'),('neck_side','頸部側彎'),
 ('shoulder_raised','聳肩'),('arm_supported','上臂支撐'),
 ('wrist_deviated','手腕尺橈偏'),('trunk_twisted','軀幹扭轉'),
 ('neck_twisted','頸部扭轉'),('trunk_extended','軀幹後伸'),
 ('neck_extended','頸部後仰'),('arm_extended','上臂後伸')]
BATCH_NUMBER_FIELDS = [('wrist_twist','手腕旋轉分數',1,2),('leg','腿足支撐分數',1,2),
                       ('muscle','肌肉使用加分',0,1),('force','負重／施力加分',0,3)]
BATCH_FIELDS = {a:b for a,b in BATCH_BOOL_FIELDS}
BATCH_FIELDS.update({a:b for a,b,lo,hi in BATCH_NUMBER_FIELDS})


def batch_sides(mode):
    return ['右手側','左手側'] if mode=='左右側同步計算' else [mode]


def batch_source_frames(session):
    mode=session.get('analyzed_mode','')
    if mode=='左右側同步計算':
        return {'右手側':session.get('df_right_only'), '左手側':session.get('df_left_only')}
    return {mode:session.get('df_records')}


def batch_base_profile(row, side):
    pre='右手' if side=='右手側' else '左手'
    def yes(col):
        return str(row.get(col,'否')).startswith('是')
    # 前臂外側工作三態：讀到「待複核」或 Front 失敗時，必須留下 None。
    # 不能用 str.startswith('是') 把未知擅自轉成 False，否則人工修正前
    # 的候選 0 分會在正式 CSV 被誤當成已通過的自動評分。
    outside_status = str(row.get('外側工作判定狀態', ''))
    outside_unknown = outside_status in ('待複核', '人工設定（正面不可用）')
    outside_value = None if outside_unknown else yes(pre+'前臂向外側工作補正')
    return {
       'abducted':yes(pre+'上臂外展補正'), 'across':yes(pre+'橫越中線補正'),
       'out_side':outside_value,
       'trunk_side':yes('軀幹側彎補正'),'neck_side':yes('頸部側彎補正'),
       'shoulder_raised':yes(pre+'肩部抬高補正'),
       'arm_supported':yes(pre+'上臂支撐／倚靠補正'),
       'wrist_deviated':yes(pre+'手腕偏移補正'),
       'trunk_twisted':yes('軀幹扭轉補正'),
       'neck_twisted':yes('頸部扭轉補正'),
       'trunk_extended':yes('軀幹向後伸展'),
       'neck_extended':yes('頸部後仰／伸展'),
       'arm_extended':yes(pre+'上臂向後伸展補正'),
       'wrist_twist':int(row.get(pre+'手腕扭轉分數',1)),
       'leg':int(row.get('腿部分數',1)),
       'muscle':int(row.get('肌肉使用加分',0)),
       'force':int(row.get('負重施力加分',0)),
    }


def batch_update_profiles(profiles, confirmed, selected, changes, action='confirm'):
    """Pure update for checkbox multi-select, safe on reruns. None means expressly unknown."""
    new={k:dict(v) for k,v in profiles.items()}
    updated=set(confirmed)
    for key in selected:
        if action=='reset':
            new.pop(key,None)
            updated.discard(key)
        else:
            current=dict(new.get(key,{}))
            for k,v in changes.items():
                if k not in BATCH_FIELDS:
                    raise ValueError('未知補正欄位: '+str(k))
                if v!='keep':
                    if v is not None and k in dict(BATCH_BOOL_FIELDS) and not isinstance(v,bool):
                        raise ValueError('布林補正型別錯誤: '+k)
                    if v is not None and k in {f for f,_,_,_ in BATCH_NUMBER_FIELDS}:
                        bounds=next((lo,hi) for f,_,lo,hi in BATCH_NUMBER_FIELDS if f==k)
                        if int(v)!=v or not bounds[0]<=int(v)<=bounds[1]:
                            raise ValueError('數值超出範圍: '+k)
                    current[k]=v
            new[key]=current
            updated.add(key)
    return new,updated


def batch_get_profile(row, side, profile_override):
    base=batch_base_profile(row,side)
    for name,value in profile_override.items():
        if name in BATCH_FIELDS:
            base[name]=value
    return base


def batch_recalculate_side(row, side, profile):
    """Use the same production RULA angle scorers, never approximate by adding to Grand Score."""
    pre='右手' if side=='右手側' else '左手'
    values={k:profile[k] for k in BATCH_FIELDS}
    if any(v is None for v in values.values()):
        # Unknown flag: retain original candidate only; caller excludes from trusted summary.
        return dict(row)
    ua=get_upper_arm_score(float(row[pre+'上臂角度']),values['abducted'],
        values['shoulder_raised'],values['arm_supported'],values['arm_extended'])
    la=get_lower_arm_score(float(row[pre+'前臂角度']),values['across'],values['out_side'])
    wrist=get_wrist_score(float(row[pre+'手腕角度']),values['wrist_deviated'])
    neck=get_neck_score(float(row['頸部角度']),values['neck_twisted'],
                         values['neck_side'],values['neck_extended'])
    trunk=get_trunk_score(float(row['軀幹角度']),values['trunk_twisted'],
                           values['trunk_side'],values['trunk_extended'])
    wrist_twist=int(values['wrist_twist']);legs=int(values['leg'])
    muscle=int(values['muscle']);force=int(values['force'])
    a,b,c,d,total,al=compute_rula_full(ua,la,wrist,wrist_twist,neck,trunk,legs,muscle,force)
    rec=dict(row)
    rec['原始自動Grand Score']=row.get(pre+' Grand Score')
    rec['原始自動AL']=row.get(pre+' AL')
    rec.update({pre+'上臂分數':ua,pre+'前臂分數':la,pre+'手腕分數':wrist,
       pre+'手腕扭轉分數':wrist_twist,'頸部分數':neck,'軀幹分數':trunk,
       '腿部分數':legs,'肌肉使用加分':muscle,'負重施力加分':force,
       pre+' Score A':a,'頸軀腿 Score B':b,pre+' Score C':c,'頸軀腿 Score D':d,
       pre+' Grand Score':total,pre+' AL':al,'最終最高分':total,'最終 AL':al})
    descriptions={
      'abducted':pre+'上臂外展補正','across':pre+'橫越中線補正',
      'out_side':pre+'前臂向外側工作補正','trunk_side':'軀幹側彎補正',
      'neck_side':'頸部側彎補正',
      'shoulder_raised':pre+'肩部抬高補正','arm_supported':pre+'上臂支撐／倚靠補正',
      'wrist_deviated':pre+'手腕偏移補正','trunk_twisted':'軀幹扭轉補正',
      'neck_twisted':'頸部扭轉補正','trunk_extended':'軀幹向後伸展',
      'neck_extended':'頸部後仰／伸展','arm_extended':pre+'上臂向後伸展補正'}
    for field,col in descriptions.items():
        rec[col]='是' if values[field] else '否'
    rec[pre+'手腕扭轉']='接近旋轉極限(2分)' if wrist_twist==2 else '中間範圍(1分)'
    rec['足底支撐狀態']='受支撐且平衡' if legs==1 else '未受支撐／不平衡'
    return rec


def batch_effective_frames(side_frames, profiles, confirmed, neck_decisions=None):
    """Rebuild from immutable raw rows; redo dual-view common neck selection after modifiers."""
    frame_map={}; status={}; cases={}
    for side,df in side_frames.items():
        if not isinstance(df,pd.DataFrame) or df.empty:
            frame_map[side]=pd.DataFrame(columns=(list(df.columns) if isinstance(df,pd.DataFrame) else []))
            continue
        records=[]
        for _,row in df.iterrows():
            key=(side,int(row['樣本編號']))
            p=batch_get_profile(row,side,profiles.get(key,{}))
            unknown=[BATCH_FIELDS[k] for k,v in p.items() if k in BATCH_FIELDS and v is None]
            # 只對唯一的外側工作未知進行敏感度判定；其餘未知仍須人工確認。
            only_outside_unknown=(p.get('out_side') is None and
                not any(v is None for k,v in p.items() if k!='out_side'))
            impact=(rula_outside_score_sensitivity(row,side,p) if only_outside_unknown else None)
            insensitive=bool(impact and impact['same_grand'])
            if insensitive:
                st='外側工作條件待辨識（有／無補正總分相同；無需逐張確認）'
                # 重算可能包含其他人工修改；沿用初始的外側工作候選設定只為計算，
                # 不將其陳述為影像辨識真值。兩種情境的 Grand Score 已驗證相同。
                pre='右手' if side=='右手側' else '左手'
                calculation_profile=dict(p)
                calculation_profile['out_side']=str(row.get(pre+'前臂向外側工作補正','否')).startswith('是')
                rec=batch_recalculate_side(row,side,calculation_profile)
                rec[pre+'前臂向外側工作補正']=row.get(pre+'前臂向外側工作補正','待辨識')
                rec['外側工作分數敏感度']='有／無 +1 皆為 RULA '+str(impact['no_outside'][0])
            else:
                st=('無法確認：'+','.join(unknown) if unknown else
                    ('補正已人工確認' if key in confirmed else '採用分析前設定（尚未額外人工修改）'))
                rec=batch_recalculate_side(row,side,p)
            status[key]=st
            # 第二階段的單張／批次修改會覆蓋影片時段值；報表須清楚顯示最終來源。
            edited=profiles.get(key,{})
            if any(field in edited for field in ('muscle','force')):
                rec['作業條件設定來源']=(str(row.get('作業條件設定來源','分析前的預設作業條件'))
                                           +'；本張已人工修改')
            rec['照片補正確認狀態']=st
            rec['照片補正來源']='批次／單張人工修改' if key in confirmed else '分析前設定／逐張自動計算'
            outside_value=p.get('out_side')
            rec['外側工作最終採用值']=('有' if outside_value else '沒有') if outside_value is not None else '未判定'
            rec['外側工作採用來源']=('人工已設定' if 'out_side' in edited else
                                     str(row.get('外側工作判定狀態','分析前設定')))
            records.append(rec)
        frame_map[side]=pd.DataFrame(records)
    if set(frame_map)>={'右手側','左手側'}:
        r=frame_map['右手側'];l=frame_map['左手側']
        if not r.empty and not l.empty:
            right={int(row['樣本編號']):i for i,row in r.iterrows()}
            left={int(row['樣本編號']):i for i,row in l.iterrows()}
            for sid in sorted(set(right)&set(left)):
                ri,li=right[sid],left[sid]
                rr=r.loc[ri].to_dict();ll=l.loc[li].to_dict()
                # Unknown conditions are not converted to false; leave these candidate rows pending.
                if (status[('右手側',sid)].startswith('無法確認') or
                    status[('左手側',sid)].startswith('無法確認')):
                    continue
                # Pre-correction shared neck scores are already a shared value. To check
                # *new* conditions per camera, calculate independent candidates above.
                case=shared_neck_decide_pair(rr,ll,sid,float(rr.get('時間(秒)',0)))
                if case is not None:
                    cases[sid]=case
                    for idx,side,rec,df in [(ri,'右手側',rr,r),(li,'左手側',ll,l)]:
                        for key in ('頸部分數','頸部共用分數','頸部共用狀態','頸部共用來源',
                                    ('右手' if side=='右手側' else '左手')+' Score A',
                                    '頸軀腿 Score B',('右手' if side=='右手側' else '左手')+' Score C',
                                    '頸軀腿 Score D',('右手' if side=='右手側' else '左手')+' Grand Score',
                                    ('右手' if side=='右手側' else '左手')+' AL','最終最高分','最終 AL'):
                            df.at[idx,key]=rec[key]
    return frame_map,status,cases


def rula_outside_audit_status(row, side, profile):
    """Never convert a vision unknown into a posture fact; prioritise only score-changing work."""
    unknown=[k for k,v in profile.items() if v is None]
    if not unknown:
        return {'status':'已判定／人工設定', 'needs_review':False,
                'score_effect':'已指定有／沒有', 'unknown':False}
    if unknown==['out_side']:
        impact=rula_outside_score_sensitivity(row,side,profile)
        if impact.get('known') and impact.get('same_grand'):
            return {'status':'未知但免確認', 'needs_review':False,
                    'score_effect':f"有／無皆 {impact['no_outside'][0]} 分", 'unknown':True}
        return {'status':'需要人工確認', 'needs_review':True,
                'score_effect':impact.get('note','無法計算分數影響'), 'unknown':True}
    return {'status':'其他條件仍需確認', 'needs_review':True,
            'score_effect':'另有未知條件，無法安全免確認', 'unknown':True}


def batch_audit_table(side_frames, profiles, confirmed):
    """The user-facing CSV distinguishes posture uncertainty from score acceptance."""
    records=[]
    for side,df in side_frames.items():
        if not isinstance(df,pd.DataFrame) or df.empty:continue
        for _,row in df.iterrows():
            sid=int(row['樣本編號']);key=(side,sid)
            p=batch_get_profile(row,side,profiles.get(key,{}))
            unknown=[k for k,v in p.items() if v is None]
            if not unknown:
                assessment={'status':'已判定／人工設定','needs_review':False,'score_effect':'已指定有／沒有','unknown':False}
            elif unknown==['out_side']:
                impact=rula_outside_score_sensitivity(row,side,p)
                safe=bool(impact.get('known') and impact.get('same_grand'))
                assessment={'status':'未知但免確認' if safe else '需要人工確認',
                            'needs_review':not safe, 'score_effect':(f"有／無皆 {impact['no_outside'][0]} 分" if safe else impact.get('note','無法計算分數影響')),
                            'unknown':True}
            else:
                assessment={'status':'其他條件仍需確認','needs_review':True,
                            'score_effect':'另有未知條件，無法安全免確認','unknown':True}
            outside=p.get('out_side')
            if outside is None:
                visible=('未判定（總分不變，免確認）' if not assessment['needs_review']
                         else '未判定（可能影響評分，待確認）')
            else:
                visible='有' if outside else '沒有'
            records.append({'樣本編號':sid,'評估側':side,
                '已確認':key in confirmed,
                '補正是否有未知項':assessment['unknown'],
                '外側工作辨識／複核狀態':assessment['status'],
                '外側工作是否需要人工確認':assessment['needs_review'],
                '外側工作對總分影響':assessment['score_effect'],
                '外側工作最終採用值':visible,
                '外側工作採用來源':('人工已設定' if 'out_side' in profiles.get(key,{})
                               else str(row.get('外側工作判定狀態','分析前設定'))),
                '人工修改欄位':','.join(BATCH_FIELDS[k] for k in profiles.get(key,{}) if k in BATCH_FIELDS),
                '原始時段肌肉使用加分':row.get('肌肉使用加分'),
                '原始時段負重施力加分':row.get('負重施力加分'),
                '參考側影片時間(秒)':row.get('參考側影片時間(秒)'),
                '作業條件設定來源':row.get('作業條件設定來源','分析前的預設作業條件'),
                **{BATCH_FIELDS[k]: (visible if k=='out_side'
                    else '未判定（需確認）' if v is None else v)
                   for k,v in p.items() if k in BATCH_FIELDS}})
    return pd.DataFrame(records)


def batch_all_photo_ids(side_frames,side):
    df=side_frames.get(side)
    if not isinstance(df,pd.DataFrame) or df.empty:return []
    return sorted({int(v) for v in df['樣本編號']})


def batch_current_effective(session):
    return batch_effective_frames(batch_source_frames(session),
        session.get('batch_profiles',{}),session.get('batch_confirmed',set()),
        session.get('shared_neck_decisions',{}))


def batch_choose_all(side, ids):
    st.session_state['batch_selected_'+side]=list(ids)

def batch_choose_none(side):
    st.session_state['batch_selected_'+side]=[]

def batch_photo_panel():
    st.subheader('照片補正｜選照片修改並重新計分')
    st.caption('自動評分先行；照片與作業條件不符時，只修改指定影格。需要補正的照片請先看相鄰「待確認照片」分頁。')
    raw=batch_source_frames(st.session_state)
    sides=[s for s in ('右手側','左手側') if s in raw]
    if not sides:
        st.warning('沒有可分析的側別');return
    side=st.radio('評估側別',sides,horizontal=True,key='batch_active_side')
    ids=batch_all_photo_ids(raw,side)
    if not ids:
        st.warning('此側沒有有效姿勢照片；請前往剔除樣本檢查辨識失敗原因。');return
    confirmed=st.session_state.setdefault('batch_confirmed',set())
    profiles=st.session_state.setdefault('batch_profiles',{})
    done=sum((side,sid) in confirmed for sid in ids)
    st.caption(f'{side}：共 {len(ids)} 張可用評分；其中 {done} 張曾進行人工修改。')
    count=st.selectbox('檢視照片',ids,key='batch_preview_id_'+side)
    images=(st.session_state.get('display_right_b64',[]) if side=='右手側' else
            st.session_state.get('display_left_b64',[])) if len(sides)==2 else st.session_state.get('display_images_b64',[])
    import re as _re
    chosen=next(((label,raw_b64) for label,raw_b64 in images
                 if _re.search(r'#'+str(count)+r'(?!\d)',label)),None)
    if chosen:
        st.caption('注意：照片上的 RULA 總分是首次分析時印入圖片的分數；批次補正後請看下方報表，不以照片舊字樣為準。')
        v10_show_image(chosen[1],chosen[0],encoded=True)
    else:
        st.caption('這筆照片不在畫廊，請查看剔除樣本或報告照片。')
    if st.checkbox('展開目前側別全部骨架縮圖',key='batch_thumb_all'):
        cols=st.columns(5)
        for i,(label,img) in enumerate(images):
            with cols[i%5]:v10_show_image(v10_thumbnail_b64(img,max_width=160),label,encoded=True)
    current_df=raw[side]
    with st.expander('目前這張照片採用的評分條件',expanded=False):
        current_row=current_df.loc[current_df['樣本編號']==count].iloc[0]
        cur=batch_get_profile(current_row,side,profiles.get((side,int(count)),{}))
        preview=pd.DataFrame([{'評分條件':BATCH_FIELDS[k],
                 '目前設定':('無法判斷' if v is None else ('有' if v else '沒有') if isinstance(v,bool) else v)}
                 for k,v in cur.items() if k in BATCH_FIELDS])
        st.dataframe(preview,use_container_width=True,hide_index=True)
        st.caption('目前顯示影片時段設定、初始作業條件及自動辨識結果；沒有不同就不必修改。')
    preview_profile=batch_get_profile(current_row,side,profiles.get((side,int(count)),{}))
    if preview_profile.get('out_side') is None:
        impact=rula_outside_score_sensitivity(current_row,side,preview_profile)
        if impact['same_grand']:
            st.info('這張的前臂外側工作仍無法從影像確定，但有／無 +1 都不會改變 RULA 總分，無須為此逐張補正；可選擇抽查。')
        else:
            st.warning('這張的前臂外側工作可能影響 RULA 分數，請在「上肢」補正為有／沒有並套用。'+impact['note'])
    with st.expander('⚡ 單張快速確認：前臂是否向外側工作', expanded=bool(preview_profile.get('out_side') is None)):
        st.caption('只會改目前選取的這張與這個側別，不會把第 1 張等編號寫死在其他新影片。請以實際作業姿勢為準。')
        quick=st.radio('這張的前臂向外側工作', ['保留原設定','有','沒有'],
                       horizontal=True,key='batch_outside_quick_'+side+'_'+str(count))
        if st.button('保存這張的外側工作，重新計分',key='batch_outside_quick_save_'+side+'_'+str(count),
                     disabled=(quick=='保留原設定')):
            new,done=batch_update_profiles(profiles,confirmed,[(side,int(count))],
                                           {'out_side':quick=='有'})
            st.session_state.batch_profiles=new
            st.session_state.batch_confirmed=done
            st.success('已保存人工確認，重新計算本張 RULA 分數。')
            st.rerun()
    selected_ids=st.multiselect('批次選取照片（可多選）',ids,default=[],key='batch_selected_'+side,
            help='只需要選擇與初始條件不同的照片；其他照片不用操作。')
    c1,c2=st.columns(2)
    with c1:
        st.button('全選這一側的有效照片',use_container_width=True,key='batch_select_all',
                  on_click=batch_choose_all,args=(side,ids))
    with c2:
        st.button('清空選取',use_container_width=True,key='batch_select_none',
                  on_click=batch_choose_none,args=(side,))
    with st.expander('快速選取連續照片（例如只改搬運階段）', expanded=False):
        c_from,c_to,c_choose=st.columns([1,1,1])
        with c_from:
            from_id=st.selectbox('起始照片',ids,index=0,key='batch_range_start_'+side)
        with c_to:
            to_id=st.selectbox('結束照片',ids,index=len(ids)-1,key='batch_range_end_'+side)
        with c_choose:
            st.caption('只需點一次')
            if st.button('選取範圍',key='batch_range_set_'+side,use_container_width=True):
                low,high=sorted((int(from_id),int(to_id)))
                st.session_state['batch_selected_'+side]=[i for i in ids if low<=i<=high]
                st.rerun()
    if selected_ids:
        st.caption(f'已選 {len(selected_ids)} 張：'+','.join(map(str,selected_ids[:25]))+('…' if len(selected_ids)>25 else ''))
    with st.form('batch_edit_form'):
        st.markdown('**只修改選中照片有變化的條件**（其餘維持原本的自動／初始設定）')
        changes={}
        groups=[('有正面時可用正面影像輔助的項目',BATCH_BOOL_FIELDS[:5]),
                ('其他隨姿勢改變的補正',BATCH_BOOL_FIELDS[5:])]
        opts=['維持原設定','有','沒有','無法判斷']
        for title,fields in groups:
            with st.expander(title,expanded=False):
                cols=st.columns(2)
                for i,(key,label) in enumerate(fields):
                    choice=cols[i%2].selectbox(label,opts,key='batch_edit_'+key)
                    changes[key]={'維持原設定':'keep','有':True,'沒有':False,'無法判斷':None}[choice]
        st.markdown('**需要按作業階段調整的分數**')
        with st.expander('四種分數的 RULA 判定提示（請先確認作業實際狀況）'):
            st.markdown('**手腕旋轉**：1＝中間範圍；2＝接近旋轉極限。\n\n'
                        '**腿足**：1＝腿足有適當支撐且平衡；2＝無適當支撐或不平衡。\n\n'
                        '**肌肉使用**：0＝不符合靜態／高重複條件；1＝姿勢維持超過 1 分鐘，或重複約每分鐘 4 次以上。\n\n'
                        '**負重／施力**：0＝低於 2kg、間歇負荷；1＝2～10kg 間歇負荷；'
                        '2＝2～10kg 靜態／重複負荷或更大間歇負荷；'
                        '3＝大負荷靜態／重複或衝擊力。請依 RULA 標準工作表和實際受力判定。')
        nums=st.columns(4)
        for i,(key,label,lo,hi) in enumerate(BATCH_NUMBER_FIELDS):
            options=['維持原設定']+[str(v) for v in range(lo,hi+1)]+['無法判斷']
            val=nums[i].selectbox(label,options,key='batch_edit_'+key)
            changes[key]='keep' if val=='維持原設定' else None if val=='無法判斷' else int(val)
        st.caption('肌肉使用依整體作業重複頻率／靜態維持判斷；負重按實際施力階段調整。'
                   '僅變更你有把握的條件；若無法判斷可選「無法判斷」，不會強制當作沒有。')
        acknowledge=st.checkbox('我確認要把這些變更套用到所選照片')
        submit=st.form_submit_button('套用所選修改',type='primary')
    if submit:
        if not selected_ids:
            st.error('請先選取至少一張照片。')
        elif not acknowledge:
            st.error('請先確認已查看照片與補正條件。')
        else:
            keys=[(side,int(n)) for n in selected_ids]
            updated,checked=batch_update_profiles(profiles,confirmed,keys,changes)
            st.session_state.batch_profiles=updated
            st.session_state.batch_confirmed=checked
            st.success(f'已保存 {len(keys)} 張的補正，重新計算 RULA 評分。')
            st.rerun()
    if selected_ids and st.button('撤銷選中照片的批次確認／修正',key='batch_reset_selected'):
        keys=[(side,int(n)) for n in selected_ids]
        new,done=batch_update_profiles(profiles,confirmed,keys,{},action='reset')
        st.session_state.batch_profiles=new
        st.session_state.batch_confirmed=done
        st.rerun()
    effective,status,cases=batch_current_effective(st.session_state)
    statuses=[]
    for sid in ids:
        row=effective[side]
        one=row[row['樣本編號']==sid].iloc[0]
        statuses.append({'照片編號':sid,'補正狀態':status[(side,sid)],
            '目前系統評分':one.get(('右手' if side=='右手側' else '左手')+' Grand Score'),
            '原始分析分數':raw[side].loc[raw[side]['樣本編號']==sid,('右手' if side=='右手側' else '左手')+' Grand Score'].iloc[0]})
    with st.expander('查看所有照片補正確認狀態',expanded=False):
        st.dataframe(pd.DataFrame(statuses),hide_index=True,use_container_width=True)
    if cases:
        pending=[sid for sid,c in cases.items() if c['需要人工確認'] and sid not in st.session_state.get('shared_neck_decisions',{})]
        if pending:st.warning(f'完成雙側補正後，有 {len(pending)} 張仍需確認共同頸部分數；請至「異常／人工覆核 → 頸部確認」。')

# ==================== 【V11.3：品質提醒也能逐張確認／人工計分／剔除】 ====================
V113_KEEP = '維持／確認使用自動評分'
V113_MANUAL = '改用人工重新評分'
V113_EXCLUDE = '剔除此側樣本'
V113_PENDING = '尚未確認'


def v113_key(sample, side):
    return f'{int(sample)}_{side}'


def v113_dataframe_for_side(side, side_frames):
    frame = (side_frames or {}).get(side)
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def v113_get_sample_row(side, sample, side_frames):
    df = v113_dataframe_for_side(side, side_frames)
    if df.empty or '樣本編號' not in df.columns:
        return None
    rows = df[pd.to_numeric(df['樣本編號'], errors='coerce') == int(sample)]
    return None if rows.empty else rows.iloc[0]


def v113_record_status(sample, side, source_row, decisions, overrides, flagged_keys, notice_keys, legacy=None):
    """Decision of one side; missing data is never automatically accepted."""
    key = v113_key(sample, side)
    decision = decisions.get(key)
    override = overrides.get((side, int(sample)))
    flagged = key in flagged_keys
    is_notice = key in notice_keys
    if decision == V113_EXCLUDE:
        return '人工剔除', None, None
    if decision == V113_MANUAL:
        if override is None:
            return '等待人工重新評分', None, None
        return '人工評分已完成', int(override['人工Grand Score']), str(override['人工AL'])
    if override is not None and decision is None:
        return '人工評分已完成', int(override['人工Grand Score']), str(override['人工AL'])
    if source_row is None:
        return '辨識失敗／無原始分數', None, None
    if decision == V113_KEEP:
        return '已人工確認採用自動評分', int(source_row['右手 Grand Score' if side=='右手側' else '左手 Grand Score']), str(source_row['右手 AL' if side=='右手側' else '左手 AL'])
    if flagged and not is_notice:
        if (legacy or {}).get(key) == '已目視確認關鍵點／分數（僅紀錄）':
            return '已人工確認採用自動評分', int(source_row['右手 Grand Score' if side=='右手側' else '左手 Grand Score']), str(source_row['右手 AL' if side=='右手側' else '左手 AL'])
        return '待複核（需人工檢查後採用）', None, None
    return ('一般提醒：預設保留自動評分' if is_notice else '自動品質通過'), int(source_row['右手 Grand Score' if side=='右手側' else '左手 Grand Score']), str(source_row['右手 AL' if side=='右手側' else '左手 AL'])


def v113_build_reviewed_data(side_frames, items, decisions, overrides, legacy=None, mode=None, neck_cases=None, neck_decisions=None, batch_status=None):
    """Produce reviewed side + combined reports without touching original automation data.

    If dual mode and only one side is valid, preserve it as partial-side evidence but
    do not claim a valid two-side maximum.
    """
    flagged_keys = {v113_key(item['樣本編號'],item['評估側']) for item in items if not v112_is_informational_notice(item)}
    notice_keys = {v113_key(item['樣本編號'],item['評估側']) for item in items if v112_is_informational_notice(item)}
    sides = [side for side in ('右手側', '左手側') if side in side_frames]
    sampled = {side: {} for side in sides}
    for side in sides:
        df = v113_dataframe_for_side(side,side_frames)
        if not df.empty:
            for _,r in df.iterrows():
                sampled[side][int(r['樣本編號'])]=r
    allids=set()
    for side in sides:
        allids.update(sampled[side].keys())
    allids.update(int(item['樣本編號']) for item in items if item.get('評估側') in sampled)
    allids.update(int(sid) for (side,sid) in overrides if side in sampled)
    allids.update(int(key.split('_',1)[0]) for key in decisions if key.split('_',1)[-1] in sampled)
    allids.update(int(sid) for sid in (neck_cases or {}).keys())
    rows=[]
    for sid in sorted(allids):
        neck_case=(neck_cases or {}).get(sid)
        neck_choice=(neck_decisions or {}).get(sid)
        for side in sides:
            original=sampled[side].get(sid)
            status,grand,al=v113_record_status(sid,side,original,decisions,overrides,flagged_keys,notice_keys,legacy)
            batch_state = (batch_status or {}).get((side,sid),'補正已確認')
            if batch_state.startswith('無法確認') and grand is not None and (side,sid) not in overrides:
                status,grand,al=('待釐清補正（請在異常／人工覆核選「有」或「沒有」）',None,None)
            elif batch_state == '補正已人工確認' and status in ('自動品質通過','一般提醒：預設保留自動評分'):
                status='照片補正已人工確認／品質篩檢通過'
            elif batch_state == '採用分析前設定（尚未額外人工修改）' and status in ('自動品質通過','一般提醒：預設保留自動評分'):
                status='使用預設設定的系統評分（未逐張人工確認）'
            if original is not None and neck_case is not None:
                if neck_case['需要人工確認'] and neck_choice is None and grand is not None:
                    # 不讓使用者在別處「維持自動評分」繞過共同頸部分級的待確認狀態。
                    status,grand,al='待頸部局部人工確認',None,None
                elif grand is not None:
                    # 不論「自動共用」或「人工確認」，都必須重新帶入共同頸分查表。
                    # 不能只有 CSV 的採用頸部分數改了，Grand Score 卻保留單側舊值。
                    adopted_neck=(int(neck_choice) if neck_choice is not None
                                  else int(neck_case['選定共用頸部分數']))
                    manual=overrides.get((side,sid))
                    try:
                        grand,al=shared_neck_reviewed_score(original,side,adopted_neck,manual)
                        if neck_choice is not None:
                            status='已確認共同頸部分數'+('／保留其他人工修改' if manual else '')
                        elif manual:
                            status='系統自動共用頸分／保留其他人工修改'
                    except (KeyError,TypeError,ValueError):
                        status,grand,al='頸部重新計分失敗，需人工檢查',None,None
            rows.append({
                '樣本編號':sid,'評估側':side,
                '原始Grand Score':(original.get('原始自動Grand Score',original.get('右手 Grand Score' if side=='右手側' else '左手 Grand Score')) if original is not None else None),
                '頸部角度(°)':(original.get('頸部角度') if original is not None else None),
                '頸部角度計算來源':(original.get('頸部_v114_計分來源','') if original is not None else ''),
                '照片補正狀態':batch_state,
                '參考側影片時間(秒)':(original.get('參考側影片時間(秒)') if original is not None else None),
                '作業條件設定來源':(original.get('作業條件設定來源','') if original is not None else ''),
                '肌肉使用加分':(original.get('肌肉使用加分') if original is not None else None),
                '負重施力加分':(original.get('負重施力加分') if original is not None else None),
                '原始AL':(original.get('原始自動AL', original.get('右手 AL' if side=='右手側' else '左手 AL')) if original is not None else None),
                '複核後Grand Score':grand, '複核後AL':al,
                '結果狀態':status,'使用者選擇':decisions.get(v113_key(sid,side),'未額外指定'),
                '是否納入可信統計':grand is not None,  # 技術欄名沿用；畫面須明示含系統評分結果
                '原始品質狀態':(original.get('品質狀態','') if original is not None else '無有效自動評分'),
                '正面補正來源':(original.get('正面補正來源','') if original is not None else '無有效資料'),
                '外側工作最終採用值':(original.get('外側工作最終採用值','') if original is not None else '無有效資料'),
                '外側工作採用來源':(original.get('外側工作採用來源','') if original is not None else '無有效資料'),
                '外側工作自動判定':(original.get('外側工作判定狀態','') if original is not None else '無有效資料'),
                '外側工作判定說明':(original.get('外側工作判定說明','') if original is not None else ''),
                '採用頸部分數':(int(neck_choice) if neck_choice is not None else
                       (neck_case['選定共用頸部分數'] if neck_case and not neck_case['需要人工確認']
                        else (int(original.get('頸部分數')) if original is not None and neck_case is None and original.get('頸部分數') is not None else None))),
                '頸部確認狀態':('待人工確認' if neck_case and neck_case['需要人工確認'] and neck_choice is None else ('人工已確認' if neck_choice is not None else ('系統自動共用' if neck_case else '單側評估'))),
            })
    review_df=pd.DataFrame(rows, columns=['樣本編號','評估側','頸部角度(°)','頸部角度計算來源','原始Grand Score','原始AL','複核後Grand Score','複核後AL','結果狀態','使用者選擇','是否納入可信統計','原始品質狀態','採用頸部分數','頸部確認狀態','照片補正狀態',
                  '參考側影片時間(秒)','作業條件設定來源','肌肉使用加分','負重施力加分','正面補正來源',
                  '外側工作自動判定','外側工作判定說明','外側工作最終採用值','外側工作採用來源'])
    consolidated=[]
    for sid in sorted(allids):
        side_rows={r['評估側']:r for r in rows if r['樣本編號']==sid}
        if len(sides)==2:
            both_valid=all(side_rows[side]['是否納入可信統計'] for side in sides)
            valid=[(side,side_rows[side]) for side in sides if side_rows[side]['是否納入可信統計']]
            label=('完整雙側有效' if both_valid else ('僅單側有效（不可視為雙側最高分）' if valid else '雙側無有效分數'))
            higher=max(valid,key=lambda x:x[1]['複核後Grand Score']) if valid else None
            # Full comparison only if BOTH sides pass, otherwise no published dual score.
            consolidated.append({'樣本編號':sid,'雙側結果狀態':label,
                '右側頸部角度(°)':side_rows['右手側'].get('頸部角度(°)'),
                '左側頸部角度(°)':side_rows['左手側'].get('頸部角度(°)'),
                '右手結果狀態':side_rows['右手側']['結果狀態'],
                '左手結果狀態':side_rows['左手側']['結果狀態'],
                '右手複核後分數':side_rows['右手側']['複核後Grand Score'],
                '左手複核後分數':side_rows['左手側']['複核後Grand Score'],
                '完整雙側最高分':(higher[1]['複核後Grand Score'] if both_valid else None),
                '完整雙側AL':(higher[1]['複核後AL'] if both_valid else None),
                '採用頸部分數':side_rows['右手側'].get('採用頸部分數'),
                '頸部確認狀態':side_rows['右手側'].get('頸部確認狀態'),
                '參考側影片時間(秒)':side_rows['右手側'].get('參考側影片時間(秒)'),
                '作業條件設定來源':side_rows['右手側'].get('作業條件設定來源'),
                '肌肉使用加分':side_rows['右手側'].get('肌肉使用加分'),
                '負重施力加分':side_rows['右手側'].get('負重施力加分'),
                '最高風險側':(('雙側同分' if side_rows['右手側']['複核後Grand Score']==side_rows['左手側']['複核後Grand Score'] else higher[0]) if both_valid else '不可判定'),
                '部分可用側別':('、'.join(side for side,_ in valid) if not both_valid else '')})
        else:
            side=sides[0]
            r=side_rows[side]
            consolidated.append({'樣本編號':sid,'雙側結果狀態':'單側分析','評估側':side,
                '頸部角度(°)':r.get('頸部角度(°)'),
                '結果狀態':r['結果狀態'],'複核後Grand Score':r['複核後Grand Score'],'複核後AL':r['複核後AL'],
                '採用頸部分數':r.get('採用頸部分數'),'頸部確認狀態':r.get('頸部確認狀態'),
                '參考側影片時間(秒)':r.get('參考側影片時間(秒)'),
                '作業條件設定來源':r.get('作業條件設定來源'),
                '肌肉使用加分':r.get('肌肉使用加分'),'負重施力加分':r.get('負重施力加分')})
    return review_df,pd.DataFrame(consolidated)


def v113_current_reports():
    mode=st.session_state.get('analyzed_mode','')
    side_frames=({'右手側':st.session_state.get('df_right_only'),
                  '左手側':st.session_state.get('df_left_only')} if mode=='左右側同步計算'
                 else {mode:st.session_state.get('df_records')})
    effective,status,cases=batch_current_effective(st.session_state)
    return v113_build_reviewed_data(effective,st.session_state.get('v8_rejected_rows',[]),
            st.session_state.get('v113_review_decisions',{}),st.session_state.get('v4_overrides',{}),
            st.session_state.get('v8_manual_rejected',{}),mode,
            cases,st.session_state.get('shared_neck_decisions',{}),status)


def v113_make_manual_score(sample, side, values, original=None, time_s=None):
    """Manual RULA components for a failed or suspect frame; no auto score fabricated."""
    mapping={k:int(values[k]) for k in ('上臂','前臂','手腕','腕扭轉','頸部','軀幹','腿部','肌肉','負重')}
    a,b,c,d,total,al=compute_rula_full(mapping['上臂'],mapping['前臂'],mapping['手腕'],mapping['腕扭轉'],mapping['頸部'],mapping['軀幹'],mapping['腿部'],mapping['肌肉'],mapping['負重'])
    prefix='右手' if side=='右手側' else '左手'
    return {'樣本編號':int(sample),'時間(秒)':float(original['時間(秒)'] if original is not None else (time_s or 0)),
            '評估側':side, '原始Grand Score':(int(original[f'{prefix} Grand Score']) if original is not None else None),
            '人工Grand Score':total,'原始AL':(original[f'{prefix} AL'] if original is not None else None),
            '人工AL':al, '人工Score A':a,'人工Score B':b,'人工Score C':c,'人工Score D':d,
            **{f'人工{k}分數':v for k,v in mapping.items()},
            '異動說明':'使用者人工輸入九項 RULA 分數；不變更原始影像及原始自動分數'}


def v113_manual_form(sample,side,item,source_row):
    prefix='右手' if side=='右手側' else '左手'
    defaults={'上臂':(f'{prefix}上臂分數',1,6),'前臂':(f'{prefix}前臂分數',1,3),
              '手腕':(f'{prefix}手腕分數',1,4),'腕扭轉':(f'{prefix}手腕扭轉分數',1,2),
              '頸部':('頸部分數',1,6),'軀幹':('軀幹分數',1,6),
              '腿部':('腿部分數',1,2),'肌肉':('肌肉使用加分',0,1),
              '負重':('負重施力加分',0,3)}
    key=(side,int(sample))
    previous=st.session_state.get('v4_overrides',{}).get(key,{})
    st.info('請依原圖與 RULA 標準工作表**人工確認各部位分數**。預設值僅供編輯，不表示自動辨識可靠。')
    with st.form(f'v113_manual_{side}_{sample}'):
        cols=st.columns(3)
        entered={}
        for i,(label,(name,low,high)) in enumerate(defaults.items()):
            initial=previous.get(f'人工{label}分數',source_row.get(name,low) if source_row is not None else low)
            try: initial=int(initial)
            except (TypeError,ValueError): initial=low
            entered[label]=cols[i%3].number_input(label+'分數',min_value=low,max_value=high,value=max(low,min(high,initial)),step=1,key=f'v113_{side}_{sample}_{label}')
        acknowledged=st.checkbox('我已人工檢查上述 RULA 各部位條件及分數',key=f'v113_ack_{side}_{sample}')
        submitted=st.form_submit_button('儲存人工評分並採用此結果',type='primary')
    if submitted:
        if not acknowledged:
            st.error('請先確認已依 RULA 工作表完成人工判斷，避免使用預設值當成測量結果。')
        else:
            result=v113_make_manual_score(sample,side,entered,source_row,item.get('時間(秒)'))
            st.session_state.v4_overrides[key]=result
            st.session_state.v113_review_decisions[v113_key(sample,side)]=V113_MANUAL
            st.success(f'已儲存人工評分：{result["人工Grand Score"]} 分（{result["人工AL"]}），將用於複核後報表。')
            st.rerun()
    if previous:
        st.success(f'目前人工評分：{previous["人工Grand Score"]} 分（{previous["人工AL"]}）')
        if st.button('取消這筆人工評分（恢復待處理）',key=f'v113_remove_manual_{side}_{sample}'):
            del st.session_state.v4_overrides[key]
            st.rerun()


def v113_finalize_report_zip(original_zip, side_frames, review_items, decisions, overrides, legacy=None, neck_cases=None, neck_decisions=None, batch_status=None, batch_audit=None, work_conditions=None, checklist=None):
    import json
    """Readable ZIP: clean user CSVs plus optional diagnostic material, original photos untouched."""
    reviewed,combined=v113_build_reviewed_data(side_frames,review_items,decisions,overrides,legacy,neck_cases=neck_cases,neck_decisions=neck_decisions,batch_status=batch_status)
    work_conditions=work_conditions or {}
    checklist=checklist or {}
    out=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original_zip),'r') as source, zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as target:
        included=set()
        for info in source.infolist():
            name=info.filename
            if name.endswith('/') or name.startswith('Final_Reviewed/') or name=='Hand_Comparison/Comparison_Data.csv':
                continue
            # Extra old candidate/duplicate reports are for development and do not belong in user CSVs.
            if name in ('Diagnostics/Neck_Comparison.csv','Diagnostics/V7_Angle_Definitions.txt',
                        'Diagnostics/V11_Quality_Guide.txt','Diagnostics/V11_1_Quality_Guide.txt',
                        'Diagnostics/V11_1_Skeleton_Quality.csv','Diagnostics/V11_2_Review_Explanation.txt',
                        'Diagnostics/V114_Neck_Method.txt','Diagnostics/Rejected_Manual_Reviews.csv'):
                continue
            if name.startswith('Diagnostics/V') and name.endswith('.txt'):
                continue
            rewrite={
                'Diagnostics/V114_Neck_Same_Side_Comparison.csv':'技術診斷/頸部計算方法對照.csv',
                'Diagnostics/V11_Skeleton_Quality.csv':'技術診斷/骨架品質明細.csv',
                'Diagnostics/Skipped_Samples.csv':'技術診斷/辨識失敗.csv',
                'Diagnostics/Rejected_Samples_Detail.csv':'技術診斷/待處理影格.csv',
                'Diagnostics/Quality_Notices.csv':'技術診斷/一般品質提醒.csv',
                'Diagnostics/Keypoint_Coordinates.jsonl':'技術診斷/關鍵點座標.jsonl',
                'Synchronization/Synchronization_Report.csv':'時間同步/鏡頭同步資訊.csv',
                'Synchronization/Frame_Time_Map.csv':'時間同步/逐筆取樣時間.csv',
                'Synchronization/Settings.json':'時間同步/同步設定.json',
                'Diagnostics/V4_Anomaly_Review.csv':'技術診斷/自動異常清單.csv',
                'Diagnostics/V4_Manual_Overrides.csv':'技術診斷/手動修改紀錄.csv',
                'Diagnostics/V4_README.txt':'技術診斷/異常判定說明.txt',
            }
            dest=rewrite.get(name,name)
            if dest.startswith('評估結果/'):
                dest='原始分析/'+dest.split('評估結果/',1)[1]
            if dest.startswith('Hand_Comparison/'):
                dest='照片/手部辨識對照/'+dest.split('Hand_Comparison/',1)[1]
            if dest.startswith('Diagnostics/Rejected_Images/'):
                dest='技術診斷/剔除與待複核照片/'+dest.split('Rejected_Images/',1)[1]
            if dest.startswith('Diagnostics/'):
                # Unrecognized internal files don't go in the visible top-level.
                dest='技術診斷/'+dest.split('Diagnostics/',1)[1]
            if dest in included: continue
            payload=source.read(name)
            if name=='Diagnostics/Rejected_Samples_Detail.csv':
                try:
                    detail_df=pd.read_csv(io.BytesIO(payload),encoding='utf-8-sig')
                    for column in detail_df.columns:
                        if '圖片' in str(column) or '照片' in str(column):
                            detail_df[column]=detail_df[column].apply(lambda v: (
                                v.replace('Diagnostics/Rejected_Images/',
                                          '技術診斷/剔除與待複核照片/') if isinstance(v,str) else v))
                    payload=detail_df.to_csv(index=False).encode('utf-8-sig')
                except (OSError,ValueError,UnicodeError,pd.errors.ParserError):
                    pass
            if name=='Diagnostics/V114_Neck_Same_Side_Comparison.csv':
                try:
                    internal=pd.read_csv(io.BytesIO(payload),encoding='utf-8-sig')
                    rename={
                        '頸部_v7_原角度':'原有頸部角度(°)',
                        '頸部_v114_同側角度':'同側頸部角度(°)',
                        '頸部_v114_品質':'關鍵點品質指標',
                        '頸部_v114_品質通過':'新基準是否可用',
                        '頸部_v114_關鍵點':'採用關鍵點',
                        '頸部_v114_距離比例':'肩耳與肩髖比例',
                        '頸部_v114_原新差值':'兩種基準差(°)',
                        '頸部_v114_診斷':'判定說明',
                        '頸部_v114_計分模式':'角度計分設定',
                        '頸部_v114_計分來源':'實際採用基準',
                        '頸部_v114_舊版GrandScore':'原有基準情境總分',
                        '頸部_v114_同側情境GrandScore':'同側基準情境總分',
                        '頸部_v114_情境總分差':'情境總分差',
                        '頸部角度':'實際採用頸角(°)',
                    }
                    internal=internal.rename(columns=rename)
                    internal=internal.loc[:,[c for c in internal if not ('_v' in c or c.startswith('v'))]]
                    payload=internal.to_csv(index=False).encode('utf-8-sig')
                except (OSError,ValueError,UnicodeError,pd.errors.ParserError):
                    continue
            target.writestr(dest,payload)
            included.add(dest)
        def write_unique(name, content):
            if name not in included:
                target.writestr(name,content)
                included.add(name)
        write_unique('技術診斷/程式與計分版本.json', json.dumps({
            '系統版本': globals().get('APP_BUILD_TAG','RULA_USER_UI_REVIEW_CENTER_2026-10-30'),
            '參考RULA工作表':'2016 蕭世平碩士論文 表3-8/3-9/3-10（與標準 RULA 工作表核對）',
            'Table_A_格數':144,'Table_B_格數':72,'Table_C_格數':81,
            '頸部側彎':'已停用無驗證的自動側彎加分；採作業設定及照片批次修改',
            '正面補正':'旋轉/肩線縮短/肩線傾斜時採預設，不視為高精度 AI 認定',
            '前臂外側工作':'雙參考肩髖正規化＋前臂方向；分數不變者保留未知、免複核；有影響者要求確認',
            '實際作業條件已確認': bool(work_conditions.get('work_conditions_verified',False)),
            '操作注意':'RULA 分數與影像姿勢辨識準確率須另以人工驗證'
        },ensure_ascii=False,indent=2).encode('utf-8'))
        # Separation of field facts and selected scores, plus a concise closure checklist.
        values=work_conditions
        verified=bool(values.get('work_conditions_verified',False))
        write_unique('評估結果/作業條件確認.csv', pd.DataFrame([{
            '現場條件已人工確認':'是' if verified else '否（待確認）',
            '填寫負重重量(kg)':values.get('actual_load_kg',''),
            '填寫重複次數(次/分鐘)':values.get('actual_repetition',''),
            '肌肉使用預設分數':values.get('muscle',''),
            '負重施力預設分數':values.get('force',''),
            '註解':'影像無法自行證明物體重量或工作頻率；需現場資訊支持'}
        ]).to_csv(index=False).encode('utf-8-sig'))
        write_unique('複核後結果/逐側結果.csv',user_report_reviewed(reviewed).to_csv(index=False).encode('utf-8-sig'))
        if batch_audit is not None:
            write_unique('複核後結果/逐張人工補正確認.csv',batch_audit.to_csv(index=False).encode('utf-8-sig'))
            outside_cols=['樣本編號','評估側','外側工作辨識／複核狀態','外側工作是否需要人工確認','外側工作對總分影響','外側工作最終採用值','外側工作採用來源']
            if all(col in batch_audit.columns for col in outside_cols):
                write_unique('複核後結果/外側工作逐筆判定與影響.csv',batch_audit[outside_cols].to_csv(index=False).encode('utf-8-sig'))
        time_rows=[]
        for side,frame in side_frames.items():
            if not isinstance(frame,pd.DataFrame) or frame.empty:continue
            for _,r in frame.iterrows():
                time_rows.append({'樣本編號':int(r['樣本編號']),'評估側':side,
                    '參考側影片時間(秒)':r.get('參考側影片時間(秒)'),
                    '作業條件設定來源':r.get('作業條件設定來源','分析前的預設作業條件'),
                    '最終肌肉使用加分':r.get('肌肉使用加分'),
                    '最終負重施力加分':r.get('負重施力加分'),
                    '人工批次修改':('是' if (batch_status or {}).get((side,int(r['樣本編號'])))=='補正已人工確認' else '否')})
        write_unique('評估結果/逐張時段與作業條件.csv',pd.DataFrame(time_rows).to_csv(index=False).encode('utf-8-sig'))

        # All final reports derive from reviewed single authoritative side result, including
        # the final shared neck override. Don't export stale pre-neck batch scores.
        final_index={(int(x['樣本編號']),x['評估側']):x for _,x in reviewed.iterrows()} if not reviewed.empty else {}
        detailed=[]
        for side,df in side_frames.items():
            if not isinstance(df,pd.DataFrame) or df.empty:continue
            pre='右手' if side=='右手側' else '左手'
            for _,r in df.iterrows():
                sid=int(r['樣本編號']);state=(batch_status or {}).get((side,sid),'補正已確認')
                final_row=final_index.get((sid,side),{})
                final_grand=final_row.get('複核後Grand Score')
                final_neck=final_row.get('採用頸部分數')
                final_al=final_row.get('複核後AL')
                detail={'樣本編號':sid,'評估側':side,'外側工作最終採用值':r.get('外側工作最終採用值',''),
                    '外側工作採用來源':r.get('外側工作採用來源',''), '頸部角度(°)':r.get('頸部角度'),
                    '頸部角度計算來源':r.get('頸部_v114_計分來源',''),
                    '補正確認狀態':state,
                    '參考側影片時間(秒)':r.get('參考側影片時間(秒)'),
                    '作業條件設定來源':r.get('作業條件設定來源',''),
                    '原始系統總分':r.get('原始自動Grand Score',r.get(pre+' Grand Score')),
                    '設定後系統總分':final_grand,
                    '正式結果狀態':final_row.get('結果狀態','未確認'),
                    '設定後AL':final_al,
                    '共同頸部來源':final_row.get('頸部確認狀態',''),
                    '採用頸部分數':final_neck}
                manual_components=(overrides or {}).get((side,sid),{})
                for field,col in [('上臂',pre+'上臂分數'),('前臂',pre+'前臂分數'),
                    ('手腕',pre+'手腕分數'),('腕扭轉',pre+'手腕扭轉分數'),
                    ('頸部','頸部分數'),('軀幹','軀幹分數'),('腿部','腿部分數'),
                    ('肌肉','肌肉使用加分'),('負重','負重施力加分')]:
                    detail[field+'分數']=(final_neck if field=='頸部' and final_neck is not None
                        else manual_components.get('人工'+field+'分數',r.get(col)))
                detailed.append(detail)
        write_unique('複核後結果/照片補正後部位評分.csv',pd.DataFrame(detailed).to_csv(index=False).encode('utf-8-sig'))
        if neck_cases:
            write_unique('複核後結果/頸部分級確認.csv',shared_neck_case_table(neck_cases,neck_decisions).to_csv(index=False).encode('utf-8-sig'))
            details=[]
            for sid,case in sorted(neck_cases.items()):
                for score,scenario in case['各候選情境'].items():
                    details.append({'樣本編號':sid,'右側原估分':case['右側原頸部分數'],
                     '左側原估分':case['左側原頸部分數'],
                     '右側原角(°)':case.get('右側原頸角'),'左側原角(°)':case.get('左側原頸角'),
                     '候選共用頸分':score,**scenario,'狀態':case['狀態']})
            write_unique('技術診斷/頸部分級情境比對.csv',pd.DataFrame(details).to_csv(index=False).encode('utf-8-sig'))
        if len([s for s,df in side_frames.items() if isinstance(df,pd.DataFrame) and not df.empty]) == 2:
            write_unique('複核後結果/雙側結果.csv',user_report_reviewed(combined,combined=True).to_csv(index=False).encode('utf-8-sig'))
        else:
            write_unique('複核後結果/單側結果.csv',user_report_reviewed(reviewed).to_csv(index=False).encode('utf-8-sig'))
        history=[{'樣本編號':x.get('樣本編號'), '評估側':x.get('評估側'),
                  '紀錄類型':x.get('紀錄類型','辨識失敗'), '原因':x.get('拒絕原因',''),
                  '使用者決定':decisions.get(v113_key(x['樣本編號'],x['評估側']),'尚未選擇')} for x in review_items]
        write_unique('複核後結果/使用者複核紀錄.csv',pd.DataFrame(history,columns=['樣本編號','評估側','紀錄類型','原因','使用者決定']).to_csv(index=False).encode('utf-8-sig'))
        manual=pd.DataFrame(list(overrides.values())) if overrides else pd.DataFrame(columns=['樣本編號','評估側','人工Grand Score','人工AL'])
        write_unique('複核後結果/人工評分明細.csv',manual.to_csv(index=False).encode('utf-8-sig'))
        # 專題研究：保留系統兩個版本的分數，人工欄位空白待評分者獨立填寫。
        # 僅屬人工比對空表，不能把它誤當成已填妥的人工真值。
        lookup={(int(r['樣本編號']),r['評估側']):r for _,r in reviewed.iterrows()} if not reviewed.empty else {}
        template=[]
        for side,df in side_frames.items():
            if not isinstance(df,pd.DataFrame) or df.empty:continue
            pre='右手' if side=='右手側' else '左手'
            for _,r in df.iterrows():
                sid=int(r['樣本編號']);state=lookup.get((sid,side),{})
                template.append({
                    '照片編號':sid,'評估側':side,
                    '影片內分析經過時間(秒)':r.get('分析經過時間(秒)',r.get('時間(秒)')),
                    '參考側影片時間(秒)':r.get('參考側影片時間(秒)'),
                    '作業條件設定來源':r.get('作業條件設定來源',''),
                    '系統肌肉使用加分':r.get('肌肉使用加分'),
                    '系統負重施力加分':r.get('負重施力加分'),
                    '系統上臂角度(°)':r.get(pre+'上臂角度'),
                    '系統前臂角度(°)':r.get(pre+'前臂角度'),
                    '系統手腕角度(°)':r.get(pre+'手腕角度'),
                    '系統軀幹角度(°)':r.get('軀幹角度'),
                    '系統頸部2D代理角(°)':r.get('頸部角度'),
                    '正面補正來源':r.get('正面補正來源','未紀錄'),
                    '系統原始系統總分':state.get('原始Grand Score',r.get('原始自動Grand Score',r.get(pre+' Grand Score'))),
                    '系統人工補正後總分':state.get('複核後Grand Score'),
                    '系統最終採用頸部分數':state.get('採用頸部分數'),
                    '系統人工補正後AL':state.get('複核後AL'),
                    '系統結果狀態':state.get('結果狀態','待判定'),
                    '人工上臂角度(°)':'','人工前臂角度(°)':'',
                    '人工手腕角度(°)':'','人工軀幹角度(°)':'',
                    '人工頸部2D代理角(°)':'',
                    '人工上臂分數':'','人工前臂分數':'','人工手腕分數':'',
                    '人工頸部分數':'','人工軀幹分數':'','人工腿部分數':'',
                    '人工Grand Score':'','人工Action Level':'',
                    '評分者代碼':'','備註／遮擋／現場條件':'',
                })
        write_unique('研究驗證/人工評分比對空表.csv',pd.DataFrame(template).to_csv(index=False).encode('utf-8-sig'))
        write_unique('閱讀指南.txt',(
            'RULA 分析報告使用說明\n\n'
            '原始分析/：未經照片批次補正的分析前系統評分結果，僅供驗證演算法。\n'
            '複核後結果/：包含分析前設定的系統評分與人工補正後分數；品質／頸部待確認者留空。系統評分不代表已經人工驗證。\n'
            '複核後結果/逐張人工補正確認.csv：每張照片最後採用的條件及人工修改欄位。\n'
            '時間同步/：各鏡頭時間校正與影格對應。\n'
            '評估結果/逐張時段與作業條件.csv：記錄每張所用負重、肌肉使用與來源，含人工批次修改資訊。\n'
            '照片：各樣本的原始影像和骨架圖。\n'
            '技術診斷/：僅供開發者追查關鍵點、品質判斷、算法來源；不屬一般正式報表。\n'
            '研究驗證/人工評分比對空表.csv：手動填入人工分數／角度；人工欄位預設空白。\n\n'
            '手腕：Hand 2D 優先，未達品質要求時 Pose 2D 備援；3D 不參與計分。\n'
            '頸部：每側報表顯示實際計分使用的 2D 頸角及計算來源；左右同步只共用 RULA 頸部分數，不平均角度。\n'
            '頸部分級衝突可能改變行動等級時，才需要局部人工確認頸部。\n'
            '所有角度是影像幾何估測值，不是經校準的解剖關節角度。\n'
            '左右頸部分級衝突且可能改變行動等級時，原始報表的系統總分會留空；完成局部頸部確認後，請以「複核後結果」為準。\n'
            '前臂向外工作不確定：若有無補正總分相同則免確認、保留診斷；否則必須人工選有／沒有，正式分數暫留空。\n'
            '只要有一側無有效評分，就不輸出完整雙側最高分。\n'
            'RULA 標準查表及角度真值尚需獨立驗證。\n'
        ).encode('utf-8'))
    return out.getvalue()


def v8_show_rejected_center():
    st.subheader('剔除樣本／骨架品質提醒：逐張查看與處置')
    _,_,_cases_for_reject=batch_current_effective(st.session_state)
    if any(x.get('需要人工確認') and sid not in st.session_state.get('shared_neck_decisions',{}) for sid,x in _cases_for_reject.items()):
        with st.expander('需要頸部分級確認的影格（只修改頸部）',expanded=True):
            shared_neck_review_panel('reject')
    st.caption('每筆都可點開原始照片與骨架照片；一般品質提醒預設保留自動分數，只有使用者認為不可靠時才改成人工評分或剔除。原始分析檔不會被覆蓋。')
    all_items=st.session_state.get('v8_rejected_rows',[])
    actionable,notices=v112_split_review_items(all_items)
    photos=st.session_state.get('v8_rejected_photos',{})
    st.session_state.setdefault('v113_review_decisions',{})
    decisions=st.session_state.v113_review_decisions
    a,b,c=st.columns(3)
    a.metric('原始待處理紀錄',len(actionable))
    b.metric('一般品質提醒（可選）',len(notices))
    c.metric('使用者已決定處理',sum(v in (V113_MANUAL,V113_EXCLUDE,V113_KEEP) for v in decisions.values()))
    if not all_items:
        st.success('沒有剔除或品質提醒紀錄。')
        return
    filter_choice=st.radio('查看哪些紀錄', ['需要處理','一般品質提醒','全部'],horizontal=True,key='v113_filter')
    shown={'需要處理':actionable,'一般品質提醒':notices,'全部':actionable+notices}[filter_choice]
    if not shown:
        st.info('這個類別目前沒有樣本，可切換到其他類別。')
        return
    show_table=pd.DataFrame([{'樣本編號':x['樣本編號'],'評估側':x['評估側'],
            '性質':('一般品質提醒' if v112_is_informational_notice(x) else '需處理'),
            '原因':x.get('拒絕原因',''),
            '使用者決定':decisions.get(v113_key(x['樣本編號'],x['評估側']),'預設保留自動評分' if v112_is_informational_notice(x) else '尚未處理')} for x in shown])
    with st.expander('展開本類別的全部紀錄',expanded=False):
        st.dataframe(show_table,use_container_width=True,hide_index=True)
    choices=[f"第{x['樣本編號']}筆｜{x['評估側']}｜{('一般提醒' if v112_is_informational_notice(x) else '需處理')}" for x in shown]
    idx=st.selectbox('選擇要查看照片的樣本',range(len(shown)),format_func=lambda i:choices[i],key=f'v113_pick_{filter_choice}')
    item=shown[idx]
    sid=int(item['樣本編號']);side=item['評估側'];key=v113_key(sid,side)
    matched=[camera for camera in ('Front','Right','Left') if f'Sample_{sid:03d}_{side}_{camera}' in photos]
    if matched:
        preferred=item.get('建議查看鏡頭','Front')
        cam=st.radio('查看鏡頭',matched,index=matched.index(preferred) if preferred in matched else 0,horizontal=True,
                     key=f'v113_cam_{key}')
        img=photos[f'Sample_{sid:03d}_{side}_{cam}']
        raw_col,rendered_col=st.columns(2)
        with raw_col:
            st.markdown('**未渲染原始照片**')
            v10_show_image(img.get('raw'))
        with rendered_col:
            st.markdown('**系統骨架與可疑關節**')
            v10_show_image(img.get('overlay'))
        st.caption(f"辨識關鍵點：{img.get('points',0)}/33｜偵測品質代理值：{img.get('quality',0):.2f}｜{img.get('note','')}")
    else:
        st.warning('本筆影格沒有可供顯示的照片；請保留原始失敗紀錄。')
    st.warning('品質原因：'+str(item.get('拒絕原因','未知')))
    if item.get('跨視角診斷'):
        st.caption('跨視角：'+str(item['跨視角診斷']))
    side_frames=({'右手側':st.session_state.get('df_right_only'),
                  '左手側':st.session_state.get('df_left_only')} if st.session_state.get('analyzed_mode')=='左右側同步計算'
                 else {st.session_state.get('analyzed_mode'):st.session_state.get('df_records')})
    original=v113_get_sample_row(side,sid,side_frames)
    if original is not None:
        st.caption('原始自動分數：'+str(original.get('右手 Grand Score' if side=='右手側' else '左手 Grand Score','未提供')))
    is_notice=v112_is_informational_notice(item)
    _profile=batch_get_profile(original,side,st.session_state.get('batch_profiles',{}).get((side,sid),{})) if original is not None else {}
    _outside_impact=(rula_outside_score_sensitivity(original,side,_profile)
                     if original is not None and _profile.get('out_side') is None else None)
    _outside_block=bool(_outside_impact and not _outside_impact.get('same_grand'))
    if _outside_block:
        st.error('這筆的前臂向外側工作會影響正式評分。「維持自動評分」不能完成確認；請先選有／沒有。')
        _choose=st.radio('人工看這張照片，前臂是否向外側工作？', ['尚未決定','有','沒有'],
                         horizontal=True,key='v113_outside_choose_'+key)
        if st.button('儲存外側工作並重算',disabled=_choose=='尚未決定',key='v113_outside_apply_'+key):
            updated,completed=batch_update_profiles(st.session_state.get('batch_profiles',{}),
                st.session_state.get('batch_confirmed',set()), [(side,sid)], {'out_side':_choose=='有'})
            st.session_state.batch_profiles=updated
            st.session_state.batch_confirmed=completed
            st.rerun()
    options=([V113_PENDING,V113_MANUAL,V113_EXCLUDE] if _outside_block else
             [V113_KEEP,V113_MANUAL,V113_EXCLUDE] if (is_notice or original is not None)
             else [V113_PENDING,V113_MANUAL,V113_EXCLUDE])
    stored=decisions.get(key, V113_PENDING if _outside_block else V113_KEEP if is_notice else V113_PENDING)
    if stored not in options:stored=options[0]
    selected=st.selectbox('這筆樣本如何處理？',options,index=options.index(stored),key=f'v113_action_{key}')
    if selected!=decisions.get(key):
        # Store even the default so the exported audit trail reflects the user's choice.
        decisions[key]=selected
    if selected==V113_KEEP:
        _profile=batch_get_profile(original,side,st.session_state.get('batch_profiles',{}).get((side,sid),{})) if original is not None else {}
        if _profile.get('out_side','not_present') is None:
            impact=rula_outside_score_sensitivity(original,side,_profile)
            if impact['same_grand']:
                st.info('前臂向外側工作仍未辨識確定，但有／無補正不改變這筆 RULA 總分；不需為此項單獨人工補正。其他品質疑慮照常處理。')
            else:
                st.warning('前臂向外側工作尚未能辨識，且可能改變 RULA 分數。請到「異常／人工覆核 → 照片補正」明確設為有／沒有，或人工重新評分。')
        else:
            st.success('此筆採用自動計算及已儲存的人工補正；仍須確認其他品質疑慮。')
    elif selected==V113_EXCLUDE:
        st.warning('已剔除此側樣本的可信分數；原圖、原始 CSV 與影像測量角度依然保留。雙側分析需兩側都可用，才會輸出完整雙側最高分。')
    elif selected==V113_MANUAL:
        v113_manual_form(sid,side,item,original)
        if (side,sid) not in st.session_state.v4_overrides:
            st.warning('尚未儲存人工 RULA 分數，此筆暫不列入可信統計。')
    else:
        st.info('仍待人工決定；不會把辨識失敗的樣本當作有效分數。')
    reviewed,combined=v113_current_reports()
    if not reviewed.empty:
        r=reviewed[(reviewed['樣本編號']==sid)&(reviewed['評估側']==side)]
        if not r.empty:
            st.caption('目前複核後結果：'+str(r.iloc[0]['結果狀態']))
    st.caption('所有決定將寫入下載 ZIP 的 Final_Reviewed/ 資料夾；可隨時回到本頁更改。')


def shared_neck_review_panel(context='angle', selected_sample=None):
    """修改單一時點的共用頸部分數，左右 RULA 複核後結果會立即重算。"""
    _,_,cases=batch_current_effective(st.session_state)
    decisions=st.session_state.setdefault('shared_neck_decisions',{})
    pending=[sid for sid,c in sorted(cases.items()) if c.get('需要人工確認') and sid not in decisions]
    if not pending:
        st.success('目前沒有必須人工確認的共同頸部分數。')
    else:
        st.warning(f'{len(pending)} 筆兩側頸部分級衝突，可能改變完整雙側行動等級；只需確認這些影格的頸部。')
    optional=[sid for sid,c in sorted(cases.items()) if c.get('需要人工確認')]
    if selected_sample is not None:
        optional=[int(selected_sample)] if int(selected_sample) in optional else []
    if not optional:
        return
    sid=st.selectbox('選擇需要局部確認的樣本',optional,
        format_func=lambda s:f"第 {s} 筆｜"+('已人工確認' if s in decisions else '待頸部確認'),
        key='shared_neck_pick_'+context)
    case=cases[sid]
    st.caption(f"右側原估分 {case['右側原頸部分數']} ／ 左側原估分 {case['左側原頸部分數']}；系統不會任選其中一個視角當成真值。")
    # 顯示雙側影格；使用統一的照片比對函式，避免標題分隔符不同而找不到照片。
    cols=st.columns(2)
    for col,side in zip(cols,('右手側','左手側')):
        with col:
            st.markdown('**'+side+'骨架照片**')
            rula_show_sample_image(side,sid)
    old=decisions.get(sid)
    with st.form('shared_neck_form_'+context+'_'+str(sid)):
        base=st.radio('頸部姿勢（RULA 基礎分數）',[
            '前屈 0～10°（1分）','前屈 10～20°（2分）','前屈超過 20°（3分）','後仰（4分）'],
            index=max(0,min(3,int(old)-1)) if old is not None else 0,
            key='shared_neck_base_'+context+'_'+str(sid))
        twists=st.checkbox('頸部扭轉（+1）',value=bool(old is not None and int(old)>=5),key='shared_neck_twist_'+context+'_'+str(sid))
        bend=st.checkbox('頸部側彎（+1）',value=bool(old is not None and int(old)>=6),key='shared_neck_bend_'+context+'_'+str(sid))
        st.caption('先確認頸部基礎姿勢，再選擇扭轉／側彎補正。兩側上肢沿用原本的自動及人工評分，不需重填。')
        submit=st.form_submit_button('確認此筆頸部分數並重新計算雙側',type='primary')
    if submit:
        confirmed=min(6,int(base.split('（')[1][0])+int(twists)+int(bend))
        decisions[int(sid)]=confirmed
        st.success(f'已採用共同頸部分數 {confirmed}；左右側 RULA 會在複核後報表同步更新。')
        st.rerun()
    if sid in decisions and st.button('撤回這筆頸部確認',key='shared_neck_undo_'+context+'_'+str(sid)):
        del decisions[sid]
        st.rerun()


def v8_show_neck_center():
    """Directly show the same side-specific 2D angle that was used at scoring time.

    Two synchronized cameras do not yield a calibrated shared neck angle;
    their *scores*, not their geometric angles, are shared.
    """
    st.subheader('頸部角度與共同頸部分數')
    st.caption('角度沿用同側耳－肩－髖 2D 幾何法；品質不足時使用原本備援法。左右角度分別列出，不取平均，也不代表解剖學頸椎真實角度。')
    effective,_,cases=batch_current_effective(st.session_state)
    sources={side:df for side,df in effective.items()
             if isinstance(df,pd.DataFrame) and not df.empty}
    if not sources:
        st.info('沒有可用的頸部角度；若影格辨識失敗，可至剔除樣本查看原因。')
        return
    rows=[]
    for side,df in sources.items():
        for _,rec in df.sort_values('樣本編號').iterrows():
            sid=int(rec['樣本編號'])
            choice=st.session_state.get('shared_neck_decisions',{}).get(sid)
            case=cases.get(sid)
            neck_score=(int(choice) if choice is not None else
                (None if case and case['需要人工確認'] else rec.get('頸部分數')))
            rows.append({'照片編號':sid,'視角':side,
                '頸部角度(°)':rec.get('頸部角度'),
                '角度來源':rec.get('頸部_v114_計分來源','既有 2D 備援'),
                '採用頸部分數':neck_score,
                '頸部分級狀態':('待頸部人工確認' if case and case['需要人工確認'] and choice is None
                         else '人工已確認' if choice is not None
                         else rec.get('頸部共用狀態','單側評估'))})
    table=pd.DataFrame(rows).sort_values(['照片編號','視角'],ascending=[True,False])
    st.dataframe(table,use_container_width=True,hide_index=True)
    if len(sources)==2:
        st.caption('同一照片的右側和左側頸角可能不同；共用的是 RULA 頸部分數，並不是把角度平均。')
        with st.expander('左右側頸角差異與分級狀態',expanded=False):
            st.dataframe(shared_neck_case_table(cases,st.session_state.get('shared_neck_decisions',{})),
                         use_container_width=True,hide_index=True)
        shared_neck_review_panel('angle')
    if len(table)>1:
        with st.expander('查看頸部角度變化圖',expanded=False):
            chart=table.pivot(index='照片編號',columns='視角',values='頸部角度(°)')
            st.line_chart(chart,y_label='頸部 2D 代理角（°）')
    st.caption('上述角度及各側計算來源同步出現在一般報表、複核後 CSV 和研究驗證空表。')

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
        raise ValueError('同步後沒有共同重疊影格，請重新調整所選鏡頭的同步時間')
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
    """Prefer Front as reference; factory mode uses Right instead.

    This is only a *suggestion* based on frame-difference motion, not an approved synchronization.
    """
    front = 'Front' if 'Front' in camera_names else ('Right' if 'Right' in camera_names else camera_names[0])
    if front not in camera_names:
        raise ValueError('未找到可用同步參考鏡頭')
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


def rula_eval_sides(eval_mode):
    """評估側別決定哪些側視角為必填；正面影片完全選填。"""
    if str(eval_mode).startswith('左右側'):
        return ('右手側', '左手側')
    if eval_mode in ('右手側', '左手側'):
        return (eval_mode,)
    raise ValueError(f'未知評估模式：{eval_mode}')


def rula_required_cameras(eval_mode, upload_map=None):
    """正面鏡頭有上傳才使用；未上傳時不要求 Front，也不假造影格。"""
    if upload_map is None:
        upload_map = {'Front': st.session_state.get('v8_uploads', {}).get('front')}
    cameras = ['Front'] if upload_map.get('Front') is not None else []
    if '右手側' in rula_eval_sides(eval_mode):
        cameras.append('Right')
    if '左手側' in rula_eval_sides(eval_mode):
        cameras.append('Left')
    return cameras


def v9_sync_signature(upload_map, eval_mode):
    import hashlib
    names = []
    cams = rula_required_cameras(eval_mode, upload_map)
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
        '相對參考鏡頭偏移(秒)': round(float(anchors[cam])-float(anchors['Front' if 'Front' in cameras else cameras[0]]), 3),
        '共同時間起點(秒)': round(float(begin), 3),
        '共同時間終點(秒)': round(float(end), 3),
        '校正方式': mode, '使用者確認': '是'
    } for cam in cameras])


def v9_make_sample_times(n, metadata, anchors, cameras):
    begin, end = v9_common_window(metadata, anchors, cameras)
    return np.linspace(begin, end, int(n)), begin, end


def v9_sample_row(shared_time, metadata, anchors, cameras, sample_id, analysis_start=None):
    # Preserve the signed time relative to the synchronization event; add a
    # user-friendly elapsed time independent of where the sync event occurred.
    elapsed = float(shared_time) - (float(analysis_start) if analysis_start is not None else float(shared_time))
    result = {'樣本編號': int(sample_id), '時間(秒)': round(float(shared_time), 3),
              '共同時間(秒)': round(float(shared_time), 3),
              '分析經過時間(秒)': round(max(0.0, elapsed), 3)}
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


# ==================== V10.2: 作業識別與安全檔名 ====================
def rula_safe_filename_component(value, fallback="未填寫", max_chars=36):
    """保留中文與一般文字，排除路徑分隔符和 Windows 不合法的檔名字元。"""
    import re
    value = str(value or "").strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', value)
    value = re.sub(r'\s+', '_', value).strip(' ._')
    return (value[:max_chars].rstrip(' ._') or fallback)


def rula_job_metadata(values):
    date_value = values.get('job_date', date.today())
    if hasattr(date_value, 'strftime'):
        date_value = date_value.strftime('%Y-%m-%d')
    return {
        '評估日期': str(date_value),
        '單位／公司／部門': str(values.get('job_unit', '')).strip(),
        '員工／受測者': str(values.get('job_person', '')).strip(),
        '作業名稱': str(values.get('job_task', '')).strip(),
    }


def rula_job_zip_name(metadata, side_tag='RULA'):
    parts = [metadata.get('評估日期', ''), metadata.get('單位／公司／部門', ''),
             metadata.get('員工／受測者', ''), metadata.get('作業名稱', '')]
    parts = [rula_safe_filename_component(p) for p in parts]
    return '_'.join(parts + [rula_safe_filename_component(side_tag)]) + '.zip'


def rula_job_csv(metadata):
    return pd.DataFrame([metadata]).to_csv(index=False).encode('utf-8-sig')

# ===== 一般使用者報表：只顯示實際採用的角度和 RULA 結果 =====
def user_report_side(df, side, with_scores=False):
    prefix = '右手' if side == '右手側' else '左手'
    name_map = [
        ('樣本編號','樣本編號'),('分析經過時間(秒)','時間(秒)'),('時間(秒)','同步相對時間(秒)'),
        ('評估側','評估側'),(f'{prefix}上臂角度','上臂角度(°)'),
        (f'{prefix}前臂角度','前臂角度(°)'),(f'{prefix}手腕角度','手腕角度(°)'),
        ('頸部角度','頸部角度(°)'),('頸部_v114_計分來源','頸部角度計算來源'),
        ('頸部分數','頸部分數'),('頸部共用狀態','頸部確認狀態'),('軀幹角度','軀幹角度(°)'),
        ('參考側影片時間(秒)','參考側影片時間(秒)'),('作業條件設定來源','作業條件設定來源'),
        ('肌肉使用加分','肌肉使用加分'),('負重施力加分','負重施力加分'),
        (f'{prefix} Grand Score','RULA總分'),(f'{prefix} AL','行動等級'),
        ('品質狀態','品質狀態'),('正面補正來源','正面補正來源')]
    score_map = [(f'{prefix}上臂分數','上臂分數'),(f'{prefix}前臂分數','前臂分數'),
                 (f'{prefix}手腕分數','手腕分數'),(f'{prefix}手腕扭轉分數','手腕扭轉分數'),
                 ('頸部分數','頸部分數'),('軀幹分數','軀幹分數'),('腿部分數','腿部分數'),
                 ('肌肉使用加分','肌肉使用加分'),('負重施力加分','負重施力加分')]
    if df is None or df.empty:
        return pd.DataFrame(columns=[v for _,v in name_map])
    out = pd.DataFrame(index=df.index)
    for src,dst in name_map + (score_map if with_scores else []):
        if src in df.columns:
            out[dst]=df[src]
        elif src == '評估側':
            out[dst]=side
    # 未確認的共用頸部分數不得把候選 Grand Score 當成正式結果。
    if '頸部共用狀態' in df.columns:
        pending=df['頸部共用狀態'].eq('待頸部人工確認')
        for name in ('RULA總分','行動等級'):
            if name in out: out.loc[pending,name]=None
        if '頸部分數' in out:out.loc[pending,'頸部分數']=None
    # 不輸出其他模型估計角度、過去演算法名稱、測試用品質欄位。
    return out.reset_index(drop=True)


def user_report_dual(df, with_scores=False):
    name_map=[
        ('樣本編號','樣本編號'),('分析經過時間(秒)','時間(秒)'),
        ('時間(秒)','同步相對時間(秒)'),
        ('右上臂角度','右上臂角度(°)'),('右前臂角度','右前臂角度(°)'),
        ('右手腕角度','右手腕角度(°)'),
        ('左上臂角度','左上臂角度(°)'),('左前臂角度','左前臂角度(°)'),
        ('左手腕角度','左手腕角度(°)'),
        ('右側頸部實際角度','右側頸部角度(°)'),
        ('左側頸部實際角度','左側頸部角度(°)'),
        ('頸部分數','共同頸部分數'),('頸部共用狀態','頸部確認狀態'),
        ('軀幹角度','軀幹角度(°)'),('參考側影片時間(秒)','參考側影片時間(秒)'),
        ('作業條件設定來源','作業條件設定來源'),('肌肉使用加分','肌肉使用加分'),
        ('負重施力加分','負重施力加分'),('右手 Grand Score','右手RULA總分'),
        ('左手 Grand Score','左手RULA總分'),('最終最高分','雙側最高分'),
        ('最終 AL','行動等級'),('最危害側','較高風險側'),('品質狀態','品質狀態'),('正面補正來源','正面補正來源')]
    scores=[('右手 Score A','右手Score A'),('左手 Score A','左手Score A'),
            ('右手 Score C','右手Score C'),('左手 Score C','左手Score C')]
    if df is None or df.empty:
        return pd.DataFrame(columns=[v for _,v in name_map])
    out=pd.DataFrame(index=df.index)
    for src,dst in name_map+(scores if with_scores else []):
        if src in df.columns:
            out[dst]=df[src]
    if '頸部共用狀態' in df.columns:
        pending=df['頸部共用狀態'].eq('待頸部人工確認')
        for name in ('右手RULA總分','左手RULA總分','雙側最高分','行動等級','共同頸部分數'):
            if name in out: out.loc[pending,name]=None
    return out.reset_index(drop=True)


def user_report_reviewed(df, combined=False):
    if df is None or df.empty:
        return pd.DataFrame()
    if combined:
        cols=['樣本編號','雙側結果狀態','右側頸部角度(°)','左側頸部角度(°)',
              '採用頸部分數','頸部確認狀態','右手複核後分數','左手複核後分數',
              '完整雙側最高分','完整雙側AL','最高風險側','部分可用側別',
              '參考側影片時間(秒)','作業條件設定來源','肌肉使用加分','負重施力加分']
    else:
        cols=['樣本編號','評估側','頸部角度(°)','頸部角度計算來源','採用頸部分數','頸部確認狀態',
              '原始Grand Score','複核後Grand Score',
              '複核後AL','結果狀態','照片補正狀態','是否納入可信統計','使用者選擇',
              '參考側影片時間(秒)','作業條件設定來源','肌肉使用加分','負重施力加分','正面補正來源',
              '外側工作最終採用值','外側工作採用來源']
    mapping={'原始Grand Score':'原始總分','複核後Grand Score':'複核後總分',
             '複核後AL':'複核後行動等級','完整雙側AL':'行動等級',
             '完整雙側最高分':'雙側最高分',
             '是否納入可信統計':'是否有可用系統評分'}
    return df[[c for c in cols if c in df.columns]].rename(columns=mapping).copy()


def user_report_neck(df, side=None):
    if df is None or df.empty: return pd.DataFrame()
    if side is not None:
        return df[[c for c in ('樣本編號','分析經過時間(秒)','頸部角度','頸部分數') if c in df.columns]].rename(
            columns={'分析經過時間(秒)':'時間(秒)','頸部角度':'頸部角度(°)'}).copy()
    return df[[c for c in ('樣本編號','分析經過時間(秒)','頸部分數','頸部共用狀態') if c in df.columns]].rename(
        columns={'分析經過時間(秒)':'時間(秒)','頸部分數':'頸部分數',
                 '頸部共用狀態':'確認狀態'}).copy()


def user_report_wrist(df, side):
    if df is None or df.empty: return pd.DataFrame()
    pre='右手' if side=='右手側' else '左手'
    cols=['樣本編號','分析經過時間(秒)',f'{pre}手腕角度',f'{pre}手腕分數']
    return df[[c for c in cols if c in df.columns]].rename(columns={
        '分析經過時間(秒)':'時間(秒)',f'{pre}手腕角度':'手腕角度(°)',
        f'{pre}手腕分數':'手腕分數'}).copy()


# ==================== 【Streamlit 網頁應用介面】 ====================
st.set_page_config(page_title="RULA 智慧姿勢評估系統", layout="wide")
st.title("RULA 智慧姿勢評估系統")
with st.sidebar.expander('系統維護資訊', expanded=False):
    st.caption(f"部署識別：{APP_BUILD_TAG}｜程式：{os.path.basename(__file__)}")

# 啟動自檢：顯示實際部署版本並在分析前確認時序檢查函式已可呼叫。
# Streamlit 會在每次互動時由上到下重跑，因此函式必須在此之前宣告。
if not callable(globals().get("add_hand_temporal_notes")):
    st.error("部署自檢失敗：add_hand_temporal_notes 未完成定義。請檢查 GitHub 上傳的 222.py 是否完整。")
    st.stop()
st.caption("自動辨識姿勢與計算 RULA 評分；品質不確定的樣本可檢視原圖、人工評分或剔除。")

st.markdown(
    "> **系統說明**：本工具以 MediaPipe Pose 關鍵點估算部分姿勢角度，遇到關鍵點可見度不足時會嘗試原圖、對比增強及放大圖；仍不可靠的樣本會略過。RULA 部分影像補正可選擇自動判讀；手腕扭轉、腿足支撐、聳肩、上臂支撐及扭轉等項目仍需人工確認，輸出仍須由受訓評估者複核，不應單獨作為醫療或職安決策依據。"
)

if "v4_overrides" not in st.session_state:
    st.session_state.v4_overrides = {}
if 'batch_profiles' not in st.session_state:
    st.session_state.batch_profiles = {}
if 'batch_confirmed' not in st.session_state:
    st.session_state.batch_confirmed = set()

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
    "1｜影片與基本設定", "2｜鏡頭時間同步",
    "3｜自動辨識設定", "4｜啟動分析",
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
st.session_state.setdefault('v113_review_decisions',{})
st.session_state.setdefault('shared_neck_cases',{})
st.session_state.setdefault('shared_neck_decisions',{})
st.session_state.setdefault('v9_confirmed',False)
st.session_state.setdefault('v9_sync_mode','尚未同步')
st.session_state.setdefault('v9_confirmed_signature',None)
st.session_state.setdefault('v9_auto_suggestion',None)
st.session_state.setdefault('v9_sync_summary',None)
for _cam in ('Front','Right','Left'):
    st.session_state.v8_values.setdefault('sync_'+_cam,0.0)



def v8_save_value(name):
    st.session_state.v8_values[name]=st.session_state.get('_v8_field_'+name)
    if name == 'eval_mode':
        st.session_state.v9_confirmed=False
        st.session_state.v9_auto_suggestion=None
        st.session_state.v8_values['factory_front_confirmed']=False
        st.session_state['_v8_field_factory_front_confirmed']=False
    if name.startswith('manual_') and st.session_state.v8_values.get('factory_front_confirmed',False):
        # A previously confirmed factory modifier set must be re-confirmed after editing.
        st.session_state.v8_values['factory_front_confirmed']=False
        st.session_state['_v8_field_factory_front_confirmed']=False


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
    if name in ('left', 'right'):
        st.session_state.v8_values['timeline_enabled']=False
        st.session_state.v8_values['timeline_count']=0
        st.session_state.v8_values.pop('timeline_video_signature',None)
        st.session_state.pop('timeline_last_saved_revision',None)
    cam={'front':'Front','right':'Right','left':'Left'}.get(name)
    if cam:
        st.session_state.v8_values['sync_'+cam]=0.0
        st.session_state.pop('_v9_anchor_'+cam,None)
    st.session_state.v9_confirmed=False
    st.session_state.v9_auto_suggestion=None
    st.session_state.v8_values['factory_front_confirmed']=False
    st.session_state['_v8_field_factory_front_confirmed']=False


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
            if name in ('left', 'right'):
                st.session_state.v8_values['timeline_enabled']=False
                st.session_state.v8_values['timeline_count']=0
                st.session_state.v8_values.pop('timeline_video_signature',None)
                st.session_state.pop('timeline_last_saved_revision',None)
            cam={'front':'Front','right':'Right','left':'Left'}.get(name)
            if cam:
                st.session_state.v8_values['sync_'+cam]=0.0
                st.session_state.pop('_v9_anchor_'+cam,None)
            st.session_state.v9_confirmed=False
            st.session_state.v9_auto_suggestion=None
            st.session_state.v8_values['factory_front_confirmed']=False
            st.session_state['_v8_field_factory_front_confirmed']=False
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
EVAL_OPTIONS=['右手側','左手側','左右側同步計算']
# 舊工作階段可能仍保存「工廠雙側」或舊的三視角長名稱；更新時自動沿用對應評估側。
_legacy_eval = st.session_state.v8_values.get('eval_mode', EVAL_OPTIONS[0])
if _legacy_eval not in EVAL_OPTIONS:
    _normalized = ('左右側同步計算' if str(_legacy_eval).startswith(('左右側', '工廠雙側'))
                   else EVAL_OPTIONS[0])
    st.session_state.v8_values['eval_mode'] = _normalized
    st.session_state['_v8_field_eval_mode'] = _normalized
    st.session_state.v9_confirmed = False
    st.session_state.v9_confirmed_signature = None
MUSCLE_OPTIONS=['0: 非靜態且動作少於每分鐘 4 次','1: 姿勢維持超過 1 分鐘或動作每分鐘至少 4 次']
FORCE_OPTIONS=['0: < 2kg (間歇施力)','1: 2-10kg (間歇施力)','2: 2-10kg 靜態／重複施力','3: >10kg 負載、重複施力或衝擊力']
LEG_OPTIONS=['請先目視確認…','腿足受支撐且平衡（1分）','腿足未受支撐或不平衡（2分）']
TWIST_OPTIONS=['請先目視確認…','中間範圍（1分）','接近旋轉極限（2分）']
AUTO_FLAGS={
    'is_abducted':'上臂外展（正面肩肘相對位置）',
    'is_across_midline':'前臂跨越身體中線（正面向量投影）',
    'is_out_to_side':'前臂向外側工作（肩髖雙參考＋前臂方向）',
    'is_side_bending':'軀幹側彎（肩髖中心線）',
    # 由正面影片人工核對發現大量誤判，此項不再列入自動辨識選項。
    'upper_arm_extended':'上臂向後伸展（側面有向角度）',
    'trunk_extended':'軀幹向後伸展（側面有向角度）',
    'neck_extended':'頸部後仰（側面保守估測；不明確時人工設定）',
}
front_labels={
    'is_abducted':'上臂外展（+1）',
    'is_across_midline':'前臂跨中線（+1）',
    'is_out_to_side':'前臂向外側工作（+1）',
    'is_side_bending':'軀幹側彎（+1）',
    'is_neck_side_bending':'頸部側彎（+1）',
}
for flag in AUTO_FLAGS:
    v8_value('auto_'+flag,True)
# 升級舊版時只對新增／先前預設關閉的方向辨識做一次預設啟用。
if not st.session_state.get('_directional_defaults_migrated'):
    for _flag in ('upper_arm_extended','trunk_extended','neck_extended'):
        st.session_state.v8_values['auto_'+_flag]=True
        st.session_state['_v8_field_auto_'+_flag]=True
    st.session_state['_directional_defaults_migrated']=True
# 頸部側彎永遠由使用者作業層級設定（一次設定＋批次例外）
st.session_state.v8_values['auto_is_neck_side_bending'] = False
for flag in front_labels:
    for side in ('右手側','左手側'):
        v8_value('manual_'+side+'_'+flag,False)
auto_selection={flag:v8_value('auto_'+flag,True) for flag in AUTO_FLAGS}
auto_selection['is_neck_side_bending'] = False
manual_front={(side,flag):v8_value('manual_'+side+'_'+flag,False)
              for side in ('右手側','左手側') for flag in front_labels}

# Defaults for all settings, even when their page isn't active.
eval_mode=v8_value('eval_mode',EVAL_OPTIONS[0])
# 正面是否存在由實際上傳檔案決定，與單側／雙側評估模式無關。
factory_mode=st.session_state.v8_uploads.get('front') is None
if factory_mode:
    # 無正面影片：這五項不能把「未能觀測」當成「否」。
    auto_selection.update({flag:False for flag in front_labels})
num_samples=v8_value('num_samples',25)
combo_muscle=v8_value('muscle',MUSCLE_OPTIONS[0])
combo_force=v8_value('force',FORCE_OPTIONS[0])
vid_front_file=st.session_state.v8_uploads.get('front')
vid_right_file=st.session_state.v8_uploads.get('right')
vid_left_file=st.session_state.v8_uploads.get('left')
leg_support_choice=v8_value('leg',LEG_OPTIONS[1])
neck_twisted_manual=v8_value('neck_twisted',False)
neck_extended_manual=v8_value('neck_extended',False)
trunk_twisted_manual=v8_value('trunk_twisted',False)
trunk_extended_manual=v8_value('trunk_extended',False)
right_wrist_twist_choice=v8_value('rw_twist',TWIST_OPTIONS[1])
left_wrist_twist_choice=v8_value('lw_twist',TWIST_OPTIONS[1])
right_wrist_deviated=v8_value('rw_deviated',False)
left_wrist_deviated=v8_value('lw_deviated',False)
right_shoulder_raised=v8_value('rs_raised',False)
left_shoulder_raised=v8_value('ls_raised',False)
right_arm_supported=v8_value('ra_supported',False)
left_arm_supported=v8_value('la_supported',False)
right_arm_extended=v8_value('ra_extended',False)
left_arm_extended=v8_value('la_extended',False)
# 以明示的基礎假設開始；本系統不聲稱由單張影像辨識腳部支撐／手腕旋轉。
if leg_support_choice == LEG_OPTIONS[0]:
    leg_support_choice=LEG_OPTIONS[1]
    st.session_state.v8_values['leg']=leg_support_choice
for _twist_name in ('rw_twist','lw_twist'):
    if st.session_state.v8_values.get(_twist_name) == TWIST_OPTIONS[0]:
        st.session_state.v8_values[_twist_name]=TWIST_OPTIONS[1]
right_wrist_twist_choice=st.session_state.v8_values['rw_twist']
left_wrist_twist_choice=st.session_state.v8_values['lw_twist']
# v10：自動啟用 Hand；模型不存在時依 Pose 2D 備援。
v6_enable_hand=True
v6_policy_label='Hand 優先、Pose 2D 備援（固定）'
neck_v114_mode=v8_value('neck_v114_mode',V114_NECK_MODES[1])
raw_image_export=v8_value('raw_export',True)
v11_qc_policy=v8_value('v11_qc_policy',V11_QC_MODES[0])
# Metadata is stored outside widgets so it survives next/previous navigation.
v8_value('job_date', date.today())
v8_value('job_unit', '')
v8_value('job_person', '')
v8_value('job_task', '')


page=st.session_state.v8_page_index if st.session_state.v9_screen == "setup" else -1
if page == 0:
    st.subheader('上傳影片與作業設定')
    st.markdown('#### 作業基本資料（用於匯出檔名與報告識別）')
    job_a,job_b=st.columns(2)
    with job_a:
        v8_widget('date_input','評估日期','job_date',date.today(),format='YYYY/MM/DD')
        v8_widget('text_input','公司／單位／部門','job_unit','',placeholder='例如：生產部')
    with job_b:
        v8_widget('text_input','員工／受測者','job_person','',placeholder='例如：員工A或匿名編號')
        v8_widget('text_input','作業名稱','job_task','',placeholder='例如：人工搬運作業')
    st.caption('可留空；未填的資料會在檔名中以「未填寫」標示。若分享 ZIP，建議使用匿名受測者編號。')

    eval_mode=v8_widget('radio','評估模式','eval_mode',EVAL_OPTIONS[0],options=EVAL_OPTIONS,horizontal=True)
    selected_sides=rula_eval_sides(eval_mode)
    st.info('正面鏡頭【選填】：建議盡量提供，可協助自動估計外展、跨中線與側彎等補正，通常能減少人工確認。但正面影像不保證角度更準，仍須檢查遮擋與骨架品質。')
    upload_columns=st.columns(1+len(selected_sides))
    with upload_columns[0]:
        vid_front_file=v8_upload('正面鏡頭（選填，建議上傳）','front')
    for column, side_name in zip(upload_columns[1:], selected_sides):
        with column:
            if side_name=='右手側':
                vid_right_file=v8_upload('右側鏡頭（必要）','right')
            else:
                vid_left_file=v8_upload('左側鏡頭（必要）','left')
    factory_mode = vid_front_file is None
    d,e,f=st.columns(3)
    with d: num_samples=v8_widget('slider','抽樣數量','num_samples',25,min_value=5,max_value=100)
    with e: combo_muscle=v8_widget('selectbox','肌肉使用預設分數','muscle',MUSCLE_OPTIONS[0],options=MUSCLE_OPTIONS)
    with f: combo_force=v8_widget('selectbox','負重／施力預設分數','force',FORCE_OPTIONS[0],options=FORCE_OPTIONS)
    with st.expander('📋 肌肉使用／負重｜結案前現場條件確認', expanded=True):
        v8_widget('text_input','實際單次負重重量（kg，可填範圍）','actual_load_kg','',
                  placeholder='例如 5 或 4～6；無資料可留空')
        v8_widget('text_input','實際作業重複頻率（次／分鐘）','actual_repetition','',
                  placeholder='例如 5；不確定可留空')
        v8_widget('checkbox','我已依現場條件核對肌肉使用 +0/+1、負重 +0～+3 及影片時段','work_conditions_verified',False)
        st.caption('勾選僅代表你已核對，不會替你猜測重量或自動改分；未勾選也可試跑，但下載報表會標示「待確認」。')
    # 影片時段以必填的「右側／左側原始影片秒數」為準，不使用相對同步秒數。
    with st.expander('🎬 影片時間軸｜拖曳區間設定負重與肌肉使用', expanded=True):
        st.caption('直接在影片縮圖時間軸選取兩端點，也能新增多組時段、拖曳起點／終點。'
                   '每段分別設定肌肉使用和負重；其餘秒數沿用上方預設。'
                   '確認後按「儲存全部區間並套用」。')
        st.caption('肌肉使用 +1 必須根據靜態姿勢持續超過一分鐘或每分鐘重複超過四次等作業資訊判斷，'
                   '不是只看一張照片有沒有出力。')
        ref_side_cam='Right' if '右手側' in selected_sides else 'Left'
        ref_file=vid_right_file if ref_side_cam=='Right' else vid_left_file
        ref_title='右側' if ref_side_cam=='Right' else '左側'
        if ref_file is None:
            st.info(f'請先上傳{ref_title}影片，才能使用拖曳時間軸。')
        else:
            try:
                ref_duration=float(v9_video_metadata_from_bytes(ref_file.getvalue())['duration'])
                st.caption(f'時間基準：{ref_title}原始影片（約 {ref_duration:.2f} 秒）。'
                           '雙側影片沿用原本的同步設定，只需要設定一次。')
                rula_timeline_editor(ref_file, ref_duration, st.session_state.v8_values, ref_side_cam)
                # Streamlit 元件有時會因首輪 iframe 掛載／收合而沒有即時顯示。
                # 原生雙端點滑桿不依賴 JS 元件，保證上傳當輪就能選取區間。
                with st.expander('備用：原生雙端點時間軸（自訂拖曳畫面未顯示時使用）',expanded=False):
                    st.caption('與上方使用同一支評估側影片。這裡可以直接拖曳兩個端點並新增一個時段。')
                    st.video(ref_file)
                    _max_seconds=max(0.1,float(ref_duration))
                    _bounds=st.slider('選擇起點與終點（秒）',min_value=0.0,max_value=_max_seconds,
                        value=(0.0,round(_max_seconds,2)),step=0.1,key='timeline_native_range')
                    _n1,_n2=st.columns(2)
                    _m=_n1.selectbox('此區間肌肉使用',options=[0,1],key='timeline_native_muscle')
                    _f=_n2.selectbox('此區間負重／施力',options=[0,1,2,3],key='timeline_native_force')
                    if st.button('加入這個時段並套用',key='timeline_native_save'):
                        _current=rula_timeline_segments_from_values(st.session_state.v8_values)
                        _current.append({'title':f'時段{len(_current)+1}','start':_bounds[0],
                            'end':_bounds[1],'muscle':_m,'force':_f})
                        try:
                            rula_timeline_apply_interactive(st.session_state.v8_values,_current,ref_duration)
                            st.success('區間已儲存；分析時會使用。')
                            st.rerun()
                        except ValueError as _exc:st.error('無法儲存：'+str(_exc))
            except (ValueError, OSError, FileNotFoundError) as exc:
                st.error('影片時間軸載入失敗：'+str(exc))
        st.caption('分析後若只有少數照片的條件不同，仍可使用「照片批次修改」，不必重跑影片。')
    with st.expander('骨架品質與遮擋篩檢（選用，預設保守）', expanded=False):
        v11_qc_policy=v8_widget('selectbox','複核警示敏感度','v11_qc_policy',V11_QC_MODES[0],options=V11_QC_MODES)
        st.caption('只列嚴重疑慮是預設；頭頸與手臂可能重疊時先做幾何及前後影格檢查，仍不可靠才送人工複核。這是品質風險篩檢，不能證明關節實際位置。')
    st.info('未上傳正面影片：仍可完成所選側的影像角度分析；正面視角無法確定的補正請依現場資訊設定。' if factory_mode else '已提供正面影片：可輔助五項姿勢補正；下一步校正所選鏡頭的共同時間。')

elif page == 1:
    st.subheader('鏡頭時間同步｜只需校正一次')
    st.caption('只需將所選鏡頭對到同一個動作瞬間，系統自動換算每張影格的時間；影片不需同時開始或等長。')
    active_cams = rula_required_cameras(eval_mode, {'Front':vid_front_file,'Right':vid_right_file,'Left':vid_left_file})
    upload_map = {'Front': vid_front_file,'Right': vid_right_file,'Left': vid_left_file}
    available = all(upload_map[c] is not None for c in active_cams)
    if not available:
        st.warning('請先到第 1 步，上傳所選評估側的必要影片。正面影片可不提供。')
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
            if len(active_cams) == 1 and not st.session_state.v9_confirmed:
                # 只有一支側面影片無須跨鏡頭同步；自動以影片 0 秒為參考。
                v9_use_zero_anchors(active_cams)
                st.session_state.v9_sync_mode = '單鏡頭（不需同步）'
            if len(active_cams) == 1:
                st.success('目前只有一支影片，無須跨鏡頭同步；已使用影片 0 秒作為基準。')
            st.markdown('#### A. 快速自動建議（選用）')
            st.caption('比對所選鏡頭的動作強度曲線；視角不同或重複動作時可能對錯，必須人工預覽確認。')
            if st.button('✨ 分析動作並建議時間偏移', key='v9_find_motion', use_container_width=True):
                try:
                    with st.spinner('正在以低解析度計算所選影片的動作變化…'):
                        motion = {c:v9_motion_from_bytes(upload_map[c].getvalue()) for c in active_cams}
                        st.session_state.v9_auto_suggestion = v9_suggest_anchors(motion,meta,active_cams)
                except Exception as exc:
                    st.session_state.v9_auto_suggestion = {'usable':False,'reason':str(exc)}
            suggestion = st.session_state.get('v9_auto_suggestion')
            if suggestion:
                if suggestion.get('usable'):
                    st.dataframe(pd.DataFrame([{'鏡頭':cam,'建議同步事件秒數':suggestion['anchors'][cam],
                        '動作相關性':suggestion['pairs'][cam]['correlation'] if cam!=active_cams[0] else 1.0,
                        '建議可信度':suggestion['pairs'][cam]['confidence'] if cam!=active_cams[0] else '參考鏡頭'}
                        for cam in active_cams]),hide_index=True,use_container_width=True)
                    st.warning(suggestion['caution'])
                    st.button('套用建議秒數，再查看預覽畫面', key='v9_apply_motion',
                              on_click=v9_apply_auto_suggestion,use_container_width=True)
                else:
                    st.info('自動同步不夠可靠：'+suggestion.get('reason','無法估計')+'。請直接使用下方手動對齊。')
            st.markdown('#### B. 影格預覽與微調')
            st.caption('找所選影片共有的動作，例如「手第一次碰到物品」。各畫面的時間都指向同一事件時，按「確認同步」。')
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
                        v10_show_image(picture, f'{title}｜{val:.3f} 秒')
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
                st.caption('共同時間 0 秒是選定的動作事件；負秒代表事件之前。所有鏡頭的採樣時間會各自加上這裡的偏移。')
                st.button('✅ 所有畫面已對到同一動作，確認同步', key='v9_confirm',
                          on_click=v9_confirm_sync,args=('預覽確認',),type='primary',use_container_width=True)
                if st.session_state.v9_confirmed and st.session_state.v9_confirmed_signature==sig:
                    st.success('已確認同步，可以按下一步繼續。')
                with st.expander('沒有時間校正需求？可明確選擇 0 秒基準'):
                    st.warning('只有確定所有使用的攝影機同時開始錄影，才建議跳過。')
                    st.button('使用全部 0 秒，並標記為未校正',key='v9_no_alignment',
                              on_click=v9_use_zero_anchors,args=(active_cams,))
            except ValueError as exc:
                st.error(f'同步時間沒有有效重疊：{exc}')

elif page == 2:
    st.subheader('自動辨識條件（預設啟用可用項目）')
    if factory_mode:
        st.warning('沒有正面鏡頭：上臂外展、跨中線、軀幹及頸部側彎等先沿用此頁的人工預設；如果不同照片條件變化，分析後再批次修改。無法確認的情況請勿宣稱為純自動準確評分。')
        st.caption('正面影像缺席時，正面代理補正會使用分析前設定，仍可用照片批次修改。')
    else:
        st.caption('勾選表示使用影像向量代理判斷；取消後會顯示右側／左側的手動勾選。')
    if st.button('一鍵啟用全部可用的自動辨識項目',key='enable_all_safe_automatic'):
        for _flag in AUTO_FLAGS:
            st.session_state.v8_values['auto_'+_flag]=True
            st.session_state['_v8_field_auto_'+_flag]=True
        st.rerun()
    cols=st.columns(2)
    for i,(flag,label) in enumerate(AUTO_FLAGS.items()):
        if factory_mode and flag in front_labels:
            continue
        with cols[i%2]:
            auto_selection[flag]=v8_widget('checkbox',label,'auto_'+flag,True)
    if factory_mode:
        auto_selection.update({flag:False for flag in front_labels})
    st.info('前臂向外側工作：以肩膀＋髖部及肘腕方向共同判斷；只有不確定且可能改變 RULA 分數者，才出現在「異常／人工覆核 → 待確認照片」。')
    st.info('頸部側彎：舊自動判讀已有假陽性，因此沒有自動加分，也不要求在分析前整批勾選。若某張確實側彎，請在分析後的頸部確認或照片補正頁修改。')
    st.info('軀幹側彎：有可靠正面影像時仍會以肩／髖中心線做候選自動判定；頸部側彎因原先誤判偏多，仍不採自動加分，初始為「沒有」，可在分析後的「異常／人工覆核 → 頸部確認／照片補正」逐張更改。')
    st.caption('手腕扭轉、腿足支撐與其他需現場確認的條件，不會宣稱由影片自動辨識。預設使用正常基礎姿勢（各 1 分）；若作業不符，請在分析後照片補正依實際影格修正。')
    if factory_mode:
        st.warning('未提供正面影片：正面補正無法自動判讀，初始採「沒有」的可修改假設；不是辨識確定沒有。請於照片補正檢查特殊姿勢。')
    st.caption('頸部會以同側耳朵、肩膀與髖部計算相對角度；關鍵點不足時使用備援方式並保留診斷紀錄。')

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
        "neck_v114_mode": neck_v114_mode,
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


def manual_without_front(side_name):
    """沒有 Front 時，只使用操作員已明確確認的補正，不將缺圖推論為否。"""
    if not factory_mode or side_name not in rula_eval_sides(eval_mode):
        raise ValueError('無正面模式或評估側別不符')
    # 分析前只是系統計算值；正式採信必須在分析後的照片批次補正中心確認。
    defaults = {flag:bool(manual_front.get((side_name,flag),False)) for flag in front_labels}
    defaults.update({'front_auto_paused':True,'front_screening_reason':'未上傳正面影片：沿用分析前的人工預設',
                     'outside_work_status':'人工設定（無正面）', 'outside_work_reason':'無正面影片；以工作初始條件暫算，例外需批次複核',
                     'front_keypoint_visibility':0.0,'front_rotation_warning':False})
    return defaults


def rula_front_frame_fallback(side_name, reason='正面影像不可用'):
    """正面自動補正失敗不應阻止仍可運作的側面 RULA 計分。"""
    base = {flag: bool(manual_front.get((side_name,flag),False)) for flag in front_labels}
    base.update({'front_auto_paused':True,'front_screening_reason':reason,
                 'outside_work_status':'人工設定（正面不可用）', 'outside_work_reason':reason+'；沿用分析前人工設定',
                 'front_rotation_warning':False,'front_keypoint_visibility':0.0,
                 'neck_lateral_angle_proxy':float('nan'),'trunk_lateral_angle_proxy':float('nan')})
    return base

# 在非設定頁也必須由保存的資料建立每次分析的選項。
v6_hand_policy='hand_first'
if page == 3:
    st.subheader('啟動分析')
    ready_files=[name for name,f in [('正面',vid_front_file),('右側',vid_right_file if '右手側' in rula_eval_sides(eval_mode) else None),('左側',vid_left_file if '左手側' in rula_eval_sides(eval_mode) else None)] if f]
    st.write('已上傳：'+('、'.join(ready_files) if ready_files else '尚未上傳影片'))
    st.write(f'抽樣數：{num_samples}　｜　分析模式：{eval_mode}　｜　手腕來源：Hand 優先／Pose 2D 備援')
    if not st.session_state.v8_values.get('work_conditions_verified',False):
        st.warning('肌肉使用／負重尚未核實，請回第 1 步確認；可試跑，但正式報告應保留未核實註記。')
    st.caption('腿足支撐與手腕扭轉使用正常基礎分數（各 1 分）作為初始假設；這兩項不由影像自動辨識。若實際姿勢不同，請到分析後「照片補正」修改。')
    st.write('時間同步：'+('已確認（'+st.session_state.v9_sync_mode+'）' if st.session_state.v9_confirmed else '尚未確認'))
    if not st.session_state.v9_confirmed:
        st.warning('先到「鏡頭時間同步」預覽畫面並確認，否則不會開始分析。')
    if factory_mode:
        st.info('沒有正面影片：使用這裡設定作為系統評分；只有作業階段不同時再多選照片調整，不需逐張確認。')
    st.caption('執行後自動跳到結果總覽；原本的分析及 ZIP 報表功能皆保留。')

# 執行分析按鈕
if page == 3 and st.button("🚀 啟動全關節生物力學分析", type="primary", use_container_width=True):
    is_factory_mode = vid_front_file is None  # 無正面影片；可為單側或雙側評估
    is_dual_mode = eval_mode.startswith('左右側同步計算')

    # 只強制要求使用者選定的評估側；正面鏡頭永遠是選填。
    required_sides = rula_eval_sides(eval_mode)
    missing_sides = [side for side in required_sides if
                     (vid_right_file if side == '右手側' else vid_left_file) is None]
    if missing_sides:
        st.error('缺少必要側面影片：'+ '、'.join(missing_sides)+'。正面影片選填。')
        st.stop()

    # V9: prevent analyzing mismatched or stale video synchronization settings.
    sync_inputs={'Front':vid_front_file,'Right':vid_right_file,'Left':vid_left_file}
    active_cams=rula_required_cameras(eval_mode, sync_inputs)
    if (not st.session_state.get('v9_confirmed') or
        st.session_state.get('v9_confirmed_signature') != v9_sync_signature(sync_inputs, eval_mode)):
        st.error('尚未確認這一組影片的同步錨點。請先到第 2 步預覽並按「確認同步」。')
        st.stop()
    sync_anchors={cam:float(st.session_state.v8_values.get('sync_'+cam,0.0)) for cam in active_cams}
    sync_mode=st.session_state.v9_sync_mode

    # 不再以隱藏的全樣本人工確認步驟阻止分析；必要照片可在複核中心修改。

    with st.spinner(f"系統正在進行 33 點姿態特徵擷取與 RULA 解算（模式：{eval_mode}）..."):
        st.session_state.v4_overrides = {}
        st.session_state.batch_profiles = {}
        st.session_state.batch_confirmed = set()
        st.session_state.display_images_b64 = []
        st.session_state.display_right_b64 = []
        st.session_state.display_left_b64 = []
        st.session_state.zip_bytes = None
        st.session_state.hand_review_images = {}
        st.session_state.v113_review_decisions = {}
        st.session_state.shared_neck_cases = {}
        st.session_state.shared_neck_decisions = {}
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
                st.warning("Hand 模型無法下載或初始化，本次改用 Pose 2D 備援進行手腕 RULA 評分。請查看模型路徑或將 hand_landmarker.task 放入 models/。")

        m_val = int(combo_muscle.split(":")[0])
        f_val = int(combo_force.split(":")[0])
        # 時段使用主側面影片的原始播放秒數；正面鏡頭不做時間基準。
        timeline_ref='Right' if '右手側' in required_sides else 'Left'
        try:
            timeline_duration=float(v9_video_metadata_from_bytes(
                (vid_right_file if timeline_ref=='Right' else vid_left_file).getvalue())['duration'])
            timeline_rules=rula_timeline_rules(st.session_state.v8_values,timeline_duration)
        except (ValueError,TypeError) as exc:
            st.error('影片時段設定有誤：'+str(exc)+'。請返回「影片與基本設定」修正。')
            st.stop()
        st.session_state['timeline_rules_used']=[dict(x) for x in timeline_rules]
        st.session_state['timeline_reference_camera']=timeline_ref
        st.session_state['timeline_default_muscle']=m_val
        st.session_state['timeline_default_force']=f_val

        # ==================== 【模式 A：左右側同步計算 (三鏡頭分開計算、分析與擷取)】 ====================
        if is_dual_mode:
            t_front = None
            if not is_factory_mode:
                t_front = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                t_front.write(vid_front_file.getvalue()); t_front.close()

            t_right = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_right.write(vid_right_file.getvalue()); t_right.close()

            t_left = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_left.write(vid_left_file.getvalue()); t_left.close()

            cap_front = cv2.VideoCapture(t_front.name) if t_front is not None else None
            cap_right = cv2.VideoCapture(t_right.name)
            cap_left  = cv2.VideoCapture(t_left.name)

            fps_front = float(cap_front.get(cv2.CAP_PROP_FPS) or 30.0) if cap_front is not None else None
            fps_right = float(cap_right.get(cv2.CAP_PROP_FPS) or 30.0)
            fps_left = float(cap_left.get(cv2.CAP_PROP_FPS) or 30.0)
            count_front = int(cap_front.get(cv2.CAP_PROP_FRAME_COUNT)) if cap_front is not None else None
            count_right = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))
            count_left = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_front = (fps_front if np.isfinite(fps_front) and fps_front > 0 else 30.0) if fps_front is not None else None
            fps_right = fps_right if np.isfinite(fps_right) and fps_right > 0 else 30.0
            fps_left = fps_left if np.isfinite(fps_left) and fps_left > 0 else 30.0
            sync_meta={'Right':v9_frame_meta(fps_right,count_right),
                       'Left':v9_frame_meta(fps_left,count_left)}
            if cap_front is not None:
                sync_meta['Front']=v9_frame_meta(fps_front,count_front)
            try:
                sample_times, sync_start, sync_end = v9_make_sample_times(
                    num_samples,sync_meta,sync_anchors,active_cams)
            except ValueError as exc:
                for c in [cap_front,cap_right,cap_left]:
                    if c is not None: c.release()
                for p in [t_front.name if t_front else None,t_right.name,t_left.name]:
                    if p:
                        try: os.remove(p)
                        except OSError: pass
                st.error('影片無法建立同步區間：'+str(exc))
                st.stop()
            sync_report=v9_sync_report(sync_meta,sync_anchors,active_cams,sync_start,sync_end,sync_mode)
            sync_samples={i+1:v9_sample_row(t,sync_meta,sync_anchors,active_cams,i+1,sync_start)
                          for i,t in enumerate(sample_times)}

            combined_records = []
            shared_neck_cases = {}
            right_only_records = []
            left_only_records = []
            zip_right_imgs = []
            zip_left_imgs = []
            zip_hand_imgs = []
            zip_raw_imgs = []
            keypoint_records = []
            v11_qc_diagnostics = []
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
                    sec = round(float(sample_time), 2)
                    local_ref_time=sync_samples[i+1][timeline_ref+'影片時間(秒)']
                    m_at,f_at,stage_source=rula_timeline_at(local_ref_time,timeline_rules,m_val,f_val)
                    idx_front = sync_samples[i+1]['Front影格索引'] if cap_front is not None else None
                    idx_right = sync_samples[i+1]['Right影格索引']
                    idx_left = sync_samples[i+1]['Left影格索引']
                    if cap_front is not None:
                        cap_front.set(cv2.CAP_PROP_POS_FRAMES, idx_front)
                    cap_right.set(cv2.CAP_PROP_POS_FRAMES, idx_right)
                    cap_left.set(cv2.CAP_PROP_POS_FRAMES, idx_left)

                    s_f, f_frame = cap_front.read() if cap_front is not None else (True,None)
                    s_r, r_frame = cap_right.read()
                    s_l, l_frame = cap_left.read()

                    # 只因 Front 單張讀取失敗，不可丟掉仍有完整側面影像的姿勢。
                    if s_r and s_l:
                        if not s_f:
                            f_frame = None
                        if raw_image_export:
                            for label, raw in (("Front", f_frame), ("Right", r_frame), ("Left", l_frame)):
                                if raw is None:
                                    continue
                                payload = _encode_preview(raw)
                                if payload:
                                    zip_raw_imgs.append((f"Raw_Frames/{label}/Sample_{i+1:03d}_{sec}s.jpg", payload))
                        # 1. 正面鏡頭分別提取右手與左手的冠狀面外展/交叉補正
                        if is_factory_mode:
                            front_mods_r=manual_without_front('右手側')
                            front_mods_l=manual_without_front('左手側')
                            front_ok_r=front_ok_l=False  # 無 Front 不得標記為正面自動辨識成功
                        elif f_frame is None:
                            front_ok_r=front_ok_l=False
                            front_mods_r=rula_front_frame_fallback('右手側','正面單張影格讀取失敗')
                            front_mods_l=rula_front_frame_fallback('左手側','正面單張影格讀取失敗')
                        else:
                            front_ok_r, front_mods_r = analyze_front_camera(f_frame, pose, "右手側")
                            front_ok_l, front_mods_l = analyze_front_camera(f_frame, pose, "左手側")
                            if not front_ok_r:
                                front_mods_r = rula_front_frame_fallback('右手側', front_mods_r.get('front_screening_reason','正面關鍵點不足'))
                            if not front_ok_l:
                                front_mods_l = rula_front_frame_fallback('左手側', front_mods_l.get('front_screening_reason','正面關鍵點不足'))

                        # 2. 右側鏡頭只評估右側關鍵點；低品質正面補正會使該樣本略過，不默認為「否」。
                        succ_r, data_r, img_r, conf_r = analyze_side_camera(
                            r_frame, pose, "右手側", front_mods_r, m_at, f_at,
                            manual_mods=build_manual_mods("右手側"),
                            hand_task=hand_task, hand_scoring_policy=v6_hand_policy
                        )

                        # 3. 左側鏡頭只評估左側關鍵點。
                        succ_l, data_l, img_l, conf_l = analyze_side_camera(
                            l_frame, pose, "左手側", front_mods_l, m_at, f_at,
                            manual_mods=build_manual_mods("左手側"),
                            hand_task=hand_task, hand_scoring_policy=v6_hand_policy
                        )
                        # 共用頸分必須先於 QC、RULA 總表與影格標註建立。
                        if succ_r and succ_l and data_r is not None and data_l is not None:
                            neck_case=shared_neck_decide_pair(data_r,data_l,i+1,sec)
                            if neck_case is not None:
                                shared_neck_cases[i+1]=neck_case

                        if not succ_r or data_r is None:
                            skipped_right += 1
                            skip_diagnostics.append({"樣本編號": i+1, "時間(秒)": sec, "評估側": "右手側", "可能原因": "側面骨架不足／角度無效（正面補正不可用但不會單獨造成側面失敗）"})
                            v8_add_rejected_sample(rejected_rows,rejected_photos,i+1,sec,"右手側",
                                "側面骨架不足／角度無效（正面補正不可用但不會單獨造成側面失敗）",
                                {"Front":f_frame,"Right":r_frame,"Left":l_frame},pose)
                        if not succ_l or data_l is None:
                            skipped_left += 1
                            skip_diagnostics.append({"樣本編號": i+1, "時間(秒)": sec, "評估側": "左手側", "可能原因": "側面骨架不足／角度無效（正面補正不可用但不會單獨造成側面失敗）"})
                            v8_add_rejected_sample(rejected_rows,rejected_photos,i+1,sec,"左手側",
                                "側面骨架不足／角度無效（正面補正不可用但不會單獨造成側面失敗）",
                                {"Front":f_frame,"Right":r_frame,"Left":l_frame},pose)

                        if succ_r and data_r is not None:
                            v11_snap_r = data_r.get('_v7_keypoints')
                            v11_qc_r = v11_update_qc(data_r,v11_snap_r,'右手側',cap_right,idx_right,fps_right,pose,l_frame,v11_qc_policy,neck_peer=data_l if succ_l else None)
                            v11_qc_r = v11_front_fallback_note(v11_qc_r,data_r,front_ok_r,auto_selection)
                            v11_qc_diagnostics.append({'樣本編號':i+1,'時間(秒)':sec,'評估側':'右手側',**v11_qc_r})
                            if v11_qc_r['品質是否列入複核中心']:
                                v11_add_qc_review(rejected_rows,rejected_photos,i+1,sec,'右手側',v11_qc_r,
                                    {'Front':f_frame,'Right':r_frame,'Left':l_frame},pose,img_r,v11_snap_r)
                            hand_points_r = data_r.pop("_v6_hand_points", None)
                            hand_angle_r = data_r.pop("_v6_hand_angle", float("nan"))
                            snap_r = data_r.pop("_v7_keypoints", None)
                            if snap_r is not None:
                                keypoint_records.append({"樣本編號": i+1, "時間(秒)": sec, **snap_r})
                            rec_r = {"樣本編號": i + 1, "時間(秒)": sec, "評估側": "右手側", "右側信賴度(%)": round(conf_r, 1)}
                            rec_r.update(data_r)
                            rec_r.update(rula_timeline_video_fields(local_ref_time,timeline_rules,m_val,f_val))
                            rec_r['正面補正來源']=data_r.get('正面補正來源','未紀錄')
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
                            v11_snap_l = data_l.get('_v7_keypoints')
                            v11_qc_l = v11_update_qc(data_l,v11_snap_l,'左手側',cap_left,idx_left,fps_left,pose,r_frame,v11_qc_policy,neck_peer=data_r if succ_r else None)
                            v11_qc_l = v11_front_fallback_note(v11_qc_l,data_l,front_ok_l,auto_selection)
                            v11_qc_diagnostics.append({'樣本編號':i+1,'時間(秒)':sec,'評估側':'左手側',**v11_qc_l})
                            if v11_qc_l['品質是否列入複核中心']:
                                v11_add_qc_review(rejected_rows,rejected_photos,i+1,sec,'左手側',v11_qc_l,
                                    {'Front':f_frame,'Right':r_frame,'Left':l_frame},pose,img_l,v11_snap_l)
                            hand_points_l = data_l.pop("_v6_hand_points", None)
                            hand_angle_l = data_l.pop("_v6_hand_angle", float("nan"))
                            snap_l = data_l.pop("_v7_keypoints", None)
                            if snap_l is not None:
                                keypoint_records.append({"樣本編號": i+1, "時間(秒)": sec, **snap_l})
                            rec_l = {"樣本編號": i + 1, "時間(秒)": sec, "評估側": "左手側", "左側信賴度(%)": round(conf_l, 1)}
                            rec_l.update(data_l)
                            rec_l.update(rula_timeline_video_fields(local_ref_time,timeline_rules,m_val,f_val))
                            rec_l['正面補正來源']=data_l.get('正面補正來源','未紀錄')
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
                                "參考側影片時間(秒)":round(float(local_ref_time),3),
                                "作業條件設定來源":stage_source,
                                "肌肉使用加分":m_at,
                                "負重施力加分":f_at,
                                "頸部角度": np.nan,  # 相反視角 2D 頸角不得算術平均成解剖角
                                "頸部分數": data_r['頸部分數'],
                                "頸部共用狀態": data_r.get('頸部共用狀態','未知'),
                                "頸部_v7_舊版平均(僅展示)": round((data_r["頸部_v7_原角度"]+data_l["頸部_v7_原角度"])/2,1),
                                "右側頸部實際角度": data_r["頸部角度"],
                                "左側頸部實際角度": data_l["頸部角度"],
                                "右側頸部原角度": data_r["頸部_v7_原角度"],
                                "左側頸部原角度": data_l["頸部_v7_原角度"],
                                "右側頸部_v114_同側角度": data_r["頸部_v114_同側角度"],
                                "左側頸部_v114_同側角度": data_l["頸部_v114_同側角度"],
                                "右側頸部_v114_品質": data_r["頸部_v114_品質"],
                                "左側頸部_v114_品質": data_l["頸部_v114_品質"],
                                "右側頸部_v114_計分來源": data_r["頸部_v114_計分來源"],
                                "左側頸部_v114_計分來源": data_l["頸部_v114_計分來源"],
                                "頸部_v114_左右角差": (round(abs(data_r["頸部_v114_同側角度"] - data_l["頸部_v114_同側角度"]),1)
                                    if np.isfinite(data_r["頸部_v114_同側角度"]) and np.isfinite(data_l["頸部_v114_同側角度"]) else np.nan),
                                "頸部_v114_左右分級差": (abs(get_neck_score(data_r["頸部_v114_同側角度"]) - get_neck_score(data_l["頸部_v114_同側角度"]))
                                    if data_r["頸部_v114_品質通過"] and data_l["頸部_v114_品質通過"] else np.nan),
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
                                "Overall_Conf": overall_conf,
                                "右側骨架品質":data_r.get('品質狀態','未知'),
                                "左側骨架品質":data_l.get('品質狀態','未知'),
                                "品質狀態":('需人工複核' if '需人工複核' in (data_r.get('品質狀態'),data_l.get('品質狀態')) else ('留意' if '留意' in (data_r.get('品質狀態'),data_l.get('品質狀態')) else '通過')),
                                "品質可直接採信":bool(data_r.get('品質可直接採信') and data_l.get('品質可直接採信')),
                                "品質原因":'右側：'+str(data_r.get('品質原因',''))+'；左側：'+str(data_l.get('品質原因','')),
                                "品質計分說明":('系統評分：至少一側需複核' if not (data_r.get('品質可直接採信') and data_l.get('品質可直接採信')) else '品質篩檢通過')
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

            for c in [cap_front, cap_right, cap_left]:
                if c is not None: c.release()
            for p in [t_front.name if t_front else None, t_right.name, t_left.name]:
                if p:
                    try: os.remove(p)
                    except Exception: pass

            v9_decorate_records(combined_records,sync_samples)
            v9_decorate_records(right_only_records,sync_samples)
            v9_decorate_records(left_only_records,sync_samples)
            v9_decorate_records(skip_diagnostics,sync_samples)
            v9_decorate_records(rejected_rows,sync_samples)
            df_comb = pd.DataFrame(combined_records) if combined_records else pd.DataFrame()
            df_r = add_hand_temporal_notes(pd.DataFrame(right_only_records), "右手側") if right_only_records else pd.DataFrame()
            df_l = add_hand_temporal_notes(pd.DataFrame(left_only_records), "左手側") if left_only_records else pd.DataFrame()

            job_metadata=rula_job_metadata(st.session_state.v8_values)
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('作業基本資料.csv',rula_job_csv(job_metadata))
                zf.writestr('作業基本資料.json',json.dumps(job_metadata,ensure_ascii=False,indent=2).encode('utf-8'))
                zf.writestr('Synchronization/Synchronization_Report.csv',sync_report.to_csv(index=False).encode('utf-8-sig'))
                zf.writestr('Synchronization/Frame_Time_Map.csv',pd.DataFrame(sync_samples.values()).to_csv(index=False).encode('utf-8-sig'))
                zf.writestr('Synchronization/Settings.json',json.dumps({'mode':sync_mode,'anchors':sync_anchors,
                    'common_start':sync_start,'common_end':sync_end,'cameras':active_cams,
                    'has_front_view':not is_factory_mode,
                    'frontal_modifiers_source':'operator_initial_pending_photo_review' if is_factory_mode else 'front_camera',
                    'manual_front_modifiers':{side:{flag:bool(manual_front.get((side,flag),False)) for flag in front_labels}
                        for side in ('右手側','左手側')} if is_factory_mode else None,
                    'neck_mode':neck_v114_mode},
                    ensure_ascii=False,indent=2).encode('utf-8'))
                if shared_neck_cases:
                    zf.writestr('Diagnostics/Shared_Neck_Cases.json',
                        json.dumps(clean_json(list(shared_neck_cases.values())),ensure_ascii=False,indent=2).encode('utf-8'))
                if not df_comb.empty:
                    zf.writestr("評估結果/雙側總覽.csv", user_report_dual(df_comb).to_csv(index=False).encode('utf-8-sig'))
                    zf.writestr("評估結果/雙側評分明細.csv", user_report_dual(df_comb,True).to_csv(index=False).encode('utf-8-sig'))
                    cols=[c for c in df_comb.columns if c.startswith('頸部_v8_') or c in
                          ('樣本編號','時間(秒)','右側頸部原角度','左側頸部原角度')]
                    zf.writestr('Diagnostics/Neck_Comparison.csv',df_comb[cols].to_csv(index=False).encode('utf-8-sig'))
                if not df_r.empty:
                    zf.writestr("評估結果/右側角度與評分.csv", user_report_side(df_r,'右手側').to_csv(index=False).encode('utf-8-sig'))
                    zf.writestr("評估結果/右側評分明細.csv", user_report_side(df_r,'右手側',True).to_csv(index=False).encode('utf-8-sig'))
                if not df_l.empty:
                    zf.writestr("評估結果/左側角度與評分.csv", user_report_side(df_l,'左手側').to_csv(index=False).encode('utf-8-sig'))
                    zf.writestr("評估結果/左側評分明細.csv", user_report_side(df_l,'左手側',True).to_csv(index=False).encode('utf-8-sig'))
                zf.writestr("Diagnostics/V114_Neck_Same_Side_Comparison.csv",
                            v114_neck_report(df_r, df_l).to_csv(index=False).encode('utf-8-sig'))
                zf.writestr("Diagnostics/V114_Neck_Method.txt", V114_NECK_METHOD.encode('utf-8'))
                zf.writestr("Diagnostics/Skipped_Samples.csv", pd.DataFrame(skip_diagnostics if skip_diagnostics else [{"樣本編號":"","時間(秒)":"","評估側":"","可能原因":""}]).to_csv(index=False).encode("utf-8-sig"))
                essential_reviews, informational_notices = v112_split_review_items(rejected_rows)
                zf.writestr("Diagnostics/Rejected_Samples_Detail.csv",pd.DataFrame(essential_reviews).to_csv(index=False).encode('utf-8-sig'))
                zf.writestr("Diagnostics/Quality_Notices.csv",pd.DataFrame(informational_notices).to_csv(index=False).encode('utf-8-sig'))
                if is_factory_mode:
                    manual_rows=[{'側別':side, **{front_labels[flag]: ('有（+1分）' if manual_front.get((side,flag),False) else '無（初始系統計算值）') for flag in front_labels}}
                                 for side in ('右手側','左手側')]
                    zf.writestr('評估結果/無正面人工補正確認.csv',pd.DataFrame(manual_rows).to_csv(index=False).encode('utf-8-sig'))
                zf.writestr('評估結果/攝影與補正方式.txt',(
                    '評估模式：'+('雙側（無正面）' if is_factory_mode else '三視角完整分析')+'\n'
                    '正面視角補正：'+('初始系統計算值；照片批次補正後確認' if is_factory_mode else '以正面影像估算／必要時人工備援')+'\n'
                    '注意：缺少正面影片不能推論外展、跨中線、側彎等均不存在。\n'
                    ).encode('utf-8'))
                zf.writestr("Diagnostics/V11_1_Skeleton_Quality.csv",pd.DataFrame(v11_qc_diagnostics).to_csv(index=False).encode('utf-8-sig'))
                zf.writestr("Diagnostics/V11_2_Review_Explanation.txt",(
                    "Rejected_Samples_Detail.csv 僅列需處理的失敗／複核；Quality_Notices.csv 保留一般品質提醒。\n"
                    "提醒不等於角度錯誤；品質通過不等於人體角度準確率證明。\n"
                    "共同時間相對同步事件，可為負；分析經過時間以分析開始為 0 秒。\n"
                    "RULA Table A/B/C 尚應與同一份標準工作表逐格比對。\n"
                ).encode('utf-8'))
                zf.writestr("Diagnostics/V11_Skeleton_Quality.csv",pd.DataFrame(v11_qc_diagnostics).to_csv(index=False).encode('utf-8-sig'))
                zf.writestr("Diagnostics/V11_1_Quality_Guide.txt",V11_QC_GUIDE.encode('utf-8'))
                zf.writestr("Diagnostics/V11_Quality_Guide.txt",V11_QC_GUIDE.encode('utf-8'))
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
            st.session_state.shared_neck_cases = shared_neck_cases
            st.session_state.display_right_b64 = display_right_b64
            st.session_state.display_left_b64 = display_left_b64
            st.session_state.zip_bytes = zip_buffer.getvalue()
            st.session_state.v8_rejected_rows = rejected_rows
            st.session_state.v8_rejected_photos = rejected_photos
            st.session_state.zip_filename = rula_job_zip_name(job_metadata, '左右側分析')
            st.session_state.analyzed_mode = "左右側同步計算"  # 兩／三鏡頭共用相同結果資料結構
            st.session_state.skipped_info = {"expected": total_samples, "right": skipped_right, "left": skipped_left}
            st.session_state.v9_sync_summary=sync_report
            st.session_state.capture_mode='雙側無正面' if is_factory_mode else '三視角完整'
            gc.collect()

        # ==================== 【模式 B：單側評估 (右手側 或 左手側)】 ====================
        else:
            target_side = eval_mode
            selected_side_vid = vid_right_file if target_side == "右手側" else vid_left_file

            t_front = None
            if vid_front_file is not None:
                t_front = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                t_front.write(vid_front_file.getvalue()); t_front.close()

            t_side = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_side.write(selected_side_vid.getvalue()); t_side.close()

            cap_front = cv2.VideoCapture(t_front.name) if t_front is not None else None
            cap_side = cv2.VideoCapture(t_side.name)

            fps_front = float(cap_front.get(cv2.CAP_PROP_FPS) or 30.0) if cap_front is not None else None
            fps_side = float(cap_side.get(cv2.CAP_PROP_FPS) or 30.0)
            count_front = int(cap_front.get(cv2.CAP_PROP_FRAME_COUNT)) if cap_front is not None else None
            count_side = int(cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_front = (fps_front if np.isfinite(fps_front) and fps_front > 0 else 30.0) if fps_front is not None else None
            fps_side = fps_side if np.isfinite(fps_side) and fps_side > 0 else 30.0
            side_camera='Right' if target_side=='右手側' else 'Left'
            sync_meta={side_camera:v9_frame_meta(fps_side,count_side)}
            if cap_front is not None:
                sync_meta['Front']=v9_frame_meta(fps_front,count_front)
            try:
                sample_times,sync_start,sync_end=v9_make_sample_times(
                    num_samples,sync_meta,sync_anchors,active_cams)
            except ValueError as exc:
                if cap_front is not None: cap_front.release()
                cap_side.release()
                for p in [t_front.name if t_front else None,t_side.name]:
                    if p:
                        try:os.remove(p)
                        except OSError:pass
                st.error('影片無法建立同步區間：'+str(exc))
                st.stop()
            sync_report=v9_sync_report(sync_meta,sync_anchors,active_cams,sync_start,sync_end,sync_mode)
            sync_samples={i+1:v9_sample_row(t,sync_meta,sync_anchors,active_cams,i+1,sync_start)
                          for i,t in enumerate(sample_times)}

            records = []
            zip_images = []
            zip_hand_imgs = []
            zip_raw_imgs = []
            keypoint_records = []
            v11_qc_diagnostics = []
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
                    sec = round(float(sample_time), 2)
                    local_ref_time=sync_samples[i+1][timeline_ref+'影片時間(秒)']
                    m_at,f_at,stage_source=rula_timeline_at(local_ref_time,timeline_rules,m_val,f_val)
                    idx_front = sync_samples[i+1]['Front影格索引'] if cap_front is not None else None
                    idx_side = sync_samples[i+1][side_camera+'影格索引']
                    if cap_front is not None:
                        cap_front.set(cv2.CAP_PROP_POS_FRAMES, idx_front)
                    cap_side.set(cv2.CAP_PROP_POS_FRAMES, idx_side)
                    s_f, f_frame = cap_front.read() if cap_front is not None else (True,None)
                    s_s, s_frame = cap_side.read()

                    if s_s:
                        if not s_f:
                            f_frame = None  # 正面讀取失敗不使有效側面照片失效
                        if raw_image_export:
                            for label, raw in (("Front", f_frame), ("Right" if target_side == "右手側" else "Left", s_frame)):
                                if raw is None:
                                    continue
                                payload = _encode_preview(raw)
                                if payload:
                                    zip_raw_imgs.append((f"Raw_Frames/{label}/Sample_{i+1:03d}_{sec}s.jpg", payload))
                        if cap_front is None:
                            front_ok = False  # 未上傳正面，不得冒充正面自動判定成功。
                            front_mods = manual_without_front(target_side)
                        else:
                            front_ok, front_mods = analyze_front_camera(f_frame, pose, target_side)
                            if not front_ok:
                                front_mods = rula_front_frame_fallback(target_side, front_mods.get('front_screening_reason','正面關鍵點不足'))
                        succ_s, data, side_annotated, conf = analyze_side_camera(
                            s_frame, pose, target_side, front_mods, m_at, f_at,
                            manual_mods=build_manual_mods(target_side),
                            hand_task=hand_task, hand_scoring_policy=v6_hand_policy
                        )
                        if not succ_s or data is None:
                            skipped_count += 1
                            skip_diagnostics.append({"樣本編號": i+1, "時間(秒)": sec, "評估側": target_side, "可能原因": "側面骨架不足／角度無效（正面補正不可用但不會單獨造成側面失敗）"})
                            v8_add_rejected_sample(rejected_rows,rejected_photos,i+1,sec,target_side,
                                "側面骨架不足／角度無效（正面補正不可用但不會單獨造成側面失敗）",
                                {"Front":f_frame,"Right" if target_side=="右手側" else "Left":s_frame},pose)

                        if succ_s and data is not None:
                            v11_snap = data.get('_v7_keypoints')
                            v11_qc = v11_update_qc(data,v11_snap,target_side,cap_side,idx_side,fps_side,pose,f_frame,v11_qc_policy)
                            v11_qc = v11_front_fallback_note(v11_qc,data,front_ok,auto_selection)
                            v11_qc_diagnostics.append({'樣本編號':i+1,'時間(秒)':sec,'評估側':target_side,**v11_qc})
                            if v11_qc['品質是否列入複核中心']:
                                v11_add_qc_review(rejected_rows,rejected_photos,i+1,sec,target_side,v11_qc,
                                    {'Front':f_frame,side_camera:s_frame},pose,side_annotated,v11_snap)
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
                            record.update(rula_timeline_video_fields(local_ref_time,timeline_rules,m_val,f_val))
                            record['正面補正來源'] = data.get('正面補正來源','未紀錄')
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

            if cap_front is not None: cap_front.release()
            cap_side.release()
            for path in (t_front.name if t_front else None, t_side.name):
                if path:
                    try: os.remove(path)
                    except OSError: pass

            v9_decorate_records(records,sync_samples)
            v9_decorate_records(skip_diagnostics,sync_samples)
            v9_decorate_records(rejected_rows,sync_samples)
            df_result = add_hand_temporal_notes(pd.DataFrame(records), target_side) if records else pd.DataFrame()
            
            job_metadata=rula_job_metadata(st.session_state.v8_values)
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('作業基本資料.csv',rula_job_csv(job_metadata))
                zf.writestr('作業基本資料.json',json.dumps(job_metadata,ensure_ascii=False,indent=2).encode('utf-8'))
                zf.writestr('Synchronization/Synchronization_Report.csv',sync_report.to_csv(index=False).encode('utf-8-sig'))
                zf.writestr('Synchronization/Frame_Time_Map.csv',pd.DataFrame(sync_samples.values()).to_csv(index=False).encode('utf-8-sig'))
                zf.writestr('Synchronization/Settings.json',json.dumps({'mode':sync_mode,'anchors':sync_anchors,
                    'common_start':sync_start,'common_end':sync_end,'cameras':active_cams,
                    'has_front_view': cap_front is not None,
                    'frontal_modifiers_source': ('front_camera' if cap_front is not None else 'operator_confirmed'),
                    'manual_front_modifiers': ({target_side: {flag: bool(manual_front.get((target_side,flag),False))
                        for flag in front_labels}} if cap_front is None else None),
                    'neck_mode':neck_v114_mode},
                    ensure_ascii=False,indent=2).encode('utf-8'))
                if not df_result.empty:
                    zf.writestr("評估結果/角度與評分.csv", user_report_side(df_result,target_side).to_csv(index=False).encode('utf-8-sig'))
                    zf.writestr("評估結果/評分明細.csv", user_report_side(df_result,target_side,True).to_csv(index=False).encode('utf-8-sig'))
                if cap_front is None:
                    manual_rows=[{'側別':target_side, **{
                        front_labels[flag]: ('有（+1分）' if manual_front.get((target_side,flag),False) else '無（初始系統計算值）')
                        for flag in front_labels}}]
                    zf.writestr('評估結果/無正面人工補正確認.csv',
                                pd.DataFrame(manual_rows).to_csv(index=False).encode('utf-8-sig'))
                zf.writestr('評估結果/攝影與補正方式.txt',(
                    '評估模式：'+target_side+'\n'
                    '正面影片：'+('無（人工補正）' if cap_front is None else '有（影像輔助）')+'\n'
                    '注意：缺少正面影片不代表外展、跨中線、側彎等補正不存在。\n'
                ).encode('utf-8'))
                zf.writestr("Diagnostics/V114_Neck_Same_Side_Comparison.csv",
                            v114_neck_report(df_result).to_csv(index=False).encode('utf-8-sig'))
                zf.writestr("Diagnostics/V114_Neck_Method.txt", V114_NECK_METHOD.encode('utf-8'))
                zf.writestr("Diagnostics/Skipped_Samples.csv", pd.DataFrame(skip_diagnostics if skip_diagnostics else [{"樣本編號":"","時間(秒)":"","評估側":"","可能原因":""}]).to_csv(index=False).encode("utf-8-sig"))
                essential_reviews, informational_notices = v112_split_review_items(rejected_rows)
                zf.writestr("Diagnostics/Rejected_Samples_Detail.csv",pd.DataFrame(essential_reviews).to_csv(index=False).encode('utf-8-sig'))
                zf.writestr("Diagnostics/Quality_Notices.csv",pd.DataFrame(informational_notices).to_csv(index=False).encode('utf-8-sig'))
                zf.writestr("Diagnostics/V11_1_Skeleton_Quality.csv",pd.DataFrame(v11_qc_diagnostics).to_csv(index=False).encode('utf-8-sig'))
                zf.writestr("Diagnostics/V11_2_Review_Explanation.txt",(
                    "Rejected_Samples_Detail.csv 僅列需處理的失敗／複核；Quality_Notices.csv 保留一般品質提醒。\n"
                    "提醒不等於角度錯誤；品質通過不等於人體角度準確率證明。\n"
                    "共同時間相對同步事件，可為負；分析經過時間以分析開始為 0 秒。\n"
                    "RULA Table A/B/C 尚應與同一份標準工作表逐格比對。\n"
                ).encode('utf-8'))
                zf.writestr("Diagnostics/V11_Skeleton_Quality.csv",pd.DataFrame(v11_qc_diagnostics).to_csv(index=False).encode('utf-8-sig'))
                zf.writestr("Diagnostics/V11_1_Quality_Guide.txt",V11_QC_GUIDE.encode('utf-8'))
                zf.writestr("Diagnostics/V11_Quality_Guide.txt",V11_QC_GUIDE.encode('utf-8'))
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
            st.session_state.zip_filename = rula_job_zip_name(job_metadata, side_en_tag+'側分析')
            st.session_state.analyzed_mode = target_side
            st.session_state.skipped_info = {"expected": total_samples, "single": skipped_count}
            st.session_state.v9_sync_summary=sync_report
            st.session_state.capture_mode='單側無正面' if cap_front is None else '單側＋正面'
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


def rula_pending_review_records(reviewed):
    """產生需使用者處理的照片清單；同一張共同頸分只列一項。"""
    if reviewed is None or reviewed.empty:
        return []
    cases = {}
    for _, row in reviewed.iterrows():
        sid = int(row['樣本編號'])
        side = str(row['評估側'])
        status = str(row.get('結果狀態',''))
        neck = str(row.get('頸部確認狀態',''))
        if neck == '待人工確認' or '待頸部局部人工確認' in status:
            cases[('頸部',sid)] = {'類型':'頸部','照片編號':sid,'評估側':'雙側共用',
                '原因':'左右側頸部分級不一致，需要確認共同頸分'}
        elif status.startswith('待釐清補正') or str(row.get('照片補正狀態','')).startswith('無法確認'):
            cases[('前臂',side,sid)]={'類型':'前臂','照片編號':sid,'評估側':side,
                '原因':'前臂向外側工作有無會改變 RULA 分數，請選「有／沒有」'}
        elif status.startswith('待複核') or status.startswith('等待人工'):
            cases[('品質',side,sid)]={'類型':'品質','照片編號':sid,'評估側':side,
                '原因':status}
        # 遺失／被剔除的影格由「剔除樣本」獨立顯示，不要求無限人工補分。
    return sorted(cases.values(),key=lambda r:(r['照片編號'],r['類型'],r['評估側']))


def rula_show_sample_image(side, sample_id):
    """顯示已儲存骨架影格，沒有骨架時改嘗試技術診斷的原始照片。"""
    import re as _re
    source = (st.session_state.get('display_right_b64',[]) if side=='右手側' else
              st.session_state.get('display_left_b64',[]))
    if not source:
        source=st.session_state.get('display_images_b64',[])
    picture=next(((cap,img) for cap,img in source
                  if _re.search(r'#\s*'+str(int(sample_id))+r'(?!\d)',str(cap))),None)
    if picture:
        v10_show_image(picture[1],picture[0],encoded=True)
        st.caption('照片內的分數為分析當下數字；人工補正後以最新報表為準。')
        return True
    photos=st.session_state.get('v8_rejected_photos',{})
    cam='Right' if side=='右手側' else 'Left'
    entry=photos.get(f'Sample_{int(sample_id):03d}_{side}_{cam}')
    if entry:
        v10_show_image(entry.get('raw'))
        return True
    st.info('沒有這張影格的預覽照片，可查看原始分析 ZIP 的影格照片。')
    return False


def rula_pending_photo_panel():
    reviewed,_=v113_current_reports()
    pending=rula_pending_review_records(reviewed)
    st.subheader('需要確認的照片')
    if not pending:
        st.success('目前沒有會阻擋正式分數的局部補正；骨架剔除與一般品質提醒請查看相鄰分頁。')
        return
    st.warning(f'目前有 {len(pending)} 個待完成的局部確認。以下列出確切照片編號與評估側，直接選取就能修改。')
    table=pd.DataFrame(pending).rename(columns={'照片編號':'樣本編號'})
    st.dataframe(table[['樣本編號','評估側','類型','原因']],use_container_width=True,hide_index=True)
    chosen=st.selectbox('選擇待確認照片',range(len(pending)),
        format_func=lambda i:f"第 {pending[i]['照片編號']} 張｜{pending[i]['評估側']}｜{pending[i]['類型']}",
        key='rula_pending_choose')
    item=pending[chosen];sid=item['照片編號'];side=item['評估側']
    st.caption('需要處理的原因：'+item['原因'])
    if item['類型']=='頸部':
        shared_neck_review_panel('rula_pending',selected_sample=sid)
        return
    rula_show_sample_image(side,sid)
    if item['類型']=='前臂':
        _option=st.radio('此側前臂是否向外側工作？', ['尚未決定','有','沒有'],
                         horizontal=True,key=f'rula_pending_outside_{side}_{sid}')
        if st.button('儲存並重新計算這張照片',disabled=_option=='尚未決定',
                     type='primary',key=f'rula_pending_save_{side}_{sid}'):
            _p,_c=batch_update_profiles(st.session_state.get('batch_profiles',{}),
                st.session_state.get('batch_confirmed',set()),[(side,int(sid))],{'out_side':_option=='有'})
            st.session_state.batch_profiles=_p
            st.session_state.batch_confirmed=_c
            st.rerun()
    else:
        st.info('這是骨架或分數品質問題，請在下面的「異常／人工評分」或「剔除樣本」核對。')


def v4_show_review_center():
    st.subheader("異常影格審核／逐張人工修正／風險時間軸")
    st.caption('顯示系統自動計算與預設補正的系統評分；人工批次修改後會重新查表。骨架照片印字是分析當下系統計算值。')
    mode = st.session_state.analyzed_mode
    both = mode == "左右側同步計算"
    available, batch_status, _ = batch_current_effective(st.session_state)
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
                    v10_show_image(payload, caption, encoded=True)
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
        st.caption("以原始系統評分與可選的人工補正後分數對照；未經人工複核不代表已驗證準確。")
        side = st.selectbox("時間軸評估側", list(available), key="v4_chart_side")
        series = available[side].sort_values("時間(秒)").copy()
        prefix = "右手" if side == "右手側" else "左手"
        series["原始RULA"] = pd.to_numeric(series.get('原始自動Grand Score', series[f"{prefix} Grand Score"]), errors="coerce")
        series["修正後RULA"] = pd.to_numeric(series[f"{prefix} Grand Score"],errors="coerce")
        for idx, row in series.iterrows():
            edit = st.session_state.v4_overrides.get((side, int(row["樣本編號"])))
            if edit:
                series.loc[idx, "修正後RULA"] = edit["人工Grand Score"]
        chart = series[["時間(秒)", "原始RULA", "修正後RULA"]].set_index("時間(秒)")
        st.line_chart(chart, y_label="RULA 分數（1～7）", x_label="影片時間（秒）")
        st.dataframe(series[["樣本編號", "時間(秒)", "原始RULA", "修正後RULA"]], use_container_width=True, hide_index=True)

    st.info('人工修正或複核完成後，請在「📦 下載」分頁取得同一份整合報告。')


# ==================== 【V6：Pose / Pose+Hand 手腕演算法比較中心】 ====================
def v6_show_hand_comparison():
    st.subheader('手腕角度')
    st.caption('顯示實際採用的手腕角度與分數；系統優先使用 Hand 2D，必要時由 Pose 2D 備援。')
    is_dual=st.session_state.analyzed_mode == '左右側同步計算'
    sources=({'右手側':st.session_state.get('df_right_only'),
              '左手側':st.session_state.get('df_left_only')} if is_dual else
             {st.session_state.analyzed_mode:st.session_state.get('df_records')})
    sources={side:data for side,data in sources.items() if data is not None and not data.empty}
    if not sources:
        st.info('沒有可顯示的手腕結果。')
        return
    side=st.selectbox('評估側',list(sources),key='wrist_simple_side')
    frame=sources[side].sort_values('樣本編號')
    prefix='右手' if side=='右手側' else '左手'
    used=frame.get(f'{prefix}手腕計分演算法')
    if used is not None:
        st.caption(f'本次 Hand 使用 {int(used.astype(str).str.contains("Hand 2D").sum())} 筆；其餘依既定備援方式評分。')
    st.dataframe(user_report_wrist(frame,side),use_container_width=True,hide_index=True)
    if len(frame)>1:
        st.line_chart(frame.set_index('樣本編號')[[f'{prefix}手腕角度']],y_label='手腕角度（度）')
    ids=frame['樣本編號'].astype(int).tolist()
    chosen=st.selectbox('查看影格',ids,key='wrist_simple_sample')
    pair=st.session_state.get('hand_review_images',{}).get((side,int(chosen)))
    if pair:
        left,right=st.columns(2)
        with left:
            st.markdown('**人體骨架**')
            v10_show_image(pair[0],encoded=True)
        with right:
            st.markdown('**手部關鍵點**')
            v10_show_image(pair[1],encoded=True)
    st.caption('當手部被遮擋、關鍵點位置不合理時，請至「剔除樣本」或「異常／人工覆核」確認。')

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
        _,_,dynamic_neck_cases=batch_current_effective(st.session_state)
        pending_neck=[sid for sid,item in dynamic_neck_cases.items()
                      if item.get('需要人工確認') and sid not in st.session_state.get('shared_neck_decisions',{})]
        if pending_neck:
            st.warning(f'有 {len(pending_neck)} 筆頸部姿勢可能改變風險行動等級，尚未確認。請至「異常／人工覆核 → 頸部確認」只確認頸部分數。未確認者不列入可信統計。')
        if not st.session_state.get('v8_values',{}).get('work_conditions_verified',False):
            st.warning('尚未確認現場負重與肌肉使用加分；匯出資料將保留「待確認」註記。')
        _,batch_states,_=batch_current_effective(st.session_state)
        batch_unknown=sum(x.startswith('無法確認') for x in batch_states.values())
        if batch_unknown:
            st.warning(f'有 {batch_unknown} 筆側別的補正可能影響 RULA 總分，尚待確認；其餘外側工作不確定但總分不變者保留紀錄、不需逐張確認。')
        else:
            st.info('已依影片時段、影像辨識和預設作業條件產生 RULA 評分；僅在條件不同時才需批次修改。')
        (tab_overview, tab_review, tab_rejected, tab_gallery,
         tab_reports, tab_download) = st.tabs([
            '📈 風險總覽', '🔎 異常／人工覆核', '🧩 剔除樣本',
            '🖼️ 影格照片', '📋 完整報表', '📦 下載',
        ], key='v9_result_tabs', on_change='rerun')
        if tab_overview.open:
            with tab_overview:
                st.subheader('RULA 風險總覽')
                info=st.session_state.get('skipped_info',{})
                st.caption(f'攝影方式：{st.session_state.get("capture_mode","既有模式")}｜跳過影格仍保留於剔除樣本複核。')
                with st.expander('查看本次影片時段如何套用到每張照片',expanded=False):
                    if st.session_state.get('timeline_rules_used'):
                        st.caption('使用原始側面影片的播放秒數；未指定時段採用分析前預設分數。')
                        st.dataframe(pd.DataFrame([{'作業階段':x['name'],'開始(秒)':x['start'],
                            '結束(秒)':x['end'],'肌肉使用加分':x['muscle'],'負重施力加分':x['force']}
                            for x in st.session_state['timeline_rules_used']]),use_container_width=True,hide_index=True)
                    else:
                        st.caption('本次沒有另外指定影片時段，全部採用分析前作業條件。')
                    frame_map,_,_=batch_current_effective(st.session_state)
                    rows=[]
                    for side,df in frame_map.items():
                        if not isinstance(df,pd.DataFrame) or df.empty:continue
                        for _,r in df.iterrows():
                            rows.append({'照片編號':int(r['樣本編號']),'評估側':side,
                                '側面影片秒數':r.get('參考側影片時間(秒)'),
                                '肌肉使用':r.get('肌肉使用加分'),'負重／施力':r.get('負重施力加分'),
                                '設定來源':r.get('作業條件設定來源','預設條件')})
                    if rows:
                        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                if st.session_state.get('v9_sync_summary') is not None:
                    with st.expander('本次鏡頭時間同步結果',expanded=False):
                        st.dataframe(st.session_state.v9_sync_summary,use_container_width=True,hide_index=True)
                reviewed, reviewed_combined=v113_current_reports()
                if (df_show is None or df_show.empty) and reviewed_combined.empty:
                    st.warning('目前尚無可用系統評分結果；可到「剔除樣本」查看品質或辨識失敗原因。')
                else:
                    if mode_now=='左右側同步計算':
                        if reviewed_combined.empty:
                            effective=pd.Series(dtype='float64')
                        else:
                            full=reviewed_combined[reviewed_combined['雙側結果狀態']=='完整雙側有效']
                            effective=pd.to_numeric(full['完整雙側最高分'],errors='coerce').dropna()
                        st.metric('完整雙側可用評分',len(effective))
                        left_ok=int(((reviewed['評估側']=='左手側') & (reviewed['是否納入可信統計'])).sum()) if not reviewed.empty else 0
                        right_ok=int(((reviewed['評估側']=='右手側') & (reviewed['是否納入可信統計'])).sum()) if not reviewed.empty else 0
                        st.caption(f'側別可用：右側 {right_ok} 筆、左側 {left_ok} 筆。若僅單側可用，完整雙側最高分留空，但該側原始資料與複核結果保留。')
                    else:
                        reviewed_ok=reviewed[reviewed['是否納入可信統計']] if not reviewed.empty else pd.DataFrame()
                        effective=pd.to_numeric(reviewed_ok['複核後Grand Score'],errors='coerce').dropna() if not reviewed_ok.empty else pd.Series(dtype='float64')
                    a,b,c=st.columns(3)
                    a.metric('可用系統評分',len(effective))
                    b.metric('平均 Grand Score',f'{effective.mean():.2f}' if not effective.empty else '—')
                    c.metric('AL3／AL4 比例',f'{effective.ge(5).mean()*100:.1f}%' if not effective.empty else '—')
                    if not effective.empty:
                        st.metric('最高分',int(effective.max()))
                    if mode_now=='左右側同步計算' and not reviewed_combined.empty:
                        full=reviewed_combined[reviewed_combined['雙側結果狀態']=='完整雙側有效']
                        if not full.empty:
                            st.line_chart(full.set_index('樣本編號')[['完整雙側最高分']],y_label='RULA Grand Score')
                        st.dataframe(user_report_reviewed(reviewed_combined,combined=True),use_container_width=True,hide_index=True)
                    elif not reviewed.empty:
                        st.line_chart(reviewed[reviewed['是否納入可信統計']].set_index('樣本編號')[['複核後Grand Score']],y_label='RULA Grand Score')
                    else:
                        st.warning('目前尚無可用的系統評分。')
                    st.caption('這裡是可用系統評分的描述性統計，可能包含未逐張人工確認的補正；若品質或頸部分級仍待必要複核會排除。')
                    st.caption('平均數僅作描述性統計，不能取代 RULA 對高風險姿勢的判斷。')
                st.info('時間欄位說明：「共同時間」以同步事件為 0 秒；「分析經過時間」從取樣開始算 0 秒。若無正面影片，正面視角相關補正由使用者人工確認。')
        if tab_review.open:
            with tab_review:
                review_pending, review_photos, review_neck, review_angles, review_other = st.tabs([
                    '📌 待確認照片', '✅ 照片補正', '🧠 頸部確認',
                    '📐 角度檢視', '🚩 異常／人工評分',
                ],key='rula_nested_review_tabs',on_change='rerun')
                if review_pending.open:
                    with review_pending:rula_pending_photo_panel()
                if review_photos.open:
                    with review_photos:batch_photo_panel()
                if review_neck.open:
                    with review_neck:v8_show_neck_center()
                if review_angles.open:
                    with review_angles:
                        st.caption('手腕角度採 Hand 2D 優先、Pose 2D 備援。頸部角度與共用分數請至「頸部確認」。')
                        v6_show_hand_comparison()
                if review_other.open:
                    with review_other:v4_show_review_center()
        if tab_rejected.open:
            with tab_rejected:
                v8_show_rejected_center()
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
                    v10_show_image(payload, cap, encoded=True)
                    if st.checkbox('展開所有縮圖（耗用較多記憶體）',key='v8_all_photo_thumbs'):
                        cols=st.columns(3)
                        for i,(caption,payload) in enumerate(images):
                            with cols[i%3]:
                                v10_show_image(v10_thumbnail_b64(payload), caption, encoded=True)
                else:
                    st.info('沒有可顯示的有效樣本照片。剔除樣本請到「剔除樣本」分頁檢查。')
        if tab_reports.open:
            with tab_reports:
                st.subheader('評估報表')
                reviewed,reviewed_combined=v113_current_reports()
                if mode_now=='左右側同步計算':
                    table_choice=st.radio('報表類型',['複核後雙側彙整','複核後逐側','原始左右共同','原始右側','原始左側'],horizontal=True,key='v8_report_choice')
                    table={'複核後雙側彙整':reviewed_combined,'複核後逐側':reviewed,
                           '原始左右共同':df_show,'原始右側':st.session_state.df_right_only,
                           '原始左側':st.session_state.df_left_only}[table_choice]
                else:
                    table_choice=st.radio('報表類型',['複核後逐側','原始分析'],horizontal=True,key='v8_report_choice')
                    table=reviewed if table_choice=='複核後逐側' else df_show
                if table is not None and not table.empty:
                    if table_choice=='原始右側':
                        simple=user_report_side(table,'右手側')
                    elif table_choice=='原始左側':
                        simple=user_report_side(table,'左手側')
                    elif table_choice=='原始左右共同':
                        simple=user_report_dual(table)
                    elif table_choice=='原始分析':
                        simple=user_report_side(table,mode_now)
                    elif table_choice=='複核後雙側彙整':
                        simple=user_report_reviewed(table,combined=True)
                    else:
                        simple=user_report_reviewed(table)
                    st.dataframe(simple,use_container_width=True,hide_index=True)
                    st.caption('僅保留評分結果與實際採用角度。更詳細的研發診斷資訊不混入一般報表。')
                else:
                    st.warning('目前這個報表沒有可用資料，請查看其他類型。')
        if tab_download.open:
            with tab_download:
                st.subheader('下載評估資料')
                st.caption('檔名：'+st.session_state.zip_filename)

                if st.session_state.zip_bytes:
                    initial_zip=st.session_state.zip_bytes
                    if st.session_state.v4_overrides:
                        if mode_now=='左右側同步計算':
                            source_dfs={'右手側':st.session_state.df_right_only,'左手側':st.session_state.df_left_only}
                        else:
                            source_dfs={mode_now:st.session_state.df_records}
                        anomaly_parts=[v4_mark_anomalies(data,side) for side,data in source_dfs.items()
                                      if data is not None and not data.empty]
                        anomaly=pd.concat(anomaly_parts,ignore_index=True) if anomaly_parts else pd.DataFrame()
                        initial_zip=v4_make_zip(initial_zip,st.session_state.v4_overrides,anomaly)
                    side_frames,batch_status,dynamic_cases=batch_current_effective(st.session_state)
                    finalized=v113_finalize_report_zip(initial_zip,side_frames,
                        st.session_state.get('v8_rejected_rows',[]),
                        st.session_state.get('v113_review_decisions',{}),
                        st.session_state.get('v4_overrides',{}),
                        st.session_state.get('v8_manual_rejected',{}),
                        dynamic_cases,
                        st.session_state.get('shared_neck_decisions',{}),
                        batch_status=batch_status,
                        batch_audit=batch_audit_table(batch_source_frames(st.session_state),
                            st.session_state.get('batch_profiles',{}),st.session_state.get('batch_confirmed',set())),
                        work_conditions=st.session_state.get('v8_values',{}),
                        )
                    render_blob_download_button(finalized,st.session_state.zip_filename,
                                                f'📥 下載 {mode_now} 原始＋複核後報表 ZIP')
                st.caption('ZIP 的「原始分析」保留首次系統計算結果；「複核後結果」包含沿用設定的系統初始與人工修正結果，並標明來源。未經人工驗證不應稱為準確率。')
# 只有分析前四步才顯示底部「上一步／下一步」；結果頁使用分頁，不再有步驟按鈕。
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
