"""在可連網電腦預先下載 Google MediaPipe Hand Landmarker 模型至專案 models/。
執行：python scripts/download_hand_model.py
"""
from pathlib import Path
import urllib.request
import os

URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DEST = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
DEST.parent.mkdir(parents=True, exist_ok=True)
if DEST.is_file() and DEST.stat().st_size > 1_000_000:
    print(f"檔案已存在：{DEST} ({DEST.stat().st_size:,} bytes)")
else:
    part = DEST.with_suffix(".task.partial")
    try:
        with urllib.request.urlopen(URL, timeout=45) as inp, open(part, "wb") as out:
            while chunk := inp.read(128 * 1024):
                out.write(chunk)
        if part.stat().st_size < 1_000_000:
            raise ValueError("下載檔案太小，疑似不完整")
        os.replace(part, DEST)
        print(f"模型下載完成：{DEST} ({DEST.stat().st_size:,} bytes)")
    finally:
        if part.exists():
            part.unlink()
