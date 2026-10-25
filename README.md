# RULA v7 — 精簡介面與手腕幾何品質實驗版

以已部署成功的 v6 Pose + Hand 版本修改，**不需要重新建立 RULA 系統**。

## 1. 在 Streamlit Cloud 部署

1. 將本資料夾的 **`222.py`、`requirements.txt`、`packages.txt`** 以及 `scripts/`、`models/` 上傳至實驗版 GitHub 儲存庫根目錄。
2. Cloud 的 Main file path 設定為 `222.py`；建議 Python 3.11（與先前成功部署的環境相同）。
3. 進入網頁後檢查頂部版本標記 `RULA_V7_WRIST_QC_TABBED_2026-10-25`。
4. 將「腿足支撐」及評估側「手腕扭轉」於 **人工確認** 分頁設定，並在 **影片與基本設定** 上傳正面與所需側面影片。
5. 建議先選 **Hand 安全比較模式**，避免新角度影響正式 RULA；若要測試以 Hand 計分，僅在實驗網站啟用實驗計分模式。
6. 按下分析按鈕後，於結果的六個分頁檢查及下載 CSV／影格照片／品質資料。

### requirements.txt

```text
streamlit
mediapipe==0.10.21
numpy==1.26.4
opencv-contrib-python==4.11.0.86
pandas
Pillow
```

### packages.txt

```text
libgl1
libglib2.0-0
```

> **特別注意：** `libglib2.0-0` 是你先前 Streamlit Cloud 缺少 `libgthread-2.0.so.0` 後加入的必要 Linux 套件。本次延用已成功部署的相容版本。不要再同時安裝 `opencv-python-headless`，以免和 `opencv-contrib-python` 共用 `cv2` 造成衝突。

### 手部模型

程式沿用 `mediapipe==0.10.21`，不需要另外安裝 `hand` Python 套件；但需要 `hand_landmarker.task` 模型檔。

可用 `python scripts/download_hand_model.py` 下載，或至官方：
`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`

放置路徑為 `models/hand_landmarker.task`。若未事先放入，網站會嘗試下載到系統可寫入的暫存目錄 `/tmp/rula_hand_assets/`。若下載失敗，會退回原本 Pose-only 分析。

## 2. 新版介面

**分析前：** 影片與基本設定／自動辨識條件／人工確認／Hand 實驗（四個分頁）。

**分析後：** 風險總覽／異常與人工覆核／手腕比較／完整報表／影格照片／下載（六個分頁）。

原有的三視角左右雙側評估、RULA 分數、人工逐影格修改、ZIP 下載、角度時間軸、異常影格清單全部保留。照片改成逐張瀏覽，只有開啟「全部照片縮圖」才載入整份相簿，以降低網頁擁擠。

## 3. 本次角度算法與資料修改

- **上臂與前臂**：評估側關節角使用評估側真實肩點；不再把左右肩混合位置當成肩關節頂點。
- **軀幹與頸部**：改用雙肩／雙髖中心作為身體參考線，仍只是單鏡頭 **2D 投影代理角度**；彎腰可能超過 90°，不應直接理解為實際脊椎角度。
- **手腕公平比較**：分別輸出 **Pose 2D、Pose 3D、原版 Pose 混合（0.6×2D+0.4×3D）、Hand 2D**。請優先比較相同影像平面的 **Pose 2D vs Hand 2D**。
- **Hand 幾何品質**：檢查 Hand／Pose 手腕距離、手掌與前臂像素比例、左右手區分、目標 Pose 肘腕可見度、Pose2D/Hand2D 差異、極端手腕角度；不能通過時退回原版 Pose 計分。這些門檻是保守工程篩檢，**不是經過臨床或量角器驗證的準確率**。
- **同條件 RULA 敏感度比較**：在保留其他關節分數、扭轉、負重與肌肉加分的前提下，獨立試算 Pose 手腕與 Hand 手腕的 Grand Score；不覆蓋原始分數，不可視為模型精度驗證。
- **Hand 時序孤立尖峰**：比較三個有效、連號抽樣影格，只在前後相近、中央突出的情況標記；僅供資料診斷，**不自動改分或判斷動作錯誤**。
- **Keypoint JSONL**：`Diagnostics/Keypoint_Coordinates.jsonl` 保存評估側取樣影格的 Pose 影像座標、可見度及匹配到的 Hand 影像像素座標；若未辨識到 Hand，手部座標為 `null`。
- **可選原始影格**：`Raw_Frames/` 未畫骨架之 JPEG；可與 `Hand_Comparison/` 和已標記骨架圖對照。照片內可能含人物，分享前請留意權限。
- **角度說明**：`Diagnostics/V7_Angle_Definitions.txt` 說明每個角度的座標來源與研究限制。

## 4. 最重要的實驗建議

請用 v6 和 v7 **同一組影片、同一個評估側、同樣的人工負重設定** 各跑一遍。核對第 9、13、25、42 筆的 Pose2D／Hand2D 差距及骨架位置，再看上臂／前臂／頸部／軀幹的角度差異。**角度和模型較一致不等於比較準確**，最終仍需要少量參考真值。

**RULA Table A/B/C** 本版保留 v6 既有查表矩陣，仍未完成與同一份標準 RULA 工作表逐格核對，這是日後正式宣稱符合標準前的必要工作。

## 5. 測試內容與限制

執行 `python -m py_compile 222.py` 檢查語法；`python tests/test_v7.py` 檢查幾何門檻、時序與 RULA 查表輸出範圍；`python tests/test_app_mock.py` 使用模擬 Streamlit/MediaPipe 物件測試設定分頁與單側計算。這些都是**離線程式測試**，不能取代 Streamlit Cloud 真實影片端到端測試，也不能證明人體角度準確度提升。

如果要沿用舊版，保留 v6 的 GitHub 或 Cloud App，將 v7 上傳到新的實驗網站最保險。
