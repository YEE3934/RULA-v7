# RULA v9.1｜六步驟設定＋結果分頁

部署版本：`RULA_V9_1_SETUP_WIZARD_RESULTS_TABS_2026-10-28`

## 這次修正什麼？

這份是以先前的 **RULA v9 三鏡頭同步完整程式**為基礎的介面更新，保留 v9 的時間同步與既有 RULA 計分方式。

1. **設定階段只有 6 步：**
   1. 影片與基本設定
   2. 三鏡頭時間同步
   3. 自動辨識設定
   4. 人工條件
   5. Hand 實驗設定
   6. 啟動分析
2. **設定頁頂端只顯示藍色進度條與當前步驟**；不再顯示頂端的上一步／下一步按鈕。
3. **設定階段的上一步／下一步按鈕只出現在頁面底部**；最後一步沒有「下一步」，請按頁面內的「🚀 啟動全關節生物力學分析」。
4. **分析成功即切換到結果模式**，不會把風險總覽當成第 7 步。
5. **結果模式改成可以直接點的分頁**，包含：
   - 📈 風險總覽（包含時間同步結果、分數時間軸）
   - 🔎 異常／人工覆核（包含原有重新計分和覆核時間軸）
   - 🧩 剔除樣本（雙照片對照及人工註記）
   - 📐 頸部／手腕比較（內含兩個子分頁）
   - 🖼️ 影格照片
   - 📋 完整報表
   - 📦 下載（含完整 RULA 分析和診斷 ZIP）
6. **按需載入**：結果分頁使用 `st.tabs(..., on_change="rerun")` 和 `.open`；只有被點選的分頁才會執行重型表格、圖片和 ZIP 建立，減少 Streamlit Cloud 的負擔。
7. **可以返回設定**：分析完成後左側可按「⚙️ 返回設定／重新分析」；上次結果仍在工作階段內，只有再次分析才會更新結果。
8. **切換設定步驟保留影片與勾選**：沿用 v9 `st.session_state.v8_values` / `v8_uploads` 保存方式。工作階段重設或 Streamlit Cloud 休眠後，仍需重新上傳影片。

## 部署方式

1. 將本資料夾內 `222.py`、`requirements.txt`、`packages.txt`、`models/`、`scripts/` 和 `tests/` 放入 GitHub 專案**根目錄**。
2. Streamlit Cloud 選擇 **Python 3.11**，並將 Main file path 指向 `222.py`。
3. `models/hand_landmarker.task` 需要保留在你原本的模型資料夾。部署包不包含 Google 的模型二進位檔；如缺少模型，可使用 `python scripts/download_hand_model.py`，或由程式嘗試下載到 `/tmp`。
4. 檢查網頁最上方的部署標記為 `RULA_V9_1_SETUP_WIZARD_RESULTS_TABS_2026-10-28`。
5. 用先前同一組三視角影片，按順序設定 → 確認時間同步 → 啟動分析。成功後直接切換各個結果分頁，核對圖片、分數及 ZIP 下載。

## 套件

沿用 v9 原本已成功運行的 Python 套件清單，沒有新增 AI 模型或變更 OpenCV/MediaPipe 版本。結果分頁的按需載入需要 **Streamlit 1.55 或更新版本**；目前 Streamlit Cloud 一般安裝的新版已支援。

## 回歸測試

```bash
python -m py_compile 222.py
python tests/test_sync.py
python tests/test_ui_mock.py
```

`test_sync.py` 核對同步時間映射、自動動作偏移建議和 ZIP 時間資料；`test_ui_mock.py` 驗證六個設定頁、結果分頁及子分頁、影片狀態保存與舊 session 的結果頁兼容。這些是離線測試，並未在你的 Streamlit Cloud 使用真實三視角影片完成端到端測試。

## 保留與未修改的部分

- MediaPipe Pose、Hand Landmarker 和各關節角度計算
- Table A／B／C、RULA 手動條件及計分公式
- 三鏡頭自動同步建議、手動錨點校正及重疊區間抽樣
- 被剔除樣本雙照片、頸部候選角、Hand 角度比較、人工覆核和完整 ZIP

**注意：** RULA 查表矩陣和角度真值仍需要後續獨立驗證。本次改善的是網頁導航和結果載入行為，不應描述為角度或 RULA 評分準確度提升。
