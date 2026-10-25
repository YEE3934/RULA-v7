# RULA v7.2｜時序函式修正與部署驗證

## 這次看到的錯誤

`NameError: name 'add_hand_temporal_notes' is not defined`，而且 traceback 是 `222.py:1758`。
這是 **v7 舊版** 的行號：原版在第 1758 行使用函式，卻要到第 1797 行才定義。

**v7.2** 已把函式放在呼叫之前、保留原本 RULA/Hand 分析邏輯，並新增啟動自檢。

## 只需要更新哪些檔案？

- **必須**：GitHub 儲存庫根目錄的 `222.py`。請完整覆蓋，不要貼局部程式碼。
- `requirements.txt` 和 `packages.txt` **不用修改**；包內只是附上與你先前成功部署一致的設定。
- Hand 模型等其他內容維持原本狀態。

## GitHub 操作檢查表

1. 下載此 ZIP，解壓縮後開啟 ZIP **根目錄**的 `222.py`。
2. GitHub → `rula-v6` 儲存庫 → Streamlit Cloud 指向的分支 → **根目錄 `222.py`** → Edit/Upload → 完整覆蓋 → Commit changes。
3. 在 GitHub 打開 `222.py`，搜尋 `RULA_V7_2_DEPLOY_VERIFIED_2026-10-26` 和 `def add_hand_temporal_notes`。兩者都必須存在。
4. 回到 Streamlit Cloud 的應用程式設定，確認 Repository、Branch、Main file path=`222.py` 完全對應。必要時 Reboot app。
5. 網站頂部應出現 `RULA_V7_2_DEPLOY_VERIFIED_2026-10-26`，並出現「✅ v7.2 部署自檢通過」。
6. **若 traceback 仍然顯示 `222.py:1758`，表示執行的仍是舊版 v7，而不是這份檔案**。先查部署路徑，不要繼續更改 RULA 算法。

## 離線測試

`python -m py_compile 222.py`
`python tests/check_deploy.py`
`python tests/test_v7.py`

> MediaPipe 的 `GPU support is not available`、`Feedback manager ...` 目前只是訊息/警告，與這次 NameError 無直接關係。
> 已驗證語法與離線邏輯；沒有在你的 Streamlit Cloud 進行真實影片端到端測試。
