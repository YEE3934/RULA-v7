這個資料夾可選擇放官方手部模型：hand_landmarker.task
來源（Google 官方）：
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

放入 models/hand_landmarker.task 並隨 222.py 上傳 GitHub，最穩定。
如果不放，222.py 首次啟用 Hand 時會嘗試下載到 Streamlit Cloud 可寫入的 /tmp。
如果外部網路不可用，Hand 功能會提示並退回原本 Pose 評估。

注意：模型檔沒有包含於此 ZIP；此環境無法取得二進位模型。
