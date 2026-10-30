"""Optional-Front regression checks for selected-side mandatory camera paths.
Execute: python tests/test_optional_front.py
No Streamlit or MediaPipe required; this is not a real-video inference test.
"""
import ast
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

APP=Path(__file__).resolve().parents[1]/'222.py'
source=APP.read_text(encoding='utf-8')
tree=ast.parse(source)
funcs={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
names=['rula_eval_sides','rula_required_cameras','v9_sync_signature',
       'v9_frame_meta','v9_common_window','v9_frame_index','v9_camera_seconds',
       'v9_make_sample_times','v9_sample_row','v9_sync_report','v9_best_motion_lag','v9_suggest_anchors']
for n in names: assert n in funcs,n
ns={'np':np,'pd':pd,'V9_AUTO_RATE_HZ':4.0,'V9_AUTO_MAX_LAG_SECONDS':9.0}
exec(compile(ast.Module(body=[funcs[n] for n in names],type_ignores=[]),str(APP),'exec'),ns)

class Video:
    def __init__(self,label): self.name=label+'.mp4'; self.raw=label.encode()
    def getvalue(self): return self.raw

modes={
    '右手側': ['Right'],
    '左手側': ['Left'],
    '左右側同步計算': ['Right','Left'],
}
for mode, required in modes.items():
    for has_front in (False,True):
        uploads={cam:Video(cam) for cam in required}
        uploads['Front']=Video('Front') if has_front else None
        cams=ns['rula_required_cameras'](mode,uploads)
        assert cams==(['Front'] if has_front else [])+required, (mode,has_front,cams)
        assert ns['rula_eval_sides'](mode)==({'右手側':('右手側',),'左手側':('左手側',),
                                           '左右側同步計算':('右手側','左手側')}[mode])
        signature=ns['v9_sync_signature'](uploads,mode)
        assert len(signature)==len(cams)
        # Adding Front must invalidate previously confirmed sync signature.
        changed={**uploads,'Front':None if has_front else Video('Front')}
        assert ns['v9_sync_signature'](changed,mode)!=signature
        meta={cam:ns['v9_frame_meta'](30+i*5,900+i*43) for i,cam in enumerate(cams)}
        anchor={cam:0.6+i*.3 for i,cam in enumerate(cams)}
        ts,start,end=ns['v9_make_sample_times'](5,meta,anchor,cams)
        assert len(ts)==5 and end>start
        assert all(np.diff(ts)>0)
        record=ns['v9_sample_row'](ts[0],meta,anchor,cams,1,start)
        assert record['分析經過時間(秒)']==0
        assert all(f'{cam}影格索引' in record for cam in cams)
        if not has_front: assert 'Front影格索引' not in record
        report=ns['v9_sync_report'](meta,anchor,cams,start,end,'預覽確認')
        assert set(report['鏡頭'])==set(cams)
        print('PASS:',mode,'Front' if has_front else '無 Front',cams)

# No-front single-view can use camera time zero, with no absent Front path.
for cam in ('Right','Left'):
    cams=[cam]; meta={cam:ns['v9_frame_meta'](25,300)}; anchors={cam:0.0}
    ts,lo,hi=ns['v9_make_sample_times'](10,meta,anchors,cams)
    assert lo==0 and len(ts)==10
    assert ns['v9_sample_row'](ts[0],meta,anchors,cams,1,lo)[cam+'影格索引']==0

# Static validation for operational branches and mandatory-side safety gates.
for fragment in [
    "is_factory_mode = vid_front_file is None",
    "if missing_sides:",
    "未上傳正面影片：沿用分析前的人工預設",
    "front_mods_r=manual_without_front('右手側')",
    "front_mods_l=manual_without_front('左手側')",
    "front_mods = manual_without_front(target_side)",
    "cap_front = cv2.VideoCapture(t_front.name) if t_front is not None else None",
    "if cap_front is None:",
    "'has_front_view': cap_front is not None",
    "'has_front_view':not is_factory_mode",
    "'評估結果/無正面人工補正確認.csv'",
    "正面鏡頭（選填，建議上傳）",
    "RULA_VERIFIED_TABLES_FRONT_REVIEW_2026-10-30",
]:
    assert fragment in source, fragment
assert '正面鏡頭（必要）' not in source
assert '工廠雙側同步計算（僅右側＋左側，無正面影片）' not in source
# Verify manual flag is respected by side score routine even without front input.
assert 'if not auto_selection.get(flag, True):' in source
assert 'front_mods[flag] = bool(manual_front.get(flag, False))' in source
print('PASS optional front modes, timestamps, validation and report safety checks')
