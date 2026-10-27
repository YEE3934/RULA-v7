"""No-Cloud tests of v9.2: sync clip, elapsed timestamps, Pose wrist QC and image markup."""
import ast, base64
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
path=Path(__file__).resolve().parents[1]/'222.py'
source=path.read_text('utf-8')
needed={'v9_frame_meta','v9_common_window','v9_make_sample_times',
        'v9_camera_seconds','v9_frame_index','v9_sample_row','v92_pose_wrist_qc',
        'v92_image_html','v92_thumbnail_b64'}
tree=ast.parse(source)
functions=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in needed]
assert {x.name for x in functions}==needed
ctx={'np':np,'pd':pd,'cv2':cv2,'base64':base64}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),ctx)
g=lambda name:ctx[name]
meta={c:g('v9_frame_meta')(fps,count) for c,fps,count in [('Front',29.98,648),('Right',30.02,651),('Left',30.01,654)]}
anchors={'Front':19.5,'Right':19.75,'Left':19.75}
full_begin,full_end=g('v9_common_window')(meta,anchors,list(meta))
assert full_begin==-19.5 and full_end>1.0
sample,start,end=g('v9_make_sample_times')(12,meta,anchors,list(meta),(-8.0,-1.0))
assert abs(start+8)<1e-6 and abs(end+1)<1e-6
row=g('v9_sample_row')(sample[0],meta,anchors,list(meta),1,analysis_start=start)
assert row['時間(秒)']==0 and row['共同時間(秒)']==-8
assert abs(row['Front影片時間(秒)']-11.5)<.01
row2=g('v9_sample_row')(sample[-1],meta,anchors,list(meta),12,analysis_start=start)
assert row2['時間(秒)']==7
assert row2['共同時間(秒)']==-1
try:
 g('v9_make_sample_times')(10,meta,anchors,list(meta),(4.0,5.0))
 raise AssertionError('Outside clip accepted')
except ValueError: pass

df=pd.DataFrame({'樣本編號':[8,9,10,11,12],
                 '右手Pose舊版手腕角度':[12,71,11,18,20]})
q=g('v92_pose_wrist_qc')(df,'右手側')
assert q['Pose手腕需複核（保守）'].tolist()==[False,True,False,False,False]
assert q['右手Pose舊版手腕角度'].tolist()==df['右手Pose舊版手腕角度'].tolist()
# Real motion can change drastically between adjacent frames: no spike if neighbors disagree.
q=g('v92_pose_wrist_qc')(pd.DataFrame({'樣本編號':[1,2,3],
    '右手Pose舊版手腕角度':[10,47,51]}),'右手側')
assert not any(q['Pose手腕需複核（保守）'])

img=np.zeros((800,250,3),dtype=np.uint8)
img[:,20:30]=200
ok,buf=cv2.imencode('.jpg',img)
assert ok
markup=g('v92_image_html')(buf.tobytes(),'縱向影格')
assert 'data:image/jpeg;base64,' in markup
assert 'height:auto' in markup and 'object-fit:contain' in markup
thumb=g('v92_thumbnail_b64')(base64.b64encode(buf).decode(),max_width=130)
decoded=cv2.imdecode(np.frombuffer(base64.b64decode(thumb),dtype=np.uint8),cv2.IMREAD_COLOR)
assert decoded.shape[1]==130 and decoded.shape[0]>130
# Regression: avoid fixed-height iframe thumbnail, Streamlit media path and negative analysis time.
assert 'components.html(f\'<img src="data:image/jpeg;base64,{payload}" style="width:100%"/>' not in source
assert "st.image(picture,caption=" not in source
assert "sec = round(float(sample_time - sync_start), 2)" in source
print('PASS: v9.2 sync trim, clean elapsed time, source frame mapping, Pose wrist conservative QC, full aspect ratio JPEG')
