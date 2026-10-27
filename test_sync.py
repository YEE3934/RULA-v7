"""Unit and integration smoke tests for RULA v9 time synchronization.
Run: python tests/test_sync.py (from project root; opencv-python required).
Uses AST to test pure helpers without importing Streamlit or downloading MediaPipe.
"""
import ast
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile
import cv2
import numpy as np
import pandas as pd

SOURCE=Path(__file__).resolve().parents[1] / '222.py'
source_text=SOURCE.read_text(encoding='utf-8')
node=ast.parse(source_text)
keep={
 'v9_frame_meta','v9_common_window','v9_camera_seconds','v9_frame_index',
 'v9_read_frame','v9_motion_features','v9_best_motion_lag','v9_suggest_anchors',
 'v9_sync_report','v9_make_sample_times','v9_sample_row','v9_decorate_records'
}
funcs=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in keep]
assert set(n.name for n in funcs)==keep
namespace={'np':np,'cv2':cv2,'pd':pd,'V9_AUTO_RATE_HZ':4.0,'V9_AUTO_MAX_SECONDS':35.0,
           'V9_AUTO_MAX_LAG_SECONDS':9.0}
exec(compile(ast.Module(body=funcs,type_ignores=[]),str(SOURCE),'exec'),namespace)
g=lambda x:namespace[x]

def close(a,b,eps=1e-3):
 assert abs(a-b)<eps,(a,b)

# Different start times, different FPS, different durations
m={'Front':g('v9_frame_meta')(30,750),
   'Right':g('v9_frame_meta')(24,672),
   'Left':g('v9_frame_meta')(60,1440)}
a={'Front':2.0,'Right':3.5,'Left':1.2}
cams=list(m)
times,start,end=g('v9_make_sample_times')(50,m,a,cams)
close(start,-1.2)
assert end>22.0
assert len(times)==50 and all(np.diff(times)>0)
row=g('v9_sample_row')(5,m,a,cams,8)
close(row['Front影片時間(秒)'],7.0)
close(row['Right影片時間(秒)'],8.5)
close(row['Left影片時間(秒)'],6.2)
assert row['Front影格索引']==210
assert row['Right影格索引']==204
assert row['Left影格索引']==372
assert g('v9_frame_index')(5,a,'Right',m)==204

# Single side: only two required cameras
st,ed=g('v9_common_window')(m,a,['Front','Right'])
assert st==-2.0
assert ed>22.0

# No overlap or invalid anchor fails safely
try:
 g('v9_common_window')(m,{'Front':(m['Front']['last_time']),'Right':0.0},['Front','Right'])
 raise AssertionError('Should fail on no overlap')
except ValueError:
 pass
try:
 g('v9_common_window')(m,{'Front':30,'Right':3.5},['Front','Right'])
 raise AssertionError('Should fail on impossible anchor')
except ValueError:
 pass

# Known synthetic event pattern with independent noise; lag sign correctness
rate=4
r=np.random.default_rng(19)
shared=np.zeros(200)
for i,p in enumerate([8,22,40,58,80,105,127]):
 shared[p:min(p+2,len(shared))] += 5 + i*0.8
f=np.concatenate([np.zeros(8),shared[:152]]) + r.normal(0,.12,160)
rr=np.concatenate([np.zeros(14),shared[:146]]) + r.normal(0,.12,160)
ll=np.concatenate([np.zeros(4),shared[:156]]) + r.normal(0,.12,160)
res_r=g('v9_best_motion_lag')(f,rr)
res_l=g('v9_best_motion_lag')(f,ll)
assert res_r['usable'] and res_l['usable']
close(res_r['lag_seconds'],1.5,eps=.26)
close(res_l['lag_seconds'],-1.0,eps=.26)
recommend=g('v9_suggest_anchors')({'Front':f,'Right':rr,'Left':ll},m,cams)
assert recommend['usable']
close(recommend['anchors']['Right']-recommend['anchors']['Front'],1.5,eps=.26)
close(recommend['anchors']['Left']-recommend['anchors']['Front'],-1.0,eps=.26)

# Real video decoding with varied frame rates + event in frame (OpenCV only).
with tempfile.TemporaryDirectory() as temp_dir:
 paths=[]
 for cam,fps,frames in [('Front',30,75),('Right',24,90),('Left',15,60)]:
  path=os.path.join(temp_dir,cam+'.avi')
  writer=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'MJPG'),fps,(240,160))
  assert writer.isOpened(),cam
  for i in range(frames):
   pic=np.full((160,240,3),80,dtype=np.uint8)
   x=int(20+170*((i%fps)/(fps-1)))
   cv2.rectangle(pic,(x,40),(min(239,x+20),70),(225,225,225),-1)
   writer.write(pic)
  writer.release()
  paths.append(path)
 for path in paths:
  motion=g('v9_motion_features')(path,limit_seconds=2.0,rate_hz=4.0)
  assert len(motion)>4 and np.isfinite(motion).all(),path
  cap=cv2.VideoCapture(path)
  success,frame=g('v9_read_frame')(cap,12)
  assert success and frame.shape[:2]==(160,240)
  cap.release()

# Full output mapping can be zipped and read back
out=g('v9_sync_report')(m,a,cams,start,end,'預覽確認')
rows={i+1:g('v9_sample_row')(t,m,a,cams,i+1) for i,t in enumerate(times)}
other=[{'樣本編號':8,'時間(秒)':5,'評估側':'右手側'}]
g('v9_decorate_records')(other,rows)
assert 'Right影片時間(秒)' in other[0]
with zipfile.ZipFile(io.BytesIO(),mode='w') as zf:
 zf.writestr('Synchronization/Synchronization_Report.csv',out.to_csv(index=False).encode('utf-8-sig'))
 zf.writestr('Synchronization/Frame_Time_Map.csv',pd.DataFrame(rows.values()).to_csv(index=False).encode('utf-8-sig'))

# Major regression checks
assert source_text.index('def add_hand_temporal_notes(')<source_text.index('df_comb = pd.DataFrame(combined_records)')
assert 'RULA_V9_1_SETUP_WIZARD_RESULTS_TABS' in source_text
assert 'v9_confirmed_signature' in source_text
assert 'v9_make_sample_times(' in source_text
assert '同步事件出現時間(秒)' in source_text
assert 'Synchronization/Frame_Time_Map.csv' in source_text
print('PASS: time offsets, mixed FPS, shared interval, automatic lag, real video read, ZIP and regression checks')
