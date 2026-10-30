"""Pose World unavailable must not discard otherwise valid 2D side frame."""
from pathlib import Path
import runpy
from types import SimpleNamespace
import numpy as np
p=Path(__file__).resolve().parent/'test_v114_integration.py'
v=runpy.run_path(str(p));env=v['ns']
record=next(x for x in v['rows'] if x['評估側']=='左手側')
pose=v['fake_result'](record)
pose.pose_world_landmarks=None
env['detect_pose_with_fallback']=lambda *_args,**_kwargs:(pose,None,None)
frame=np.zeros((record['影像高'],record['影像寬'],3),dtype=np.uint8)
front={k:False for k in ('is_abducted','is_across_midline','is_out_to_side','is_side_bending','is_neck_side_bending')}
manual={'auto_selection':{},'leg_score':1,'wrist_twist_score':1,'neck_v114_mode':env['V114_NECK_MODES'][1]}
succ,data,_,conf=env['analyze_side_camera'](frame,None,'左手側',front,0,0,manual,None)
assert succ and data['左手 Grand Score'] in range(1,8),(succ,data)
print('PASS side camera accepts valid 2D pose even when Pose World is missing')
