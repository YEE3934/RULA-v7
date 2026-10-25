"""Offline synthetic regression tests; does not claim measurement accuracy."""
import ast
import io
import json
import math
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.joinpath('222.py').read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)
# Compile non-Streamlit geometry/scoring/diagnostic functions from the REAL application code.
NAMES = {'calculate_2d_angle','calculate_3d_angle','_landmark_xy',
         'extract_hand_comparison','evaluate_hand_geometry','make_pose_keypoint_snapshot',
         'add_hand_temporal_notes','v4_mark_anomalies','get_wrist_score',
         'get_upper_arm_score','get_lower_arm_score','get_trunk_score','get_neck_score',
         'compute_rula_full','clean_json','safe_json_value'}
NODES = [x for x in TREE.body if isinstance(x,ast.FunctionDef) and x.name in NAMES]
TABLES=[x for x in TREE.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'TABLE_A','TABLE_B','TABLE_C'} for t in x.targets)]
ctx={'np':np,'pd':pd,
     'V41_WRIST_EXTREME_DEG':60.0, 'V41_ELBOW_EXTREME_DEG':170.0,
     'V41_ISOLATED_SPIKE_DEG':85.0,'V41_NEIGHBOR_SIMILAR_DEG':25.0}
exec(compile(ast.Module(body=TABLES+NODES,type_ignores=[]),str(ROOT/'222.py'),'exec'),ctx)


def landmark(x,y,visibility=0.96):
    return SimpleNamespace(x=float(x),y=float(y),z=0.0,visibility=visibility)


def test_hand():
    wrist=np.array([100.,100.]); elbow=np.array([100.,160.]); other=np.array([260.,100.])
    points=[landmark(100/400,100/300) for _ in range(21)]
    for i in [5,9,13,17]: points[i]=landmark(100/400,70/300)
    fake=SimpleNamespace(hand_landmarks=[points],hand_world_landmarks=[],handedness=[])
    h=ctx['extract_hand_comparison'](fake,(300,400,3),wrist,other,elbow)
    assert h['matched'],h
    assert h['angle_2d']<0.001,h
    assert abs(h['palm_forearm_ratio']-.5)<.001
    passed,why=ctx['evaluate_hand_geometry'](h,0,0.99,0.99)
    assert passed,why
    passed,why=ctx['evaluate_hand_geometry'](h,36,0.99,0.99)
    assert not passed and any('差異' in x for x in why)
    passed,why=ctx['evaluate_hand_geometry'](h,0,0.1,0.99)
    assert not passed
    h_bad=ctx['extract_hand_comparison'](fake,(300,400,3),wrist,np.array([101.,100.]),elbow)
    assert h_bad['matched'] # detection can match; QC rejects ambiguity
    passed,why=ctx['evaluate_hand_geometry'](h_bad,0,0.99,0.99)
    assert not passed
    assert not ctx['extract_hand_comparison'](None,(300,400,3),wrist,other,elbow)['matched']


def test_temporal():
    # Natural posture transition should NOT trigger an anomaly.
    xs=pd.DataFrame({'樣本編號':[1,2,3,4,5],'時間(秒)':[0,1,2,3,4],
           '右手PoseHand實驗手腕角度':[1,51,52,53,55]})
    out=ctx['add_hand_temporal_notes'](xs,'右手側')
    assert not out['Hand時序孤立尖峰（僅供研究）'].any()
    xs.loc[2,'右手PoseHand實驗手腕角度']=100
    xs.loc[1,'右手PoseHand實驗手腕角度']=5
    xs.loc[3,'右手PoseHand實驗手腕角度']=7
    out=ctx['add_hand_temporal_notes'](xs,'右手側')
    assert int(out['Hand時序孤立尖峰（僅供研究）'].sum())==1
    assert bool(out.iloc[2]['Hand時序孤立尖峰（僅供研究）'])


def test_rula_matrix():
    # Functional completeness and Grand Score range; not evidence of official worksheet correctness.
    n=0
    for ua in range(1,7):
      for la in range(1,4):
       for w in range(1,5):
        for wt in [1,2]:
         for nk in range(1,7):
          for tk in range(1,7):
           for lg in [1,2]:
            ret=ctx['compute_rula_full'](ua,la,w,wt,nk,tk,lg,0,0)
            assert 1 <= ret[-2] <= 7
            assert ret[-1] in ('AL1','AL2','AL3','AL4')
            n+=1
    assert n==10368,n


def test_prior_case():
    # Recheck previous v6 CSV. Anomalies should stay conservative, including #9 and #25.
    path=Path('/mnt/data/RULA_V4_Review_Report.zip')
    if not path.is_file(): return
    with zipfile.ZipFile(path) as z:
        df=pd.read_csv(io.BytesIO(z.read('RULA_Report.csv')))
    chk=ctx['v4_mark_anomalies'](df,'右手側')
    ids=set(chk.loc[chk['需要複核'],'樣本編號'].astype(int))
    assert 9 in ids and 25 in ids,ids
    assert 1 not in ids,ids
    assert len(ids)<10,ids


def test_serialize():
    payload={'angle':float('nan'),'list':[float('inf'),np.float64(2.3),np.int64(3)]}
    raw=json.dumps(ctx['clean_json'](payload),ensure_ascii=False,allow_nan=False)
    obj=json.loads(raw)
    assert obj['angle'] is None and obj['list'][0] is None and obj['list'][2]==3


def test_structure():
    assert SOURCE.count('model_complexity=POSE_MODEL_COMPLEXITY')==2
    for x in ['"🎬 影片與基本設定"','"🖐️ 手腕比較"','"📦 下載"',
              'Keypoint_Coordinates.jsonl','Raw_Frames','Hand品質原因','Pose2D與Hand2D差異']:
        assert x in SOURCE,x


if __name__=='__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_') and callable(f):
            f(); print('PASS',name)
    print('All offline tests passed (no physical-video angle accuracy claim).')
