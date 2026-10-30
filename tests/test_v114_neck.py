"""No Streamlit/MediaPipe imports needed: extract actual functions from full source AST."""
import ast
import json
import math
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import cv2

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / '222.py'
tree = ast.parse(CODE.read_text(encoding='utf-8'))
by_name = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
names = ['calculate_2d_angle','v114_same_side_neck','v114_overlay_neck_lines','v114_neck_report','get_neck_score','compute_rula_full']
ns = dict(np=np,pd=pd,cv2=cv2)
exec(compile(ast.Module(body=[by_name[n] for n in names],type_ignores=[]),str(CODE),'exec'),ns)
measure = ns['v114_same_side_neck']

def synth(ear=(102,30),shoulder=(100,70),hip=(100,190),side='左手側',ear_vis=.99):
    coords={7:ear,8:ear,11:shoulder,12:shoulder,23:hip,24:hip}
    lms=[SimpleNamespace(x=0.,y=0.,visibility=.99) for _ in range(33)]
    for ix,(x,y) in coords.items():lms[ix]=SimpleNamespace(x=x/500.,y=y/500.,visibility=ear_vis if ix in (7,8) else .99)
    return measure(lms,500,500,side),lms

neutral,_=synth()
assert neutral['valid'] and neutral['angle']<4,neutral
bent,_=synth(ear=(40,82),shoulder=(90,120),hip=(240,175))
assert bent['valid'] and np.isfinite(bent['angle']),bent
low,_=synth(ear_vis=.30)
assert not low['valid'] and '可見度' in low['note'],low
short,_=synth(ear=(101,68))
assert not short['valid'],short
big,_=synth(ear=(100,-60))
assert not big['valid'],big
blank=[SimpleNamespace(x=float('nan'),y=.5,visibility=1.) for _ in range(33)]
assert not measure(blank,500,500,'右手側')['valid']
# The requested side must be used, not the opposite ear.
m,l=synth()
l[7].visibility=.01
assert not measure(l,500,500,'左手側')['valid']
assert measure(l,500,500,'右手側')['valid']
# Overlay on real image has exactly the same dimensions and changes pixel data.
canvas=np.zeros((400,600,3),np.uint8)
m,l=synth(ear=(110,42),shoulder=(100,80),hip=(100,220))
res=ns['v114_overlay_neck_lines'](canvas,m,legacy_angle=30)
assert res.shape==canvas.shape and res.sum()>0
assert (ns['v114_overlay_neck_lines'](canvas,{'ear':None,'shoulder':None,'hip':None})==canvas).all()

# Report persists important values and includes both sides.
report=ns['v114_neck_report'](pd.DataFrame([{'樣本編號':1,'評估側':'右手側','頸部_v114_同側角度':4.8}]),pd.DataFrame([{'樣本編號':1,'評估側':'左手側','頸部_v114_同側角度':8.1}]))
assert len(report)==2 and '頸部_v114_同側角度' in report
assert ns['v114_neck_report']().empty

# Reuse exported MediaPipe pose keypoints: compare old angle with the new same-side proxy.
fixture=ROOT/'tests/fixtures/v114_keypoints.jsonl'
items=[json.loads(line) for line in fixture.read_text(encoding='utf-8').splitlines() if line.strip()]
records=[]
for record in items:
    points=record['Pose影像像素座標']; width=record['影像寬']; height=record['影像高']
    lms=[SimpleNamespace(x=0.,y=0.,visibility=0.) for _ in range(33)]
    for si,p in points.items():
        lms[int(si)]=SimpleNamespace(x=p['x_px']/width,y=p['y_px']/height,visibility=p['visibility'])
    candidate=measure(lms,width,height,record['評估側'])
    legacy=record['角度原始估計']['頸部2d代理']
    records.append({'樣本編號':record['樣本編號'],'側':record['評估側'],'舊角':legacy,
                    '新角':candidate['angle'],'有效':candidate['valid'],'品質':candidate['quality'],'原因':candidate['note'],
                    '差':abs(candidate['angle']-legacy) if np.isfinite(candidate['angle']) else np.nan})
df=pd.DataFrame(records)
assert len(df)>=10
print('Regression real records:',len(df),'valid candidate:',df['有效'].sum(),
      'median angle abs difference:',round(df['差'].median(),2),
      'failures:',df.loc[~df['有效'],'原因'].value_counts().to_dict())
print('largest differences:')
print(df.sort_values('差',ascending=False).head(8).to_string(index=False))
# Exactly 1 side condition should never read the opposite side as fallback.
assert any('同側耳' in r for r in df['原因'])
# New scoring path is wired in code and kept opt-in.
src=CODE.read_text(encoding='utf-8')
assert 'use_new_neck = bool(neck_mode == V114_NECK_MODES[1] and neck_v114["valid"])' in src
assert 'ang_neck = neck_v114["angle"] if use_new_neck else neck_legacy' in src
assert src.count('Diagnostics/V114_Neck_Same_Side_Comparison.csv')>=2
assert 'neck_v114_mode=v8_value' in src
assert 'frame_annotated = v114_overlay_neck_lines' in src
print('v11.4 neck QA: PASS')
