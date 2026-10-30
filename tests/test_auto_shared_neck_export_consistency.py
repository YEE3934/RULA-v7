"""Independent regression: automatically selected neck score must be consistent in ALL reports."""
import runpy, io,zipfile
from pathlib import Path
import pandas as pd
src=runpy.run_path(str(Path(__file__).resolve().parent/'test_shared_neck.py'))
ns=src['ns']; make=src['row'];R=src['R'];L=src['L']
# Find a conflicting 1-vs-3 case that leaves the combined AL unchanged.
for ua in range(1,7):
 for w in range(1,5):
  for tk in range(1,7):
   a=make(R,sid=27,neck=1,ua=ua,w=w,tk=tk);b=make(L,sid=27,neck=3,ua=ua,w=w,tk=tk)
   c=ns['shared_neck_decide_pair'](a,b,27,2.0)
   if c['狀態']=='自動保守共用' and not c['需要人工確認']:break
  else:continue
  break
 else:continue
 break
else:raise AssertionError('no example')
# Perturb the side row after the common-neck decision, simulating an older/out-of-sync CSV.
a['頸部分數']=1
a['右手 Grand Score']=ns['shared_neck_rula_with_score'](a,R,1)[4]
base_zip=io.BytesIO()
with zipfile.ZipFile(base_zip,'w') as f:f.writestr('照片/sample.jpg',b'photo')
result=ns['v113_finalize_report_zip'](base_zip.getvalue(),{R:pd.DataFrame([a]),L:pd.DataFrame([b])},[],{}, {},neck_cases={27:c})
with zipfile.ZipFile(io.BytesIO(result)) as f:
 side=pd.read_csv(io.BytesIO(f.read('複核後結果/逐側結果.csv')))
 details=pd.read_csv(io.BytesIO(f.read('複核後結果/照片補正後部位評分.csv')))
 study=pd.read_csv(io.BytesIO(f.read('研究驗證/人工評分比對空表.csv')))
 assert f.read('照片/sample.jpg')==b'photo'
 for label in (R,L):
  s=side[side['評估側']==label].iloc[0]
  d=details[details['評估側']==label].iloc[0]
  v=study[study['評估側']==label].iloc[0]
  expected=ns['shared_neck_rula_with_score'](a if label==R else b,label,3)[4]
  assert int(s['採用頸部分數'])==int(d['頸部分數'])==int(v['系統最終採用頸部分數'])==3
  assert int(s['複核後總分'])==int(d['設定後系統總分'])==int(v['系統人工補正後總分'])==expected
print('PASS automatic shared neck: recomputation is authoritative, even if prior row values differ')
