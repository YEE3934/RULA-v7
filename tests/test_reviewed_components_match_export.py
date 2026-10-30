"""Cross-file contract: after shared-neck manual decision, every CSV has same neck/Grand Score.
This catches the actual 2221 sample 23 export divergence.
"""
import runpy,io,zipfile
from pathlib import Path
import pandas as pd
base=Path(__file__).resolve().parent
ns=runpy.run_path(str(base/'test_shared_neck.py'))
env=ns['ns'];make=ns['row'];R=ns['R'];L=ns['L']
r=make(R,sid=23,neck=3,ua=3,la=2,w=2,wt=1,tk=4)
l=make(L,sid=23,neck=3,ua=3,la=2,w=2,wt=1,tk=4)
case=env['shared_neck_decide_pair'](r,l,23,1.5)
bio=io.BytesIO()
with zipfile.ZipFile(bio,'w') as z:z.writestr('Raw_Frames/Right/test.jpg',b'original_photo')
output=env['v113_finalize_report_zip'](bio.getvalue(),{R:pd.DataFrame([r]),L:pd.DataFrame([l])},[],{},{},neck_cases={23:case},neck_decisions={23:1})
with zipfile.ZipFile(io.BytesIO(output)) as z:
    side=pd.read_csv(io.BytesIO(z.read('複核後結果/逐側結果.csv')))
    components=pd.read_csv(io.BytesIO(z.read('複核後結果/照片補正後部位評分.csv')))
    comparison=pd.read_csv(io.BytesIO(z.read('研究驗證/人工評分比對空表.csv')))
    assert z.read('Raw_Frames/Right/test.jpg')==b'original_photo'
    assert '技術診斷/程式與計分版本.json' in z.namelist()
    for s in (R,L):
        r1=side.query('評估側==@s').iloc[0]
        r2=components.query('評估側==@s').iloc[0]
        r3=comparison.query('評估側==@s').iloc[0]
        assert r1['採用頸部分數']==r2['頸部分數']==r3['系統最終採用頸部分數']==1,(r1,r2,r3)
        assert r1['複核後總分']==r2['設定後系統總分']==r3['系統人工補正後總分']==5,(r1,r2,r3)
        assert r1['複核後行動等級']==r2['設定後AL']==r3['系統人工補正後AL']=='AL3',(r1,r2,r3)
print('PASS sample23 style: all finalized CSVs agree after manually shared neck score; photos preserved')
