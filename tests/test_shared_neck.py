"""Shared neck score & AL-sensitive confirmation regression. No Streamlit / MediaPipe needed."""
from pathlib import Path
import ast,io,zipfile
import pandas as pd,numpy as np
p=Path(__file__).resolve().parents[1]/'222.py'
tree=ast.parse(p.read_text(encoding='utf-8'))
functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
vars={}
for n in tree.body:
    if isinstance(n,ast.Assign):
        for target in n.targets:
            if isinstance(target,ast.Name) and target.id in ('TABLE_A','TABLE_B','TABLE_C','V113_KEEP','V113_MANUAL','V113_EXCLUDE','V113_PENDING'):
                vars[target.id]=ast.literal_eval(n.value)
needed=['compute_rula_full','shared_neck_rula_with_score','shared_neck_apply_to_side','shared_neck_decide_pair',
        'shared_neck_reviewed_score','shared_neck_case_table','v112_is_informational_notice','v113_key',
        'v113_dataframe_for_side','v113_record_status','v113_build_reviewed_data',
        'user_report_side','user_report_dual','user_report_reviewed','v113_finalize_report_zip']
ns=dict(pd=pd,np=np,io=io,zipfile=zipfile,**vars)
exec(compile(ast.Module(body=[functions[n] for n in needed],type_ignores=[]),str(p),'exec'),ns)

def row(side,sid=1,neck=1,ua=1,la=1,w=1,wt=1,tk=1,lg=1,m=0,f=0):
    pre='右手' if side=='右手側' else '左手'
    d={'樣本編號':sid,'評估側':side,'頸部分數':neck,'頸部角度':8.0 if neck==1 else 51.0,
       '品質狀態':'通過','軀幹分數':tk,'腿部分數':lg,'肌肉使用加分':m,'負重施力加分':f,
       pre+'上臂分數':ua,pre+'前臂分數':la,pre+'手腕分數':w,pre+'手腕扭轉分數':wt,
       '分析經過時間(秒)':0.0,
       pre+'上臂角度':30.0,pre+'前臂角度':65.0,pre+'手腕角度':3.9}
    a,b,c,dscore,grand,al=ns['compute_rula_full'](ua,la,w,wt,neck,tk,lg,m,f)
    d.update({pre+' Score A':a,'頸軀腿 Score B':b,pre+' Score C':c,'頸軀腿 Score D':dscore,
              pre+' Grand Score':grand,pre+' AL':al})
    return d
R='右手側';L='左手側'
# Case A: same score automatically shared, score remains consistent
r=row(R,neck=1);l=row(L,neck=1)
c=ns['shared_neck_decide_pair'](r,l,1,0)
assert c['狀態']=='一致，自動採用' and not c['需要人工確認'] and r['頸部分數']==l['頸部分數']==1
# Case B: disagreement that does NOT change AL -> conservative common score, no manual
found=None
for ua in range(1,7):
 for w in range(1,5):
  for tk in range(1,7):
   dr=row(R,neck=1,ua=ua,w=w,tk=tk);dl=row(L,neck=3,ua=ua,w=w,tk=tk)
   ca=ns['shared_neck_decide_pair'](dr,dl,2,0)
   if not ca['需要人工確認'] and ca['狀態']=='自動保守共用':found=(dr,dl,ca);break
  if found:break
 if found:break
assert found, 'Expected same-AL conflict scenario in RULA table'
dr,dl,ca=found
assert dr['頸部分數']==dl['頸部分數']==3
assert dr['右手 Grand Score']==ns['shared_neck_rula_with_score'](dr,R,3)[4]
# Case C: disagreement that DOES change combined AL -> both provisional pending
r=row(R,neck=1,ua=1,w=1,tk=1);l=row(L,neck=4,ua=1,w=1,tk=1)
case=ns['shared_neck_decide_pair'](r,l,3,0)
assert case['需要人工確認'],case
assert set(v['雙側AL'] for v in case['各候選情境'].values())=={'AL1','AL2'}
assert r['頸部分數']==l['頸部分數']==4 and r['頸部共用狀態']=='待頸部人工確認'
sources={R:pd.DataFrame([r]), L:pd.DataFrame([l])}
build=ns['v113_build_reviewed_data']
reviewed,combined=build(sources,[],{}, {},neck_cases={3:case},neck_decisions={})
# Careful synthetic id differs from row id: rows sample id 1 until changed below
r['樣本編號']=l['樣本編號']=3
sources={R:pd.DataFrame([r]), L:pd.DataFrame([l])}
reviewed,combined=build(sources,[],{}, {},neck_cases={3:case},neck_decisions={})
assert reviewed['複核後Grand Score'].isna().all()
assert set(reviewed['結果狀態'])=={'待頸部局部人工確認'}
assert pd.isna(combined.iloc[0]['完整雙側最高分'])
assert ns['user_report_side'](sources[R],R).iloc[0]['RULA總分'] is None or pd.isna(ns['user_report_side'](sources[R],R).iloc[0]['RULA總分'])
assert pd.isna(ns['user_report_dual'](pd.DataFrame([{
 '樣本編號':3,'頸部共用狀態':'待頸部人工確認','頸部分數':4,
 '右手 Grand Score':4,'左手 Grand Score':4,'最終最高分':4,'最終 AL':'AL2'
}])).iloc[0]['雙側最高分'])
# Manual neck decision adopted by both sides, other scores untouched, clean reviewed report
reviewed,combined=build(sources,[],{}, {},neck_cases={3:case},neck_decisions={3:1})
assert list(reviewed['複核後Grand Score'])==[1,1],reviewed
assert list(reviewed['採用頸部分數'])==[1,1]
assert combined.iloc[0]['完整雙側最高分']==1 and combined.iloc[0]['完整雙側AL']=='AL1'
# If other manual side score changed, new neck should recompute with that other manual component
manual={(R,3):{'人工上臂分數':3,'人工Grand Score':5,'人工AL':'AL3'}}
reviewed2,combined2=build(sources,[],{},manual,neck_cases={3:case},neck_decisions={3:1})
rr=reviewed2[reviewed2['評估側']==R].iloc[0]
assert rr['複核後Grand Score']==ns['shared_neck_rula_with_score'](r,R,1,manual[(R,3)])[4]
# If user excludes one side, do not report the remaining side as full dual assessment
reviewed3,combined3=build(sources,[],{ns['v113_key'](3,R):ns['V113_EXCLUDE']},{},neck_cases={3:case},neck_decisions={3:1})
assert pd.isna(combined3.iloc[0]['完整雙側最高分'])
# ZIP finalization contains reviewed side/dual, neck confirmation and technical cross-view scenarios
bio=io.BytesIO()
with zipfile.ZipFile(bio,'w') as z:
 z.writestr('評估結果/雙側總覽.csv','base')
 z.writestr('Raw_Frames/Left/test.jpg',b'test')
output=ns['v113_finalize_report_zip'](bio.getvalue(),sources,[],{}, {},neck_cases={3:case},neck_decisions={3:1})
with zipfile.ZipFile(io.BytesIO(output)) as z:
 assert z.testzip() is None
 assert '複核後結果/頸部分級確認.csv' in z.namelist()
 assert '技術診斷/頸部分級情境比對.csv' in z.namelist()
 assert 'Raw_Frames/Left/test.jpg' in z.namelist()
 actual=pd.read_csv(io.BytesIO(z.read('複核後結果/雙側結果.csv')))
 assert actual.loc[0,'雙側最高分']==1
print('PASS shared neck: aligned, AL-insensitive, AL-conflict pending, manual both, overrides, exclusion, reports, ZIP')
