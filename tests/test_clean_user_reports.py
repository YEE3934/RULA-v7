"""Normal report uses one adopted neck and wrist value; developer variants stay outside normal reports."""
import ast,io,zipfile,collections
from pathlib import Path
import pandas as pd
p=Path(__file__).resolve().parents[1]/'222.py'
src=p.read_text(encoding='utf-8')
tree=ast.parse(src)
funcs={x.name:x for x in tree.body if isinstance(x,ast.FunctionDef)}
wanted=['user_report_side','user_report_dual','user_report_reviewed','user_report_neck','user_report_wrist',
        'v112_is_informational_notice','v113_key','v113_dataframe_for_side','v113_get_sample_row',
        'v113_record_status','v113_build_reviewed_data','v113_finalize_report_zip']
ns=dict(pd=pd,io=io,zipfile=zipfile)
for x in tree.body:
    if isinstance(x,ast.Assign):
        for tgt in x.targets:
            if isinstance(tgt,ast.Name) and tgt.id in ('V113_KEEP','V113_MANUAL','V113_EXCLUDE','V113_PENDING'):
                ns[tgt.id]=ast.literal_eval(x.value)
exec(compile(ast.Module(body=[funcs[n] for n in wanted],type_ignores=[]),str(p),'exec'),ns)
side=pd.DataFrame([{
    '樣本編號':1,'時間(秒)':-18.5,'分析經過時間(秒)':0.0,'右手上臂角度':32,
    '右手前臂角度':67,'右手手腕角度':3.9,'頸部角度':8.1,'頸部分數':1,'軀幹角度':7.0,
    '右手 Grand Score':4,'右手 AL':'AL2','品質狀態':'通過',
    '右手Pose舊版手腕角度':30.1,'右手手腕角度_3D':74.0,
    '頸部_v8_候選角度':38.0,'頸部_v114_同側角度':8.1,'頸部_v7_原角度':46.1,
    '右手手腕分數':1}])
clean=ns['user_report_side'](side,'右手側')
assert len(clean)==1
assert clean.iloc[0]['手腕角度(°)']==3.9
assert clean.iloc[0]['頸部分數']==1
assert clean.iloc[0]['時間(秒)']==0.0
assert not any('v8' in c.lower() or '3D' in c or '候選' in c for c in clean.columns)
assert len([c for c in clean if '手腕角度' in c])==1
assert clean.iloc[0]['頸部角度(°)']==8.1
assert '上臂分數' not in clean.columns
with_scores=ns['user_report_side'](side,'右手側',with_scores=True)
assert with_scores.iloc[0]['手腕分數']==1
both=pd.DataFrame([{
    '樣本編號':1,'分析經過時間(秒)':0.0,'右手腕角度':3.9,'左手腕角度':7.4,
    '右側頸部實際角度':8.1,'左側頸部實際角度':6.2, '頸部分數':1, '頸部共用狀態':'一致，自動採用',
    '右手 Grand Score':4,'左手 Grand Score':5,'最終最高分':5,'最終 AL':'AL3',
    '最危害側':'左手側','頸部_v8_左側候選':70.0,
}])
c=ns['user_report_dual'](both)
assert len([k for k in c if '手腕角度' in k])==2
assert len([k for k in c if '頸部角度' in k])==2
assert c.iloc[0]['右側頸部角度(°)']==8.1
assert c.iloc[0]['左側頸部角度(°)']==6.2
assert c.iloc[0]['共同頸部分數']==1
assert not any('_v' in x or '候選' in x for x in c.columns)
# ZIP includes readable top-level and research-only technical area; no development version in main report.
orig=io.BytesIO()
with zipfile.ZipFile(orig,'w') as f:
    f.writestr('評估結果/右側角度與評分.csv',clean.to_csv(index=False).encode('utf-8-sig'))
    f.writestr('Diagnostics/V114_Neck_Same_Side_Comparison.csv',pd.DataFrame([{
        '樣本編號':1,'頸部_v7_原角度':46.1,'頸部_v114_同側角度':8.1,'頸部_v114_品質':.9}]).to_csv(index=False).encode('utf-8-sig'))
    f.writestr('Hand_Comparison/Sample_001_Pose.jpg',b'abc')
    f.writestr('Diagnostics/V11_Skeleton_Quality.csv',b'qc')
    f.writestr('Diagnostics/V11_1_Skeleton_Quality.csv',b'duplicate')
    f.writestr('Synchronization/Frame_Time_Map.csv',b'timestamps')
    f.writestr('Photos/sample.jpg',b'photo')
final=ns['v113_finalize_report_zip'](orig.getvalue(),{'右手側':side},[],{},{} )
with zipfile.ZipFile(io.BytesIO(final)) as z:
    assert z.testzip() is None
    names=z.namelist()
    assert len(names)==len(set(names))
    assert '閱讀指南.txt' in names
    assert '原始分析/右側角度與評分.csv' in names
    assert '技術診斷/頸部計算方法對照.csv' in names
    assert '照片/手部辨識對照/Sample_001_Pose.jpg' in names
    assert '複核後結果/逐側結果.csv' in names
    assert '時間同步/逐筆取樣時間.csv' in names
    assert not any('V11_' in n or 'v8' in n or 'V114_' in n for n in names)
    data=pd.read_csv(io.BytesIO(z.read('原始分析/右側角度與評分.csv')))
    assert '手腕角度(°)' in data and '頸部分數' in data and '頸部角度(°)' in data
    assert all('v8' not in c and '_v' not in c for c in data.columns)
print('PASS: visible neck angle single/dual/score CSV, zip layout, photo preservation, readable filenames')
