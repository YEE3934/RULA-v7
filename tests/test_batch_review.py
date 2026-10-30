"""End-to-end scoring semantics of photo batch reviews; runs without Streamlit/MediaPipe."""
from pathlib import Path
import ast, io, zipfile, json
import pandas as pd
import numpy as np
from collections import defaultdict
p=Path(__file__).resolve().parents[1]/'222.py'
t=ast.parse(p.read_text(encoding='utf-8'))
functions={n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
vals={}
for n in t.body:
 if isinstance(n,ast.Assign):
  for target in n.targets:
   if isinstance(target,ast.Name) and target.id in ('TABLE_A','TABLE_B','TABLE_C','V113_KEEP','V113_MANUAL','V113_EXCLUDE','V113_PENDING','BATCH_BOOL_FIELDS','BATCH_NUMBER_FIELDS'):
    vals[target.id]=ast.literal_eval(n.value)
needed=['get_upper_arm_score','get_lower_arm_score','get_wrist_score','get_neck_score','get_trunk_score','compute_rula_full',
 'shared_neck_rula_with_score','shared_neck_apply_to_side','shared_neck_decide_pair','shared_neck_reviewed_score','shared_neck_case_table',
 'v112_is_informational_notice','v113_key','v113_dataframe_for_side','v113_record_status','v113_build_reviewed_data','v113_finalize_report_zip',
 'user_report_side','user_report_reviewed','batch_sides','batch_source_frames','batch_base_profile','batch_update_profiles',
 'batch_get_profile','batch_recalculate_side','batch_effective_frames','batch_audit_table','batch_all_photo_ids']
assert all(x in functions for x in needed)
ns={'np':np,'pd':pd,'io':io,'zipfile':zipfile,'BATCH_FIELDS':{x:y for x,y in vals['BATCH_BOOL_FIELDS']+[(x,y) for x,y,_,_ in vals['BATCH_NUMBER_FIELDS']]},**vals}
exec(compile(ast.Module(body=[functions[name] for name in needed],type_ignores=[]),str(p),'exec'),ns)
R='右手側';L='左手側'
def make(side,sid,neck_angle=8,trunk=20,upper=34,force=1):
 pre='右手' if side==R else '左手'
 r={'樣本編號':sid,'時間(秒)':float(sid),'分析經過時間(秒)':float(sid), '評估側':side,
 '頸部角度':neck_angle,'軀幹角度':trunk,'品質狀態':'通過',
 '肌肉使用加分':0,'負重施力加分':force,'腿部分數':1,
 '頸部側彎補正':'否', '軀幹側彎補正':'否','軀幹扭轉補正':'否',
 '頸部扭轉補正':'否','軀幹向後伸展':'否', '頸部後仰／伸展':'否',
 pre+'上臂角度':upper,pre+'前臂角度':95.,pre+'手腕角度':6.,
 pre+'上臂外展補正':'否',pre+'橫越中線補正':'否',pre+'前臂向外側工作補正':'否',
 pre+'肩部抬高補正':'否',pre+'上臂支撐／倚靠補正':'否',pre+'手腕偏移補正':'否',
 pre+'上臂向後伸展補正':'否',pre+'手腕扭轉分數':1}
 x=ns['batch_base_profile'](r,side)
 return ns['batch_recalculate_side'](r,side,x)
# 5 photos R and L each, unchanged source must be untouched even after edits.
raw={R:pd.DataFrame([make(R,i,neck_angle=5 if i<3 else 25) for i in range(1,6)]),
     L:pd.DataFrame([make(L,i,neck_angle=5 if i<3 else 25) for i in range(1,6)])}
copy={side:df.copy(deep=True) for side,df in raw.items()}
empty,untrusted,cases=ns['batch_effective_frames'](raw,{},set())
assert len(untrusted)==10 and all(v=='採用分析前設定（尚未額外人工修改）' for v in untrusted.values())
reviewed,combined=ns['v113_build_reviewed_data'](empty,[],{}, {},batch_status=untrusted)
assert reviewed['複核後Grand Score'].notna().all(),reviewed
assert combined['完整雙側最高分'].notna().all()
# confirm 10 pictures in single call, no change => production table scores conserved.
keys=[(side,sid) for side in (R,L) for sid in range(1,6)]
profiles,confirmed=ns['batch_update_profiles']({},set(),keys,{'trunk_twisted':'keep','force':'keep'})
eff,status,cases=ns['batch_effective_frames'](raw,profiles,confirmed)
assert len(status)==10 and all(v=='補正已人工確認' for v in status.values())
assert all(eff[R].loc[i-1,'右手 Grand Score']==raw[R].loc[i-1,'右手 Grand Score'] for i in range(1,6))
# Selected R picture 3: twisting on only this picture; L unaffected.
profiles,confirmed=ns['batch_update_profiles'](profiles,confirmed,[(R,3)],{'trunk_twisted':True})
eff,status,cases=ns['batch_effective_frames'](raw,profiles,confirmed)
assert eff[R].loc[2,'軀幹分數']==raw[R].loc[2,'軀幹分數']+1
assert eff[R].loc[1,'軀幹分數']==raw[R].loc[1,'軀幹分數']
assert eff[L].loc[2,'軀幹分數']==raw[L].loc[2,'軀幹分數']
# Force dropped for R frames 4,5 only, original raw kept.
profiles,confirmed=ns['batch_update_profiles'](profiles,confirmed,[(R,4),(R,5)],{'force':0})
eff,status,cases=ns['batch_effective_frames'](raw,profiles,confirmed)
assert eff[R].loc[3,'負重施力加分']==0 and eff[R].loc[4,'負重施力加分']==0
assert eff[R].loc[1,'負重施力加分']==1 and eff[L].loc[3,'負重施力加分']==1
pd.testing.assert_frame_equal(raw[R],copy[R]);pd.testing.assert_frame_equal(raw[L],copy[L])
# explicitly unknown cannot count as trusted, even if previous confirmed.
profiles,confirmed=ns['batch_update_profiles'](profiles,confirmed,[(R,2)],{'across':None})
eff,status,cases=ns['batch_effective_frames'](raw,profiles,confirmed)
assert status[(R,2)].startswith('無法確認')
reviewed,combined=ns['v113_build_reviewed_data'](eff,[],{}, {},neck_cases=cases,batch_status=status)
x=reviewed[(reviewed['樣本編號']==2)&(reviewed['評估側']==R)].iloc[0]
assert not x['是否納入可信統計'] and pd.isna(x['複核後Grand Score'])
assert pd.isna(combined[combined['樣本編號']==2]['完整雙側最高分'].iloc[0])
# unknowable should still be marked pending if someone clicks V113_KEEP.
rev,_=ns['v113_build_reviewed_data'](eff,[],{'2_'+R:vals['V113_KEEP']},{},neck_cases=cases,batch_status=status)
assert not rev[(rev['樣本編號']==2)&(rev['評估側']==R)]['是否納入可信統計'].iloc[0]
# reset selected only reverts these photos, not the rest.
profiles,confirmed=ns['batch_update_profiles'](profiles,confirmed,[(R,2)],{},'reset')
assert (R,2) not in confirmed and (R,3) in confirmed and (R,2) not in profiles
# Front absent but user confirms a flag manually for some photos, no global replacement.
profiles,confirmed=ns['batch_update_profiles'](profiles,confirmed,[(L,2)],{'abducted':True,'neck_side':True})
eff,status,cases=ns['batch_effective_frames'](raw,profiles,confirmed)
assert eff[L].loc[1,'左手上臂分數']==raw[L].loc[1,'左手上臂分數']+1
assert eff[L].loc[1,'頸部側彎補正'].startswith('是')
assert eff[L].loc[2,'左手上臂分數']==raw[L].loc[2,'左手上臂分數']
# Explicitly original score is available even when scores changed by manual override.
audit=ns['batch_audit_table'](raw,profiles,confirmed)
assert len(audit)==10 and '已確認' in audit and '人工修改欄位' in audit
orig=io.BytesIO()
with zipfile.ZipFile(orig,'w') as z:
 z.writestr('評估結果/初始側分數.csv',b'original')
 z.writestr('Photos/001.jpg',b'jpeg')
final=ns['v113_finalize_report_zip'](orig.getvalue(),eff,[],{}, {},neck_cases=cases,
                                   batch_status=status,batch_audit=audit)
with zipfile.ZipFile(io.BytesIO(final)) as z:
 names=z.namelist()
 assert z.testzip() is None and len(names)==len(set(names))
 assert '原始分析/初始側分數.csv' in names and z.read('原始分析/初始側分數.csv')==b'original'
 assert '複核後結果/逐張人工補正確認.csv' in names
 assert '複核後結果/照片補正後部位評分.csv' in names
 assert '複核後結果/逐側結果.csv' in names
 assert 'Photos/001.jpg' in names
 assert '研究驗證/人工評分比對空表.csv' in names
 comparison=pd.read_csv(io.BytesIO(z.read('研究驗證/人工評分比對空表.csv')))
 assert len(comparison)==10 and comparison['人工Grand Score'].isna().all()
 assert len(comparison['評估側'].unique())==2
 print('ZIP correct:',len(names),'files')
print('PASS batch review: non-destructive, confirmed/unknown, multi-select, stage load, single exception, dual sync, ZIP.')
