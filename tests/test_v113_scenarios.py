import ast, pandas as pd, io, zipfile, math
import pathlib
source=(pathlib.Path(__file__).resolve().parents[1]/'222.py').read_text(encoding='utf-8')
tree=ast.parse(source)
funcs={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
selected=['v112_is_informational_notice','v113_key','v113_dataframe_for_side','v113_get_sample_row','v113_record_status','v113_build_reviewed_data','v113_make_manual_score','v113_finalize_report_zip','user_report_reviewed']
ns={'pd':pd,'io':io,'zipfile':zipfile}
ns.update({k:next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==k for x in n.targets)) for k in ['V113_KEEP','V113_MANUAL','V113_EXCLUDE','V113_PENDING']})
# compile only selected def, not whole streamlit top-level UI
code=compile(ast.Module(body=[funcs[name] for name in selected],type_ignores=[]),'<extracted>','exec')
exec(code,ns)
R='右手側'; L='左手側'
right=pd.DataFrame([{'樣本編號':1,'時間(秒)':0,'右手 Grand Score':5,'右手 AL':'AL3','品質狀態':'留意'},
                    {'樣本編號':2,'時間(秒)':1,'右手 Grand Score':4,'右手 AL':'AL2','品質狀態':'通過'},
                    {'樣本編號':3,'時間(秒)':2,'右手 Grand Score':6,'右手 AL':'AL3','品質狀態':'通過'}])
left=pd.DataFrame([{'樣本編號':1,'時間(秒)':0,'左手 Grand Score':6,'左手 AL':'AL3','品質狀態':'通過'},
                   {'樣本編號':2,'時間(秒)':1,'左手 Grand Score':7,'左手 AL':'AL4','品質狀態':'需人工複核'},
                   {'樣本編號':3,'時間(秒)':2,'左手 Grand Score':5,'左手 AL':'AL3','品質狀態':'通過'}])
side_frames={R:right,L:left}
items=[{'樣本編號':1,'時間(秒)':0,'評估側':R,'紀錄類型':'骨架品質提醒（仍可採信）','拒絕原因':'腕投影接近'},
       {'樣本編號':2,'時間(秒)':1,'評估側':L,'紀錄類型':'骨架品質待複核（有候選 RULA 分數）','拒絕原因':'遮擋'},
       {'樣本編號':4,'時間(秒)':3,'評估側':L,'紀錄類型':'辨識失敗','拒絕原因':'缺點'}]
build=ns['v113_build_reviewed_data']; keep=ns['V113_KEEP'];manual=ns['V113_MANUAL']; exclude=ns['V113_EXCLUDE']
# Default only notice accepted, pending left excluded
s,c=build(side_frames,items,{},{}); assert s.query('樣本編號==1 and 評估側==@R').iloc[0]['是否納入可信統計']
assert not s.query('樣本編號==2 and 評估側==@L').iloc[0]['是否納入可信統計']
assert c.query('樣本編號==2').iloc[0]['雙側結果狀態'].startswith('僅單側有效')
assert pd.isna(c.query('樣本編號==2').iloc[0]['完整雙側最高分'])
assert pd.isna(c.query('樣本編號==4').iloc[0]['完整雙側最高分'])
# Notice manually rejected; no change to other side
s,c=build(side_frames,items,{'1_'+R:exclude},{})
assert not s.query('樣本編號==1 and 評估側==@R').iloc[0]['是否納入可信統計']
assert s.query('樣本編號==1 and 評估側==@L').iloc[0]['是否納入可信統計']
assert c.query('樣本編號==3').iloc[0]['完整雙側最高分']==6
# Manual pending then finished
s,c=build(side_frames,items,{'2_'+L:manual},{})
assert s.query('樣本編號==2 and 評估側==@L').iloc[0]['結果狀態']=='等待人工重新評分'
overrides={(L,2):{'人工Grand Score':3,'人工AL':'AL2'},(L,4):{'人工Grand Score':7,'人工AL':'AL4'}}
s,c=build(side_frames,items,{'2_'+L:manual,'4_'+L:manual},overrides)
assert c.query('樣本編號==2').iloc[0]['完整雙側最高分']==4
assert c.query('樣本編號==2').iloc[0]['最高風險側']==R
assert c.query('樣本編號==4').iloc[0]['雙側結果狀態'].startswith('僅單側有效')
# explicit accept warning
s,c=build(side_frames,items,{'2_'+L:keep},{})
assert c.query('樣本編號==2').iloc[0]['完整雙側最高分']==7
# zip supports nonduplicate correct paths and keeps original
buff=io.BytesIO()
with zipfile.ZipFile(buff,'w') as z:
 z.writestr('RULA_original.csv','original unchanged')
 z.writestr('Diagnostics/Rejected_Manual_Reviews.csv','OLD')
final=ns['v113_finalize_report_zip'](buff.getvalue(),side_frames,items,{'1_'+R:exclude,'2_'+L:manual},overrides)
with zipfile.ZipFile(io.BytesIO(final)) as z:
 assert z.testzip() is None
 names=z.namelist(); assert len(names)==len(set(names))
 assert z.read('RULA_original.csv')==b'original unchanged'
 assert '複核後結果/使用者複核紀錄.csv' in names
 combo=pd.read_csv(io.BytesIO(z.read('複核後結果/雙側結果.csv')))
 assert pd.isna(combo.query('樣本編號==1').iloc[0]['雙側最高分'])
 assert combo.query('樣本編號==2').iloc[0]['雙側最高分']==4
assert 'RULA_VERIFIED_TABLES_FRONT_REVIEW_2026-10-30' in source
assert source.index('def v113_build_reviewed_data') < source.index('if st.session_state.v9_screen == \'results\':')
print('Passed: 12 targeted review/report/ZIP scenarios')
