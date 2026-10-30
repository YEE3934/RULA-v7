"""Missing/low quality frontal frame should not force 30 photos into review."""
from pathlib import Path
import ast
p=Path(__file__).resolve().parents[1]/'222.py'
mod=ast.parse(p.read_text(encoding='utf8'))
f=next(n for n in mod.body if isinstance(n,ast.FunctionDef) and n.name=='v11_front_fallback_note')
ns={};exec(compile(ast.Module(body=[f],type_ignores=[]),str(p),'exec'),ns)
func=ns['v11_front_fallback_note']
q={'品質狀態':'通過','品質可直接採信':True,'品質是否列入複核中心':False,'品質原因':'骨架位置正常'}
data={}
res=func(q,data,False,{'is_abducted':True})
assert res['品質狀態']=='留意' and res['品質可直接採信']
assert res['品質是否列入複核中心'] is False
assert '作業設定' in res['品質原因']
q2={'品質狀態':'需人工複核','品質可直接採信':False,'品質是否列入複核中心':True,'品質原因':'手腕骨架明顯異常'}
r2=func(q2,{},False,{'is_abducted':True})
assert r2['品質狀態']=='需人工複核' and r2['品質是否列入複核中心']
q3={'品質狀態':'通過','品質可直接採信':True,'品質是否列入複核中心':False}
assert func(q3,{},True,{'is_abducted':True})['品質狀態']=='通過'
print('PASS missing front fallback does not force per-image reviews; real skeleton QC preserved')
