"""Independent table-by-table regression against source-PDF fixed fixture, not self-derived expected scores.
Source: 蕭世平 (2016), RULA 人因評估影像之資訊管理系統研究, pages 29-30 of uploaded PDF, tables 3-8,3-9,3-10.
"""
import json,ast
from pathlib import Path
here=Path(__file__).resolve().parent
reference=json.loads((here/'references/2016_rula_reference_table_values.json').read_text())
code=ast.parse((here.parent/'222.py').read_text())
def literal(k):
    return ast.literal_eval(next(n.value for n in code.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==k for t in n.targets)))
a=literal('TABLE_A');b=literal('TABLE_B');c=literal('TABLE_C')
ref_a=reference['RULA_TABLE_A_ROWS'];ref_b=reference['RULA_TABLE_B_ROWS'];ref_c=reference['RULA_TABLE_C_ROWS']
assert len(ref_a)==18 and all(len(r)==8 for r in ref_a)
assert len(ref_b)==6 and all(len(r)==12 for r in ref_b)
assert len(ref_c)==9 and all(len(r)==9 for r in ref_c)
for ua in range(1,7):
    for la in range(1,4):
        for w in range(1,5):
            for wt in (1,2):
                expected=ref_a[(ua-1)*3+la-1][(w-1)*2+wt-1]
                assert a[ua][la][w][wt]==expected,(ua,la,w,wt,expected,a[ua][la][w][wt])
for nk in range(1,7):
    for tk in range(1,7):
        for lg in (1,2):
            expected=ref_b[nk-1][(tk-1)*2+lg-1]
            assert b[nk][tk][lg]==expected,(nk,tk,lg,expected,b[nk][tk][lg])
for i in range(9):
    for j in range(9):
        assert c[i][j]==ref_c[i][j],(i,j)
fn=next(n for n in code.body if isinstance(n,ast.FunctionDef) and n.name=='compute_rula_full')
ns=dict(TABLE_A=a,TABLE_B=b,TABLE_C=c)
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<rula-scoring>','exec'),ns)
f=ns['compute_rula_full']
assert f(6,3,4,2,6,6,2,0,0)[:2]==(9,9)
assert f(6,3,4,2,6,6,2,1,3)[4]==7  # high scores saturate Table C
assert f(1,1,1,1,1,1,1,0,0)[4]==1
assert f(1,1,1,1,1,1,1,0,1)[4]==2
print('PASS independent RULA A 144 / B 72 / C 81 cells, Table C saturation and representative scores')
