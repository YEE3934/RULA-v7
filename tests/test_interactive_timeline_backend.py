"""Test production timeline persistence and time-sliced scoring without importing heavy ML packages."""
import ast
from pathlib import Path
import numpy as np

source=(Path(__file__).resolve().parents[1]/'222.py').read_text(encoding='utf-8')
tree=ast.parse(source)
functions={x.name:x for x in tree.body if isinstance(x,ast.FunctionDef)}
wanted=['rula_timeline_rules','rula_timeline_at','rula_timeline_apply_interactive','rula_timeline_segments_from_values']
exec(compile(ast.Module(body=[functions[name] for name in wanted],type_ignores=[]),'<timeline>','exec'), env:={'np':np})

s=[{'title':'搬運','start':4,'end':9,'muscle':1,'force':2}, {'title':'放置','start':9,'end':13,'muscle':0,'force':1}]
values={}
loaded=env['rula_timeline_apply_interactive'](values,s,21.69)
assert loaded==s
assert values['timeline_count']==2 and values['timeline_enabled'] is True
rules=env['rula_timeline_rules'](values,21.69)
for second,pair in [(0,(0,0)),(4,(1,2)),(8.999,(1,2)),(9,(0,1)),(12.999,(0,1)),(13,(0,1)),(13.1,(0,0)),(21,(0,0))]:
    assert env['rula_timeline_at'](second,rules,0,0)[:2]==pair,second
old=values.copy()
for invalid in [
    [{'title':'overlap','start':8,'end':12,'muscle':0,'force':0},s[0]],
    [{'title':'outside','start':12,'end':33,'muscle':0,'force':0}],
    [{'title':'invalid','start':3,'end':3.03,'muscle':0,'force':0}],
    [{'title':'score','start':3,'end':5,'muscle':0,'force':9}],
    s * 13,
]:
    try: env['rula_timeline_apply_interactive'](values,invalid,21.69)
    except (ValueError,TypeError,KeyError,OverflowError): pass
    else: raise AssertionError('accepted invalid segments: '+repr(invalid))
    assert values==old,'bad segment should not mutate last known-good values'
assert env['rula_timeline_apply_interactive'](values,[],21.69)==[]
assert values['timeline_enabled'] is False and values['timeline_count']==0
assert env['rula_timeline_rules'](values,21.69)==[]
# max number of non-overlapping intervals supported
segments=[{'title':f'區段{i+1}','start':i*.5,'end':i*.5+.25,'muscle':0,'force':0} for i in range(24)]
assert len(env['rula_timeline_apply_interactive'](values,segments,21.69))==24
print('PASS timeline backend: 24 intervals, empty reset, half-open boundaries, rejecting overlap/out-of-range')
