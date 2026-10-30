"""Directional geometry, consistent default candidate scoring, review-on-demand without model runtime."""
import ast
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]/'222.py'
t=ast.parse(P.read_text())
fn=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='side_directional_posture')
ns={'np':np}
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(P),'exec'),ns)
f=ns['side_directional_posture']

def check(flip, shoulder=(100,100), hip=(100,200), elbow=(70,170), eye=(120,74)):
    # mirror everything around x=300 (left-facing case)
    def pt(p):return (300-p[0],p[1]) if flip else p
    return f(pt((135,82)),pt((95,83)),pt(eye),pt(shoulder),pt(hip),pt(shoulder),pt(elbow))

for flip in (False,True):
    neutral=check(flip,elbow=(100,170))
    assert neutral['trunk_extended'] is False,neutral
    assert neutral['upper_arm_extended'] is False,neutral
    assert neutral['neck_extended'] is None,neutral
    arm=check(flip)
    assert arm['upper_arm_extended'] is True,arm
    trunk=check(flip,shoulder=(77,100),hip=(100,200),elbow=(77,170))
    assert trunk['trunk_extended'] is True,trunk
    head=check(flip,eye=(116,47))
    assert head['neck_extended'] is True,head
    # near zero signed trunk angle should not silently trigger extension
    borderline=check(flip,shoulder=(91,100))
    assert borderline['trunk_extended'] is None,borderline
# If face occluded or head direction indeterminate, no auto overrides.
unknown=f((99,80),(95,82),(120,75),(100,100),(100,200),(100,100),(60,170))
assert all(unknown[key] is None for key in ('upper_arm_extended','trunk_extended','neck_extended'))
source=P.read_text()
assert '上臂向後伸展（側面有向角度）' in source and '實驗性）' not in source.split('AUTO_FLAGS={',1)[1].split('front_labels=',1)[0]
assert "auto_selection[flag]=v8_widget('checkbox',label,'auto_'+flag,True)" in source
assert "'採用分析前設定（尚未額外人工修改）'" in source
assert '未確認姿勢／負荷補正。請至' not in source
print('PASS directionality: left/right camera mirrored, arm/trunk/neck extension, near-threshold, occlusion; defaults enabled')
