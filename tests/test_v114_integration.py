"""AST-extracted analyze_side_camera integration smoke test with real exported keypoints.
No MediaPipe or Streamlit runtime required; fake detector returns captured Pose landmarks.
"""
import ast,zipfile,json,math,time,logging
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd,cv2
p=Path(__file__).resolve().parents[1]/'222.py'
tree=ast.parse(p.read_text(encoding='utf-8'))
funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
names=['side_directional_posture','calculate_2d_angle','calculate_3d_angle','get_upper_arm_score','get_lower_arm_score',
       'get_wrist_score','get_neck_score','get_trunk_score','compute_rula_full',
       'v8_neck_candidate','v114_same_side_neck','v114_overlay_neck_lines','analyze_side_camera']
vals={}
for x in tree.body:
    if isinstance(x,ast.Assign):
        for t in x.targets:
            if isinstance(t,ast.Name) and t.id in ('TABLE_A','TABLE_B','TABLE_C'):
                vals[t.id]=ast.literal_eval(x.value)
ns=dict(np=np,cv2=cv2,time=time,logging=logging,**vals)
ns['V114_NECK_MODES']=("只比較新舊頸角（推薦；正式分數維持舊版）","品質達標採用同側頸角（實驗性；影響 RULA 分數）")
exec(compile(ast.Module(body=[funcs[n] for n in names],type_ignores=[]),str(p),'exec'),ns)
ns['make_pose_keypoint_snapshot']=lambda *args: {'stub':'no full MediaPipe runtime'}
ns['draw_full_biomechanical_skeleton']=lambda image,*args,**kwargs: image.copy()
ns['evaluate_hand_geometry']=lambda *a:(False,[])
ns['landmarks_meet_visibility']=lambda *a,**kw:True
ns['_visibility_values']=lambda result,indices:[float(result.pose_landmarks.landmark[i].visibility) for i in indices]
fixture=p.parent/'tests/fixtures/v114_keypoints.jsonl'
rows=[json.loads(line) for line in fixture.read_text(encoding='utf-8').splitlines() if line.strip()]

def fake_result(record):
    w,h=record['影像寬'],record['影像高']
    # Real landmarks from exported snapshot where possible. For omitted facial points,
    # use the nose; this integration checks control-flow, not new model inference.
    pts=record['Pose影像像素座標']; nose=pts.get('0',{'x_px':w*.5,'y_px':h*.3})
    landmarks=[]; worlds=[]
    for i in range(33):
        p=pts.get(str(i),nose)
        x=float(p['x_px'])/w;y=float(p['y_px'])/h
        landmarks.append(SimpleNamespace(x=x,y=y,z=.02*i,visibility=float(p.get('visibility',.99))))
        worlds.append(SimpleNamespace(x=x,y=y,z=.02*i))
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmarks),
                           pose_world_landmarks=SimpleNamespace(landmark=worlds))

ok=0; tested=[]
for record in rows:
    if record['評估側']!='左手側':continue
    side='左手側'; h,w=record['影像高'],record['影像寬']
    result=fake_result(record)
    ns['detect_pose_with_fallback']=lambda *args,**kwargs: (result,None,None)
    frame=np.zeros((h,w,3),dtype=np.uint8)
    front={key:False for key in ('is_abducted','is_across_midline','is_out_to_side','is_side_bending','is_neck_side_bending')}
    manual={'auto_selection':{},'leg_score':2,'wrist_twist_score':1}
    outputs=[]
    for mode in ns['V114_NECK_MODES']:
        manual['neck_v114_mode']=mode
        success,data,img,quality=ns['analyze_side_camera'](frame,None,side,front,0,0,manual,None)
        if not success:break
        outputs.append(data)
    if len(outputs)!=2:continue
    old,new=outputs
    assert old['頸部角度']==old['頸部_v7_原角度']
    assert old['頸部_v114_計分來源']=='舊版（比較模式）'
    assert new['頸部_v114_計分來源']=='同側耳肩髖（品質通過）'
    assert new['頸部角度']==round(new['頸部_v114_同側角度'],1)
    assert new['頸部_v114_舊版GrandScore']==old['左手 Grand Score']
    assert new['頸部_v114_同側情境GrandScore']==new['左手 Grand Score']
    assert old['左手手腕角度']==new['左手手腕角度']
    assert img.sum()>0  # v11.4 reference lines rendered (on blank black input)
    tested.append((record['樣本編號'],old['頸部角度'],new['頸部角度'],old['左手 Grand Score'],new['左手 Grand Score']))
    ok+=1
assert ok >= 8,(ok,tested)
print('v11.4 analyze_side_camera integration PASS:',ok,'real landmark samples; changed Grand Score:',sum(x[3]!=x[4] for x in tested))
print('selected sample differences:',tested[:5])
