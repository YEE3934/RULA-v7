"""Frontal pose checks: yaw gate, shoulder tilt gate, clear view, no auto neck-lateral +1.
These are engineering integration tests; human labels still required for true accuracy.
"""
from pathlib import Path
import ast,math
from types import SimpleNamespace
import numpy as np,cv2
p=Path(__file__).resolve().parents[1]/'222.py'
tree=ast.parse(p.read_text())
funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
fn=funcs['analyze_front_camera']
ns={'np':np,'cv2':cv2,'math':math,
 'RULA_OUTSIDE_SHOULDER_YES':.30,'RULA_OUTSIDE_HIP_YES':.28,
 'RULA_OUTSIDE_FOREARM_YES':.16,'RULA_OUTSIDE_DIRECTION_YES':.26}
exec(compile(ast.Module(body=[funcs['rula_outside_work_geometry'],fn],type_ignores=[]),'<front>','exec'),ns)
xy={7:(.43,.19),8:(.57,.19),11:(.35,.30),12:(.65,.30),13:(.25,.43),14:(.83,.43),15:(.2,.47),16:(.91,.47),23:(.38,.72),24:(.62,.72)}

def pose(yaw=0,raise_shoulder=False,cross=False):
    points=[];world=[]
    coords=dict(xy)
    if raise_shoulder:coords[12]=(.65,.52)
    if cross:coords[16]=(.38,.48)
    for i in range(33):
        x,y=coords.get(i,(.5,.5))
        points.append(SimpleNamespace(x=x,y=y,z=0,visibility=.99))
        world.append(SimpleNamespace(x=x,y=y,z=(yaw if i==12 else 0),visibility=.99))
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=points),pose_world_landmarks=SimpleNamespace(landmark=world))
ns['detect_pose_with_fallback']=lambda image,model,req:(pose(),None,None)
ns['landmarks_meet_visibility']=lambda results,req,threshold=True:True
ns['_visibility_values']=lambda results,req:[results.pose_landmarks.landmark[i].visibility for i in req]
frame=np.zeros((1000,1000,3),dtype=np.uint8)
ok,m=ns['analyze_front_camera'](frame,None,'右手側')
assert ok and m['is_abducted'] is True and m['is_neck_side_bending'] is False,m
ns['detect_pose_with_fallback']=lambda image,model,req:(pose(cross=True),None,None)
ok,m=ns['analyze_front_camera'](frame,None,'右手側')
assert ok and m['is_across_midline'] is True,m
ns['detect_pose_with_fallback']=lambda image,model,req:(pose(yaw=.9),None,None)
ok,m=ns['analyze_front_camera'](frame,None,'右手側')
assert not ok and m['front_auto_paused'] and m['front_rotation_warning'],m
ns['detect_pose_with_fallback']=lambda image,model,req:(pose(raise_shoulder=True),None,None)
ok,m=ns['analyze_front_camera'](frame,None,'右手側')
assert not ok and m['front_auto_paused'],m
print('PASS front auto abduction/midline on clear synthetic pose; suspend unsafe yaw/tilt and neck-side false-positive proxy')
