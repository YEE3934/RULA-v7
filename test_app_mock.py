"""Mocks Streamlit to test startup controls, plus fake-landmark side analysis.
Does NOT exercise actual MediaPipe model or Streamlit Cloud browser.
"""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys, runpy, numpy as np

class Session(dict):
    def __getattr__(self,k): return self[k]
    def __setattr__(self,k,v): self[k]=v
class Ctx:
    def __enter__(self): return self
    def __exit__(self,*args): return False
class StreamlitStub(ModuleType):
    def __init__(self):
        super().__init__('streamlit'); self.session_state=Session()
    def cache_resource(self,f): return lambda *a,**kw:None
    def __getattr__(self,k):
        if k in ('tabs','columns'):
            return lambda x,*a,**kw:[Ctx() for _ in (x if isinstance(x,list) else range(x))]
        if k in ('container','expander'):
            return lambda *a,**kw:Ctx()
        if k=='checkbox': return lambda *a,**kw:kw.get('value',False)
        if k=='radio': return lambda *a,**kw:a[1][kw.get('index',0)] if len(a)>1 else kw['options'][kw.get('index',0)]
        if k=='selectbox':return lambda *a,**kw:a[1][kw.get('index',0)] if len(a)>1 else kw['options'][kw.get('index',0)]
        if k=='slider':return lambda *a,**kw:kw.get('value',25)
        if k=='button':return lambda *a,**kw:False
        if k=='file_uploader':return lambda *a,**kw:None
        return lambda *a,**kw:None

st=StreamlitStub()
c1=ModuleType('streamlit.components'); c2=ModuleType('streamlit.components.v1')
c2.html=lambda *a,**kw:None
st.__path__=[]; c1.__path__=[]; st.components=c1; c1.v1=c2
sys.modules.update({'streamlit':st,'streamlit.components':c1,'streamlit.components.v1':c2,
                    'mediapipe':ModuleType('mediapipe')})
source=str(Path(__file__).resolve().parents[1]/'222.py')
namespace=runpy.run_path(source)
assert 'eval_mode' in namespace
assert namespace['auto_selection']['is_abducted'] is True
assert namespace['v6_hand_policy']=='compare'
assert namespace['raw_image_export'] is True
assert not st.session_state.analysis_done
print('PASS startup mocked settings and tabs')

# Simulate MediaPipe Pose landmarks; real image geometry operations run on synthetic points.
import cv2

def lm(x,y,z=0):return SimpleNamespace(x=float(x),y=float(y),z=float(z),visibility=.98)
coords={
 11:(.43,.33),12:(.57,.33),13:(.39,.50),14:(.61,.50),
 15:(.37,.68),16:(.66,.67),17:(.37,.74),18:(.68,.73),
 19:(.36,.74),20:(.71,.73),21:(.35,.71),22:(.7,.70),
 23:(.44,.66),24:(.56,.66),25:(.44,.83),26:(.56,.83),
 27:(.44,.94),28:(.56,.94),29:(.44,.94),30:(.56,.94),
 31:(.44,.97),32:(.56,.97),7:(.43,.20),8:(.57,.20),
 0:(.50,.16),2:(.48,.16),5:(.52,.16)
}
pts=[lm(*coords.get(i,(.50,.3))) for i in range(33)]
world=[lm(v.x,v.y,v.z) for v in pts]
res=SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=pts),
                    pose_world_landmarks=SimpleNamespace(landmark=world))
class PoseFake:
    def process(self, image):return res
ns=namespace
ns['draw_full_biomechanical_skeleton']=lambda img,*a,**kw:img
# Important: overriding runpy dictionary may not alter function globals; modify globals dict instead
fn=ns['analyze_side_camera']
fn.__globals__['draw_full_biomechanical_skeleton']=lambda img,*a,**kw:img
fn.__globals__['FONT_FILE_PATH']=None
img=np.zeros((600,400,3),dtype=np.uint8)
manual={'leg_score':1,'wrist_twist_score':1,'auto_selection':{},'manual_front':{}}
front={'is_abducted':False,'is_across_midline':False,'is_out_to_side':False,
       'is_side_bending':False,'is_neck_side_bending':False}
success,data,rendered,conf=fn(img,PoseFake(),'右手側',front,0,0,manual,hand_task=None)
assert success,(success,data,conf)
assert data['右手上臂分數'] in range(1,7)
assert data['右手前臂分數'] in range(1,4)
assert data['右手手腕計分演算法']=='Pose v5'
assert '_v7_keypoints' in data and len(data['_v7_keypoints']['Pose影像像素座標'])==21
assert not data['右手Hand品質通過']
print('PASS synthetic complete side analysis with absent Hand (Pose fallback)')
# Synthetic Hand detected on the same target wrist, matches 2D arm projection.
mpstub=sys.modules['mediapipe']
mpstub.ImageFormat=SimpleNamespace(SRGB=1)
mpstub.Image=lambda **kw:kw['data']
wh_x,wh_y=coords[16]
hp=[lm(wh_x,wh_y) for _ in range(21)]
for i in [5,9,13,17]:hp[i]=lm(wh_x+.02,wh_y+.067)
handres=SimpleNamespace(hand_landmarks=[hp],hand_world_landmarks=[],handedness=[])
class HandTaskFake:
    def detect(self,img):return handres
success,data,rendered,conf=fn(img,PoseFake(),'右手側',front,0,0,manual,
                               hand_task=HandTaskFake(),hand_scoring_policy='gated')
assert success,(success,data,conf)
assert data['右手Hand匹配成功'],data
assert data['右手Pose2D與Hand2D差異']<20,data
assert data['右手Hand品質通過'],data['右手Hand品質原因']
assert data['右手Hand用於計分']
print('PASS synthetic Pose+Hand match, fair 2D comparison and gated score')
