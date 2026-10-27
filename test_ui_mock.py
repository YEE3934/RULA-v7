"""Mock Streamlit page smoke test (no MediaPipe binary download needed).
Does not replace real Streamlit Cloud E2E testing.
"""
import io
import os
import sys
import types
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pandas as pd
import zipfile

ROOT=Path(__file__).resolve().parents[1]

class AttrDict(dict):
 def __getattr__(self,k):
  try:return self[k]
  except KeyError:raise AttributeError(k)
 def __setattr__(self,k,v):self[k]=v

class Box:
 def __init__(self, opened=True):self.open=opened
 def __getattr__(self,name):return lambda *a,**kw:None
 def __enter__(self):return self
 def __exit__(self,*args):return False

class DummyUpload:
 def __init__(self,name,data):self.name=name;self.data=data
 def getvalue(self):return self.data

class FakeStreamlit(types.ModuleType):
 def __init__(self,session):
  super().__init__('streamlit')
  self.session_state=session
  self.sidebar=Box()
  self.components=types.ModuleType('streamlit.components')
 def __getattr__(self,name):
  if name=='cache_resource':return lambda f:f
  if name=='cache_data':return lambda *a,**kw:(lambda f:f)
  if name in ('sidebar','expander','spinner','form'):return Box()
  if name=='columns':return lambda x:[Box() for _ in range(x if isinstance(x,int) else len(x))]
  if name in ('stop','rerun'):
   return lambda *a,**kw:(_ for _ in ()).throw(RuntimeError(name))
  return lambda *a,**kw:None
 def columns(self,options):return [Box() for _ in range(options if isinstance(options,int) else len(options))]
 def expander(self,*a,**kw):return Box()
 def form(self,*a,**kw):return Box()
 def tabs(self,labels,key=None,**kw):
  selected=self.session_state.get('mock_active_'+str(key),0)
  return [Box(i==selected) for i in range(len(labels))]
 def spinner(self,*a,**kw):return Box()
 def selectbox(self,label,options=None,key=None,on_change=None,**kw):
  return self.session_state.get(key,options[0] if options else None)
 def radio(self,label,options=None,key=None,**kw):
  return self.session_state.get(key,options[0] if options else None)
 def checkbox(self,label,key=None,value=False,**kw):return self.session_state.get(key,value)
 def slider(self,label,key=None,value=0,**kw):return self.session_state.get(key,value)
 def number_input(self,label,key=None,min_value=0.0,**kw):return self.session_state.get(key,min_value)
 def file_uploader(self,label,key=None,**kw):return self.session_state.get(key)
 def button(self,*a,**kw):return False
 def form_submit_button(self,*a,**kw):return False

def make_video(fps,frames):
 with tempfile.NamedTemporaryFile(suffix='.avi',delete=False) as tmp:
  path=tmp.name
 try:
  out=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'MJPG'),fps,(192,120))
  assert out.isOpened()
  for i in range(frames):
   frame=np.full((120,192,3),i%225,np.uint8)
   cv2.circle(frame,(30+(i%70),55),12,(200,200,200),-1)
   out.write(frame)
  out.release()
  return Path(path).read_bytes()
 finally:
  os.unlink(path)

uploads={'front':DummyUpload('front.avi',make_video(25,50)),
         'right':DummyUpload('right.avi',make_video(30,60)),
         'left':DummyUpload('left.avi',make_video(20,40))}

for current_page in [0,1,2,3,4,5]:
 session=AttrDict()
 session['v8_page_index']=current_page
 session['v8_values']={'eval_mode':'左右側同步計算 (需上傳正面、右手側、左手側三個視角影片)'}
 session['v8_uploads']=uploads.copy()
 fake=FakeStreamlit(session)
 sys.modules['streamlit']=fake
 sys.modules['streamlit.components']=fake.components
 components_v1=types.ModuleType('streamlit.components.v1')
 fake.components.v1=components_v1
 components_v1.html=lambda *a,**kw: None
 sys.modules['streamlit.components.v1']=components_v1
 sys.modules['mediapipe']=types.ModuleType('mediapipe')
 env={'__file__':str(ROOT/'222.py'),'__name__':'__main__'}
 try:
  exec(compile((ROOT/'222.py').read_text('utf-8'),str(ROOT/'222.py'),'exec'),env)
 except Exception as e:
  raise AssertionError(f'Streamlit mock page {current_page} failed: {type(e).__name__}: {e}') from e
 print('PASS: v9.1 mock Streamlit setup step',current_page)

# Verify completed analysis uses independent result tabs and preserves setup state.
def fake_side(prefix):
 d={'樣本編號':9,'時間(秒)':3.52,'頸部角度':12.0,'軀幹角度':18.0,'最終最高分':5,
    '頸部分數':2,'軀幹分數':2,'腿部分數':1,'肌肉使用加分':0,'負重施力加分':0}
 for part,angle,score in [('上臂',25,2),('前臂',85,1),('手腕',14,2)]:
  d[f'{prefix}{part}角度']=angle
  d[f'{prefix}{part}分數']=score
 d.update({f'{prefix}手腕扭轉分數':1,f'{prefix} Grand Score':5,f'{prefix} AL':'AL3',
    f'{prefix}Pose舊版手腕角度':14.0,f'{prefix}PoseHand實驗手腕角度':18.0,
    f'{prefix}Pose2D與Hand2D差異':4.0,f'{prefix}手腕角度_2D':14.0,
    f'{prefix}手腕角度_3D':14.0,f'{prefix}Hand品質原因':'通過',
    f'{prefix}Pose基準GrandScore':5,f'{prefix}Hand情境GrandScore':5,
    f'{prefix}Hand匹配成功':True,f'{prefix}Hand品質通過':True,
    f'{prefix}Hand用於計分':False,f'{prefix}手腕計分演算法':'Pose',
    f'{prefix}HandWorld掌心長度(模型單位)':0.07,
    f'{prefix}HandWorld掌指張角(度)':12.0,
    f'{prefix}側面Pose偵測耗時(ms)':20.0,
    f'{prefix}Hand偵測耗時(ms)':30.0,
    })
 return pd.DataFrame([d])
right=fake_side('右手')
left=fake_side('左手')
combined=pd.DataFrame([{'樣本編號':9,'時間(秒)':3.52,'頸部角度':12.0,
   '最終最高分':5,'最危害側':'雙側同分',
   '頸部_v8_候選差值':5.0,'頸部_v8_右側候選':12.0,'頸部_v8_左側候選':17.0}])
result_zip=io.BytesIO()
with zipfile.ZipFile(result_zip,'w',zipfile.ZIP_DEFLATED) as z:
 z.writestr('Report.csv',combined.to_csv(index=False))

session=AttrDict({
 'analysis_done':True,'v9_screen':'results','v8_page_index':5,
 'v8_values':{'eval_mode':'左右側同步計算 (需上傳正面、右手側、左手側三個視角影片)'},
 'v8_uploads':uploads.copy(),'v4_overrides':{},'v8_manual_rejected':{},
 'v8_rejected_rows':[],'v8_rejected_photos':{},
 'df_records':combined,'df_right_only':right,'df_left_only':left,
 'analyzed_mode':'左右側同步計算', 'display_right_b64':[], 'display_left_b64':[],
 'hand_review_images':{},'hand_model_status':'已啟用 Hand Landmarker',
 'zip_bytes':result_zip.getvalue(),'zip_filename':'Report.zip',
})
for main_tab in range(7):
 for nested in ([0,1] if main_tab==3 else [0]):
  session['mock_active_v9_result_tabs']=main_tab
  session['mock_active_v9_angle_tabs']=nested
  fake=FakeStreamlit(session)
  seen_tabs=[]
  def collect_tabs(labels,key=None,**kw):
   seen_tabs.append(list(labels))
   selected=session.get('mock_active_'+str(key),0)
   return [Box(i==selected) for i in range(len(labels))]
  fake.tabs=collect_tabs
  sys.modules['streamlit']=fake
  sys.modules['streamlit.components']=fake.components
  components_v1=types.ModuleType('streamlit.components.v1')
  components_v1.html=lambda *a,**kw:None
  fake.components.v1=components_v1
  sys.modules['streamlit.components.v1']=components_v1
  sys.modules['mediapipe']=types.ModuleType('mediapipe')
  env={'__file__':str(ROOT/'222.py'),'__name__':'__main__'}
  try:
   exec(compile((ROOT/'222.py').read_text('utf-8'),str(ROOT/'222.py'),'exec'),env)
  except Exception as e:
   raise AssertionError(f'Completed result tab {main_tab}/{nested} failed: {type(e).__name__}: {e}') from e
  assert seen_tabs and len(seen_tabs[0])==7,seen_tabs
  if main_tab==3:assert len(seen_tabs)==2 and len(seen_tabs[1])==2,seen_tabs
  else:assert len(seen_tabs)==1,seen_tabs
assert session.v9_screen=='results' and session.v8_page_index==5
assert session.v8_uploads['front'] is uploads['front']
print('PASS: all seven lazy result tabs plus two angle sub-tabs / uploads saved')

# Upgrade an existing v9 session whose result page was index 6.
legacy=AttrDict({'analysis_done':True,'v8_page_index':6,'df_records':None,'analyzed_mode':'右手側',
                  'df_right_only':None,'df_left_only':None,'zip_bytes':None,
                  'v8_uploads':{},'v8_values':{},'v4_overrides':{},
                  'display_images_b64':[],'display_right_b64':[],'display_left_b64':[]})
fake=FakeStreamlit(legacy)
fake.tabs=lambda labels,**kw:[Box(i==0) for i in range(len(labels))]
sys.modules['streamlit']=fake
sys.modules['streamlit.components']=fake.components
components_v1=types.ModuleType('streamlit.components.v1');components_v1.html=lambda *a,**kw:None
fake.components.v1=components_v1;sys.modules['streamlit.components.v1']=components_v1
sys.modules['mediapipe']=types.ModuleType('mediapipe')
exec(compile((ROOT/'222.py').read_text('utf-8'),str(ROOT/'222.py'),'exec'),
     {'__file__':str(ROOT/'222.py'),'__name__':'__main__'})
assert legacy.v9_screen=='results' and legacy.v8_page_index==5
print('PASS: legacy v9 analysis-result screen migration')

source=(ROOT/'222.py').read_text('utf-8')
assert 'v8_prev_top' not in source and 'v8_next_top' not in source
assert 'V8_STEPS = [' in source and source.count('"6｜啟動分析"')==1
assert "st.session_state.v9_screen='results'" in source
print('PASS: navigation has no top buttons; analysis finishes into result tabs')
