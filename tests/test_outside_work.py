"""不 import Streamlit 主程式，即可檢查外側工作純函式與正面流程。
run: python -m unittest discover -s tests -v
"""
import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cv2

MAIN = Path(__file__).resolve().parents[1] / '222.py'
ORIGINAL = None
SOURCE = MAIN.read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)
FUNCTIONS = ('rula_outside_work_geometry', 'analyze_front_camera', 'get_lower_arm_score', 'v11_front_fallback_note','rula_outside_score_sensitivity')
CONSTANTS = { 'RULA_OUTSIDE_SHOULDER_YES': .30,
              'RULA_OUTSIDE_HIP_YES': .28,
              'RULA_OUTSIDE_FOREARM_YES': .16,
              'RULA_OUTSIDE_DIRECTION_YES': .26 }
NODES = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
assert len(NODES) == len(FUNCTIONS), '未找到需要測試的三個函式'
CODE = compile(ast.fix_missing_locations(ast.Module(body=NODES, type_ignores=[])), str(MAIN), 'exec')
ENV = {'np': np, 'math':math, 'cv2':cv2, **CONSTANTS}
exec(CODE, ENV)


def geom(sl=(100,100), sr=(200,100), hl=(105,200), hr=(195,200),
         elbow=(220,160), wrist=(285,160), side='右手側'):
    return ENV['rula_outside_work_geometry'](sl, sr, hl, hr, elbow, wrist, side)


class TestGeometry(unittest.TestCase):
    def test_right_side_positive(self):
        self.assertEqual(geom()['status'], '自動是')
    def test_left_side_positive(self):
        self.assertEqual(geom(elbow=(80,160), wrist=(15,160),side='左手側')['status'], '自動是')
    def test_natural_arm_down_is_no(self):
        self.assertEqual(geom(elbow=(215,170), wrist=(215,235))['status'], '自動否')
    def test_upper_arm_abduction_only_not_outside(self):
        self.assertEqual(geom(elbow=(265,110),wrist=(265,190))['status'], '自動否')
    def test_front_reach_small_projection_not_outside(self):
        self.assertEqual(geom(elbow=(202,150),wrist=(210,160))['status'], '待複核')
    def test_intermediate_not_forced_no(self):
        self.assertEqual(geom(elbow=(215,150),wrist=(230,172))['status'], '待複核')
    def test_short_forearm_needs_review(self):
        self.assertEqual(geom(elbow=(270,160),wrist=(271,163))['status'], '待複核')
    def test_missing_coordinate_needs_review(self):
        self.assertEqual(geom(wrist=(float('nan'),10))['status'], '待複核')
    def test_translation_and_scale_invariant(self):
        base=geom()['status']
        tr=lambda p: (p[0]*1.7+40,p[1]*1.7+30)
        self.assertEqual(geom(*(tr(p) for p in ((100,100),(200,100),(105,200),(195,200),(220,160),(285,160))))['status'],base)
    def test_mirrored_image_preserves_physical_side(self):
        mirror=lambda p:(400-p[0],p[1])
        self.assertEqual(geom(*(mirror(p) for p in ((100,100),(200,100),(105,200),(195,200),(220,160),(285,160))))['status'],'自動是')
    def test_rula_no_double_count(self):
        score=ENV['get_lower_arm_score']
        self.assertEqual(score(80,True,True), score(80,True,False))
        self.assertEqual(score(80,True,True), 2)
    def test_outside_is_only_one_score(self):
        self.assertEqual(ENV['get_lower_arm_score'](80,False,True),2)


def fake_landmarks(wrist_x=380,elbow_x=300,elbow_vis=0.99):
    coords={11:(150,150),12:(250,150),23:(155,260),24:(245,260),
            13:(150,195),14:(elbow_x,190),15:(120,225),16:(wrist_x,190),
            7:(170,100),8:(230,100)}
    lm=[]
    for i in range(33):
        x,y=coords.get(i,(210,210))
        visibility=elbow_vis if i==14 else 0.99
        lm.append(SimpleNamespace(x=x/500,y=y/500,visibility=visibility))
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=lm),pose_world_landmarks=None)


class TestFrontIntegration(unittest.TestCase):
    def setUp(self):
        self.frame=np.zeros((500,500,3),dtype=np.uint8)
        ENV['_visibility_values']=lambda r,req:[r.pose_landmarks.landmark[i].visibility for i in req]
        ENV['landmarks_meet_visibility']=lambda r,req,threshold:all(r.pose_landmarks.landmark[i].visibility>=threshold for i in req)
    def run_front(self,**kwargs):
        ENV['detect_pose_with_fallback']=lambda frame,model,required:(fake_landmarks(**kwargs),None,None)
        ok, r=ENV['analyze_front_camera'](self.frame,None,'右手側')
        self.assertTrue(ok)
        return r
    def test_auto_positive_integrates(self):
        r=self.run_front()
        self.assertTrue(r['is_out_to_side'])
        self.assertEqual(r['outside_work_status'],'自動是')
    def test_auto_negative_integrates(self):
        r=self.run_front(wrist_x=235,elbow_x=200)
        self.assertFalse(r['is_out_to_side'])
        self.assertEqual(r['outside_work_status'],'自動否')
    def test_low_quality_is_not_assumed_negative(self):
        r=self.run_front(elbow_vis=.35)
        self.assertFalse(r['is_out_to_side'])
        self.assertEqual(r['outside_work_status'],'待複核')
    def test_midline_cross_not_double_counted(self):
        r=self.run_front(wrist_x=125,elbow_x=180)
        self.assertTrue(r['is_across_midline'])
        self.assertFalse(r['is_out_to_side'])
        self.assertEqual(r['outside_work_status'],'自動否')


class TestFrontReviewFallback(unittest.TestCase):
    def test_rotation_while_auto_on_enters_review(self):
        qc={'品質狀態':'通過','品質原因':'原有品質正常', '品質是否列入複核中心':False,
            '品質可直接採信':True,'品質計分說明':'原有正常'}
        data={'外側工作判定狀態':'人工設定（正面不可用）'}
        result=ENV['v11_front_fallback_note'](qc,data,False,{'is_out_to_side':True})
        self.assertEqual(result['品質狀態'],'需人工複核')
        self.assertTrue(result['品質是否列入複核中心'])
        self.assertFalse(result['品質可直接採信'])
    def test_no_front_video_does_not_flood_review(self):
        qc={'品質狀態':'通過','品質原因':'原有品質正常','品質是否列入複核中心':False,
            '品質可直接採信':True,'品質計分說明':'原有正常'}
        data={'外側工作判定狀態':'人工設定（無正面）'}
        result=ENV['v11_front_fallback_note'](qc,data,False,{'is_out_to_side':False})
        self.assertEqual(result['品質狀態'],'通過')
        self.assertFalse(result['品質是否列入複核中心'])


if __name__=='__main__':
    unittest.main()
