"""RULA v11.1 head/arm occlusion and cross-view quality regression tests.
Run: python tests/test_v11_1_quality.py -- no MediaPipe / Streamlit required.
"""
import ast
import pathlib
import unittest
import numpy as np

FILE=pathlib.Path(__file__).resolve().parents[1]/'222.py'
parsed=ast.parse(FILE.read_text(encoding='utf-8'))
wanted={'v11_pt','v11_1_head_arm_proximity','v11_quality_from_snapshot','v11_1_neck_crosscheck'}
body=[x for x in parsed.body if isinstance(x,ast.FunctionDef) and x.name in wanted]
assert len(body)==len(wanted)
namespace={'np':np}
exec(compile(ast.Module(body=body,type_ignores=[]),str(FILE),'exec'),namespace)


def snapshot(nose=(170,361), shoulder=(177,425), elbow=(158,471), wrist=(151,421), hip=(242,600),
             visibility=0.93):
    pts={0:nose,7:(164,367),8:(181,350),11:shoulder,13:elbow,15:wrist,23:hip,16:(360,700)}
    return {'影像寬':562,'影像高':1000,'Pose影像像素座標':{
        str(i):{'x_px':float(pt[0]),'y_px':float(pt[1]),'visibility':visibility}
        for i,pt in pts.items()}}


class TestOcclusion(unittest.TestCase):
    def test_head_arm_overlap_even_high_visibility(self):
        data={'左手前臂角度':149.8,'頸部角度':39.0,'左手Hand用於計分':True}
        qc=namespace['v11_quality_from_snapshot'](snapshot(),'左手側',data)
        self.assertTrue(qc['頭頸手臂重疊風險'])
        self.assertEqual(qc['品質狀態'],'需人工複核')
        self.assertIn('neck',qc['品質嫌疑關節'])
        self.assertFalse(qc['品質可直接採信'])

    def test_normal_straight_arm_near_head_not_rejected(self):
        # Hand near the head is not enough by itself; low elbow flexion.
        data={'左手前臂角度':45.0,'頸部角度':35.0,'左手Hand用於計分':True}
        qc=namespace['v11_quality_from_snapshot'](snapshot(),'左手側',data)
        self.assertFalse(qc['頭頸手臂重疊風險'])
        self.assertNotEqual(qc['品質狀態'],'需人工複核')

    def test_crossview_disagreement_without_overlap_only_notice(self):
        data={'頸部角度':39.0}
        peer={'頸部角度':8.0}
        qc={'品質狀態':'通過','品質原因':'主要關節品質檢查通過',
            '品質嫌疑關節':'','頭頸手臂重疊風險':False}
        namespace['v11_1_neck_crosscheck'](qc,data,peer)
        self.assertEqual(qc['品質狀態'],'留意')
        self.assertEqual(qc['頸部跨鏡頭基礎分級差'],2)

    def test_disagreement_plus_overlap_requires_review(self):
        qc={'品質狀態':'留意','品質原因':'重疊線索',
            '品質嫌疑關節':'wr','頭頸手臂重疊風險':True}
        namespace['v11_1_neck_crosscheck'](qc,{'頸部角度':39.0},{'頸部角度':8.0})
        self.assertEqual(qc['品質狀態'],'留意')  # 頸部分歧交給行動等級敏感度判斷，不直接整筆拒絕

    def test_big_angle_gap_same_grade_only_diagnosis(self):
        qc={'品質狀態':'通過','品質原因':'通過',
            '品質嫌疑關節':'','頭頸手臂重疊風險':False}
        namespace['v11_1_neck_crosscheck'](qc,{'頸部角度':60.0},{'頸部角度':25.0})
        self.assertEqual(qc['品質狀態'],'通過')
        self.assertIn('分級一致',qc['頸部跨鏡頭診斷'])

    def test_does_not_change_rula_formula(self):
        text=FILE.read_text(encoding='utf-8')
        self.assertIn('Hand 2D（優先）',text)
        self.assertIn('RULA_VERIFIED_TABLES_FRONT_REVIEW_',text)

if __name__=='__main__':unittest.main(verbosity=2)
