"""Regression: score-neutral outside uncertainty must not demand 1-by-1 confirmation.
Samples 5, 20, 21 represent score components read from prior manual carry ZIP;
no claiming vision accuracy or estimating anatomical side reach from the fixtures.
"""
import ast
import unittest
from pathlib import Path
import pandas as pd

P=Path(__file__).resolve().parents[1]/'222.py'
T=ast.parse(P.read_text(encoding='utf-8'))
N={'pd':pd}
need_assign={'TABLE_A','TABLE_B','TABLE_C','BATCH_BOOL_FIELDS','BATCH_NUMBER_FIELDS','BATCH_FIELDS'}
for n in T.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in need_assign for t in n.targets):
        exec(compile(ast.Module(body=[n],type_ignores=[]),str(P),'exec'),N)
N['BATCH_FIELDS'].update({a:b for a,b,lo,hi in N['BATCH_NUMBER_FIELDS']})
funcs={'get_upper_arm_score','get_lower_arm_score','get_wrist_score','get_neck_score',
       'get_trunk_score','compute_rula_full','rula_outside_score_sensitivity',
       'batch_base_profile','batch_get_profile','batch_recalculate_side','batch_effective_frames'}
exec(compile(ast.Module(body=[n for n in T.body if isinstance(n,ast.FunctionDef) and n.name in funcs], type_ignores=[]),str(P),'exec'),N)

def side_row(id, upper,fore,wrist,twist,neck,trunk,leg,muscle,force, ua_angle,fore_angle,wrist_angle,neck_angle,trunk_angle):
    return {'樣本編號':id, '時間(秒)':float(id), '外側工作判定狀態':'待複核',
            '右手上臂分數':upper,'右手前臂分數':fore,'右手手腕分數':wrist,
            '右手手腕扭轉分數':twist,'頸部分數':neck,'軀幹分數':trunk,
            '腿部分數':leg,'肌肉使用加分':muscle,'負重施力加分':force,
            '右手上臂角度':ua_angle,'右手前臂角度':fore_angle,
            '右手手腕角度':wrist_angle,'頸部角度':neck_angle,'軀幹角度':trunk_angle,
            '右手橫越中線補正':'否','右手前臂向外側工作補正':'待複核（暫按 0 補正）',
            '右手 Grand Score':3,'右手 AL':'AL2'}

class ScoringImpact(unittest.TestCase):
    def test_high_load_case_cannot_change_grand(self):
        row=side_row(5,4,1,2,1,3,2,1,0,3,123.8,84.9,2.5,22.0,6.3)
        impact=N['rula_outside_score_sensitivity'](row,'右手側')
        self.assertEqual(impact['no_outside'],(7,'AL4'))
        self.assertEqual(impact['yes_outside'],(7,'AL4'))
        self.assertTrue(impact['same_grand'])
    def test_sample20_action_level_changes_needs_review(self):
        row=side_row(20,2,2,2,1,3,4,1,0,0,29.1,27.4,3.8,35.5,62.5)
        impact=N['rula_outside_score_sensitivity'](row,'右手側')
        self.assertEqual(impact['no_outside'],(4,'AL2'))
        self.assertEqual(impact['yes_outside'],(5,'AL3'))
        self.assertFalse(impact['same_grand'])
    def test_sample21_no_score_change(self):
        row=side_row(21,1,2,2,1,3,1,1,0,0,4.0,12.4,3.3,26.0,2.8)
        impact=N['rula_outside_score_sensitivity'](row,'右手側')
        self.assertEqual(impact['no_outside'],(3,'AL2'))
        self.assertEqual(impact['yes_outside'],(3,'AL2'))
        self.assertTrue(impact['same_grand'])
    def test_batch_safe_row_does_not_force_manual(self):
        row=side_row(21,1,2,2,1,3,1,1,0,0,4.0,12.4,3.3,26.0,2.8)
        frames,status,_=N['batch_effective_frames']({'右手側':pd.DataFrame([row])},{},set())
        self.assertIn('總分相同', status[('右手側',21)])
        self.assertEqual(int(frames['右手側'].iloc[0]['右手 Grand Score']),3)
        self.assertEqual(frames['右手側'].iloc[0]['外側工作判定狀態'],'待複核')
    def test_batch_sensitive_row_remains_unconfirmed(self):
        row=side_row(20,2,2,2,1,3,4,1,0,0,29.1,27.4,3.8,35.5,62.5)
        frames,status,_=N['batch_effective_frames']({'右手側':pd.DataFrame([row])},{},set())
        self.assertTrue(status[('右手側',20)].startswith('無法確認'))
        self.assertNotIn('外側工作分數敏感度',frames['右手側'].columns)

if __name__=='__main__':unittest.main()
