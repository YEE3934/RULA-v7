"""新增三態與正式輸出防誤採信：無需 Streamlit/MediaPipe 即可驗證。"""
from pathlib import Path
from types import SimpleNamespace
import ast
import unittest
import pandas as pd

P=Path(__file__).resolve().parents[1]/'222.py'
T=ast.parse(P.read_text(encoding='utf-8'))
F={n.name:n for n in T.body if isinstance(n,ast.FunctionDef)}
N=['batch_base_profile','batch_get_profile','batch_recalculate_side','batch_effective_frames',
   'v113_key','v113_record_status','v113_dataframe_for_side','v113_build_reviewed_data',
   'v11_front_fallback_note','rula_outside_score_sensitivity',
   'get_lower_arm_score','compute_rula_full']
TABLES={}
for n in T.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('TABLE_A','TABLE_B','TABLE_C') for t in n.targets):
        exec(compile(ast.Module(body=[n],type_ignores=[]),str(P),'exec'),TABLES)
NS={'pd':pd,'BATCH_FIELDS':{'out_side':'前臂向外工作'},**TABLES,
    'V113_KEEP':'維持／確認使用自動評分','V113_MANUAL':'改用人工重新評分',
    'V113_EXCLUDE':'剔除此側樣本'}
exec(compile(ast.Module(body=[F[n] for n in N],type_ignores=[]),str(P),'exec'),NS)

class PendingTest(unittest.TestCase):
    def row(self,status):
        return {'樣本編號':1,'時間(秒)':0.,'右手 Grand Score':4,'右手 AL':'AL2',
                '右手前臂向外側工作補正':'否（影像代理判斷）',
                '外側工作判定狀態':status}
    def test_unknown_not_false(self):
        for status in ('待複核','人工設定（正面不可用）'):
            self.assertIsNone(NS['batch_base_profile'](self.row(status),'右手側')['out_side'])
    def test_manual_or_automatic_class_kept(self):
        for status,expected in [('自動否',False),('自動是',True),('人工設定',False),('人工設定（無正面）',False)]:
            row=self.row(status)
            row['右手前臂向外側工作補正']='是 (+1分)' if expected else '否'
            self.assertEqual(NS['batch_base_profile'](row,'右手側')['out_side'],expected)
    def test_manual_override_resolves_unknown(self):
        row=self.row('待複核')
        self.assertIsNone(NS['batch_get_profile'](row,'右手側',{})['out_side'])
        self.assertTrue(NS['batch_get_profile'](row,'右手側',{'out_side':True})['out_side'])
        self.assertFalse(NS['batch_get_profile'](row,'右手側',{'out_side':False})['out_side'])
    def test_front_fail_with_uploaded_front_is_review(self):
        qc={'品質狀態':'通過','品質可直接採信':True,'品質是否列入複核中心':False,'品質原因':''}
        data={'外側工作判定狀態':'人工設定（正面不可用）'}
        result=NS['v11_front_fallback_note'](qc,data,False,{'is_out_to_side':True})
        self.assertTrue(result['品質是否列入複核中心'])
        self.assertEqual(result['品質狀態'],'需人工複核')
    def test_absent_front_manual_setting_does_not_force_review(self):
        qc={'品質狀態':'通過','品質可直接採信':True,'品質是否列入複核中心':False,'品質原因':''}
        data={'外側工作判定狀態':'人工設定（無正面）'}
        result=NS['v11_front_fallback_note'](qc,data,False,{'is_out_to_side':False,'is_abducted':True})
        self.assertFalse(result['品質是否列入複核中心'])
    def test_keep_provisional_does_not_publish(self):
        # An unrelated QC confirm cannot turn an unknown outside-work into an accepted Grand Score.
        row=self.row('待複核')
        res,_con=NS['v113_build_reviewed_data'](
            {'右手側':pd.DataFrame([row])},[],{'1_右手側':NS['V113_KEEP']}, {},
            batch_status={('右手側',1):'無法確認：前臂向外工作'})
        got=res.iloc[0]
        self.assertTrue(pd.isna(got['複核後Grand Score']))
        self.assertIn('待釐清補正',got['結果狀態'])

if __name__ == '__main__':unittest.main()
