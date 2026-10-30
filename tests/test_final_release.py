"""Focused final-release checks: actual artifact semantics, not computer vision accuracy."""
import ast
import io
import json
import numpy as np
import unittest
import zipfile
from pathlib import Path
import pandas as pd
import test_outside_score_priority as fixture

P=Path(__file__).resolve().parents[1]/'222.py'
T=ast.parse(P.read_text(encoding='utf-8'))
NS=fixture.N.copy()
NS.update({'io':io,'zipfile':zipfile,'json':json,'np':np})
for name in ('batch_audit_table','v113_record_status','v113_build_reviewed_data','v113_key','v113_dataframe_for_side','v112_is_informational_notice','v113_finalize_report_zip','user_report_reviewed','shared_neck_reviewed_score'):
    nodes=[n for n in T.body if isinstance(n,ast.FunctionDef) and n.name==name]
    if nodes:
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(P),'exec'),NS)
for n in T.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('V113_KEEP','V113_MANUAL','V113_PENDING','V113_EXCLUDE') for t in n.targets):
        exec(compile(ast.Module(body=[n],type_ignores=[]),str(P),'exec'),NS)

class FinalRelease(unittest.TestCase):
    def get_row(self,sid,fore,upper=1,trunk=1,force=0):
        return fixture.side_row(sid,upper,2,2,1,3,trunk,1,0,force,29.1,fore,3.8,26.0,2.8)

    def test_ambiguous_score_neutral_audit_does_not_demand_confirmation(self):
        row=fixture.side_row(21,1,2,2,1,3,1,1,0,0,4.0,12.4,3.3,26.0,2.8)
        df=NS['batch_audit_table']({'右手側':pd.DataFrame([row])},{},set())
        self.assertFalse(bool(df.iloc[0]['外側工作是否需要人工確認']))
        self.assertIn('免確認',df.iloc[0]['外側工作最終採用值'])
        self.assertTrue(bool(df.iloc[0]['補正是否有未知項']))

    def test_ambiguous_score_sensitive_audit_requires_review(self):
        row=fixture.side_row(20,2,2,2,1,3,4,1,0,0,29.1,27.4,3.8,35.5,62.5)
        df=NS['batch_audit_table']({'右手側':pd.DataFrame([row])},{},set())
        self.assertTrue(bool(df.iloc[0]['外側工作是否需要人工確認']))
        self.assertIn('待確認',df.iloc[0]['外側工作最終採用值'])

    def test_explicit_no_resolves_sensitive_unknown_and_score(self):
        row=fixture.side_row(20,2,2,2,1,3,4,1,0,0,29.1,27.4,3.8,35.5,62.5)
        keys={('右手側',20):{'out_side':False}}
        frames,status,_=NS['batch_effective_frames']({'右手側':pd.DataFrame([row])},keys,{('右手側',20)})
        self.assertEqual(int(frames['右手側'].iloc[0]['右手 Grand Score']),4)
        self.assertEqual(frames['右手側'].iloc[0]['外側工作採用來源'],'人工已設定')
        self.assertEqual(frames['右手側'].iloc[0]['外側工作最終採用值'],'沒有')
        self.assertFalse(status[('右手側',20)].startswith('無法確認'))

    def test_no_phantom_final_score_when_keep_was_clicked(self):
        row=fixture.side_row(20,2,2,2,1,3,4,1,0,0,29.1,27.4,3.8,35.5,62.5)
        frames,status,_=NS['batch_effective_frames']({'右手側':pd.DataFrame([row])},{},set())
        reviewed,_=NS['v113_build_reviewed_data'](frames,[],{'20_右手側':NS['V113_KEEP']},{},batch_status=status)
        self.assertTrue(pd.isna(reviewed.iloc[0]['複核後Grand Score']))
        self.assertIn('待釐清補正',reviewed.iloc[0]['結果狀態'])

    def test_export_contains_outside_audit_and_worksite_confirmation(self):
        row=fixture.side_row(20,2,2,2,1,3,4,1,0,0,29.1,27.4,3.8,35.5,62.5)
        profiles={('右手側',20):{'out_side':False}}
        side_frames,status,_=NS['batch_effective_frames']({'右手側':pd.DataFrame([row])},profiles,{('右手側',20)})
        audit=NS['batch_audit_table']({'右手側':pd.DataFrame([row])},profiles,{('右手側',20)})
        source=io.BytesIO()
        with zipfile.ZipFile(source,'w') as z:z.writestr('Original/keep.txt','original')
        try:
            out=NS['v113_finalize_report_zip'](source.getvalue(),side_frames,[],{}, {},batch_status=status,batch_audit=audit,
                work_conditions={'work_conditions_verified':True,'actual_load_kg':'5','actual_repetition':'5','muscle':'1: test','force':'1: test'})
        except NameError as e:
            # Some optional test-only functions are not included in the AST subset; fail loudly.
            self.fail(f'ZIP export references missing helper: {e}')
        with zipfile.ZipFile(io.BytesIO(out)) as z:
            names=set(z.namelist())
            self.assertIn('Original/keep.txt',names)
            self.assertNotIn('結案驗收/驗收清單.csv',names)
            self.assertIn('評估結果/作業條件確認.csv',names)
            self.assertIn('複核後結果/外側工作逐筆判定與影響.csv',names)
            work=pd.read_csv(io.BytesIO(z.read('評估結果/作業條件確認.csv')))
            self.assertEqual(work.iloc[0]['現場條件已人工確認'],'是')
            audit=pd.read_csv(io.BytesIO(z.read('複核後結果/外側工作逐筆判定與影響.csv')))
            self.assertEqual(audit.iloc[0]['外側工作最終採用值'],'沒有')

if __name__=='__main__':unittest.main()
