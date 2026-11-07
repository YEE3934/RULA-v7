"""Regression tests for stable-version review triage and failure reasons.
No Streamlit/MediaPipe installation required; the real scoring code remains untouched.
"""
import ast
import csv
import io
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT/'222.py').read_text(encoding='utf-8'))
FN = {n.name:n for n in TREE.body if isinstance(n,ast.FunctionDef)}

def extracted(*names, extra=None):
    ns={'np':__import__('numpy'),'logging':__import__('logging')}
    ns['V14_LANDMARK_LABELS']=next(ast.literal_eval(n.value) for n in TREE.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='V14_LANDMARK_LABELS' for t in n.targets))
    if extra:ns.update(extra)
    for name in names:
        exec(compile(ast.Module(body=[FN[name]],type_ignores=[]),str(ROOT/'222.py'),'exec'),ns)
    return ns

class StableReviewQ14(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns=extracted('v112_is_informational_notice','v14_is_auto_excluded','v14_split_review_display','v14_diagnose_failed_side','v14_failure_info')

    def test_one_of_each_category(self):
        rows=[{'評分狀態':'排除，不自動計入 RULA','拒絕原因':'右手腕低可見度'},
              {'紀錄類型':'骨架品質待複核（有RULA 系統分數）','評分狀態':'僅系統評分'},
              {'紀錄類型':'骨架品質提醒（仍可採信）'}]
        excluded,review,notice=self.ns['v14_split_review_display'](rows)
        self.assertEqual((len(excluded),len(review),len(notice)),(1,1,1))
        self.assertIs(excluded[0],rows[0])

    def test_failure_diagnostic_no_image(self):
        d=self.ns['v14_diagnose_failed_side'](None,None,'右手側')
        self.assertIn('影片影格',d['主要原因'])

    def test_failure_diagnostic_specific_joint(self):
        import numpy as np
        keypoints=[SimpleNamespace(x=.5,y=.6,visibility=.9) for _ in range(33)]
        keypoints[16].visibility=.1
        result=SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=keypoints))
        self.ns['detect_pose_with_fallback']=lambda frame,pose,inds:(result,.4,'original')
        self.ns['cv2']=__import__('cv2')
        d=self.ns['v14_diagnose_failed_side'](np.zeros((100,100,3),np.uint8),object(),'右手側')
        self.assertIn('右手腕',d['主要原因'])
        self.assertIn('重讀影格',d['診斷方式'])

    def test_real_inspection_zip_counts(self):
        path=Path('/mnt/data/品檢.zip')
        if not path.exists():self.skipTest('user inspection ZIP unavailable')
        import pandas as pd
        with zipfile.ZipFile(path) as z:
            d=pd.read_csv(io.BytesIO(z.read('技術診斷/待處理影格.csv'))).fillna('')
        ex,rev,info=self.ns['v14_split_review_display'](d.to_dict('records'))
        self.assertEqual((len(ex),len(rev),len(info)),(16,11,0))

if __name__=='__main__':unittest.main()
