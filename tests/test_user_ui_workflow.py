import ast
from pathlib import Path
import unittest
import pandas as pd
P=Path(__file__).resolve().parents[1]/'222.py'
S=P.read_text(encoding='utf-8')
T=ast.parse(S)
fn=next(x for x in T.body if isinstance(x,ast.FunctionDef) and x.name=='rula_pending_review_records')
NS={}
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(P),'exec'),NS)

class UserFacingReview(unittest.TestCase):
    def test_pending_records_merge_two_neck_sides(self):
        df=pd.DataFrame([
            {'樣本編號':4,'評估側':'右手側','結果狀態':'待釐清補正（需人工確認）','照片補正狀態':'無法確認','頸部確認狀態':''},
            {'樣本編號':12,'評估側':'右手側','結果狀態':'待頸部局部人工確認','照片補正狀態':'','頸部確認狀態':'待人工確認'},
            {'樣本編號':12,'評估側':'左手側','結果狀態':'待頸部局部人工確認','照片補正狀態':'','頸部確認狀態':'待人工確認'},
            {'樣本編號':20,'評估側':'右手側','結果狀態':'待釐清補正','照片補正狀態':'無法確認','頸部確認狀態':''},
            {'樣本編號':22,'評估側':'右手側','結果狀態':'待頸部局部人工確認','照片補正狀態':'','頸部確認狀態':'待人工確認'},
            {'樣本編號':22,'評估側':'左手側','結果狀態':'待頸部局部人工確認','照片補正狀態':'','頸部確認狀態':'待人工確認'},
            {'樣本編號':1,'評估側':'右手側','結果狀態':'自動品質通過','照片補正狀態':'正常','頸部確認狀態':''},
            {'樣本編號':15,'評估側':'左手側','結果狀態':'辨識失敗／無原始分數','照片補正狀態':'','頸部確認狀態':''},
        ])
        rows=NS['rula_pending_review_records'](df)
        self.assertEqual([(x['照片編號'],x['類型']) for x in rows],[(4,'前臂'),(12,'頸部'),(20,'前臂'),(22,'頸部')])
        self.assertEqual(rows[0]['評估側'],'右手側')
    def test_nothing_pending(self):
        self.assertEqual(NS['rula_pending_review_records'](pd.DataFrame()),[])
    def test_review_is_in_user_side_and_closure_is_not(self):
        self.assertIn('rula_pending_photo_panel()',S)
        self.assertIn("'✅ 照片補正'",S)
        self.assertIn("'🧠 頸部確認'",S)
        self.assertIn('"4｜啟動分析"',S)
        self.assertIn("if page == 3:\n    st.subheader('啟動分析')", S)
        self.assertNotIn("if page == 4:\n    st.subheader('啟動分析')", S)
        self.assertNotIn("'✅ 結案驗收'",S)
        self.assertNotIn("'結案驗收/驗收清單.csv'",S)
        self.assertNotIn('人工確認條件（必要項）',S)
    def test_timeline_is_open_by_default_and_has_native_alternative(self):
        self.assertIn("with st.expander('🎬 影片時間軸｜拖曳區間設定負重與肌肉使用', expanded=True)",S)
        self.assertIn("st.slider('選擇起點與終點（秒）'",S)
        self.assertIn('st.video(ref_file)',S)
if __name__=='__main__':unittest.main()
