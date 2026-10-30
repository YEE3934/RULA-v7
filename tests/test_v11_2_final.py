"""Pure-function regression tests for V11.2 review and time labels."""
import ast
import pathlib
import unittest
import numpy as np
SCRIPT=pathlib.Path(__file__).resolve().parents[1]/'222.py'
text=SCRIPT.read_text(encoding='utf-8')
parsed=ast.parse(text)
wanted={'v112_is_informational_notice','v112_split_review_items',
        'v9_sample_row','v9_camera_seconds','v9_frame_index'}
functions=[node for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name in wanted]
assert {x.name for x in functions} == wanted
ns={'np':np}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(SCRIPT),'exec'),ns)

class TestV112(unittest.TestCase):
    def test_review_vs_notice_do_not_disappear(self):
        items=[{'紀錄類型':'骨架品質提醒（仍可採信）','樣本編號':3},
               {'紀錄類型':'骨架品質待複核（有候選 RULA 分數）','樣本編號':13},
               {'評分狀態':'排除，不自動計入 RULA','樣本編號':15},
               {'紀錄類型':'其他未知紀錄','樣本編號':20}]
        review,notice=ns['v112_split_review_items'](items)
        self.assertEqual([x['樣本編號'] for x in review],[13,15,20])
        self.assertEqual([x['樣本編號'] for x in notice],[3])
        self.assertEqual(len(review)+len(notice),len(items))
    def test_time_maps_different_camera_offsets_and_positive_elapsed(self):
        m={'Front':{'fps':30.,'count':900},'Right':{'fps':25.,'count':900},'Left':{'fps':29.,'count':900}}
        a={'Front':19.5,'Right':19.75,'Left':19.75}
        row=ns['v9_sample_row'](-17.5,m,a,list(m),3,analysis_start=-19.5)
        self.assertEqual(row['時間(秒)'],-17.5)
        self.assertEqual(row['分析經過時間(秒)'],2.)
        self.assertEqual(row['Front影片時間(秒)'],2.)
        self.assertEqual(row['Right影片時間(秒)'],2.25)
        self.assertEqual(row['Right影格索引'],56)
    def test_non_actionable_photo_export_is_preserved(self):
        # Source retains all photos and exports informational records separately.
        self.assertIn('v8_zip_rejected_photos(zf,rejected_photos)',text)
        self.assertIn('Diagnostics/Quality_Notices.csv',text)
        self.assertIn('Diagnostics/Rejected_Samples_Detail.csv',text)
    def test_scoring_corrected_against_reference(self):
        # Regression: old v11 matrix was wrong; it MUST NOT remain unchanged.
        expr=next(n for n in parsed.body if isinstance(n,ast.Assign)
                  and any(isinstance(x,ast.Name) and x.id=='TABLE_A' for x in n.targets))
        table=ast.literal_eval(expr.value)
        self.assertEqual(table[6][3][4][2],9)
        self.assertEqual(table[1][2][2][2],2)
        self.assertIn('TABLE_C[min(max(score_c, 1), 9)',text)

if __name__=='__main__':unittest.main()
