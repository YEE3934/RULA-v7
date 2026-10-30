"""Standalone production-function and integration assertions for the staged RULA scores."""
import ast
import pathlib
import unittest
import numpy as np
import pandas as pd

P=pathlib.Path(__file__).resolve().parents[1]/'222.py'
source=P.read_text(encoding='utf-8')
tree=ast.parse(source)
functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
required=['rula_timeline_rules','rula_timeline_at','rula_timeline_video_fields',
          'v9_sample_row','v9_camera_seconds','v9_frame_index',
          'compute_rula_full','get_upper_arm_score','get_lower_arm_score',
          'get_wrist_score','get_neck_score','get_trunk_score',
          'batch_base_profile','batch_recalculate_side','batch_update_profiles',
          'batch_get_profile','batch_effective_frames','shared_neck_rula_with_score',
          'shared_neck_apply_to_side','shared_neck_decide_pair']
assert all(name in functions for name in required)
vars={}
for n in tree.body:
    if isinstance(n,ast.Assign):
        for t in n.targets:
            if isinstance(t,ast.Name) and t.id in ('TABLE_A','TABLE_B','TABLE_C','BATCH_BOOL_FIELDS','BATCH_NUMBER_FIELDS'):
                vars[t.id]=ast.literal_eval(n.value)
ns={'np':np,'pd':pd,**vars}
ns['BATCH_FIELDS']={name:label for name,label in vars['BATCH_BOOL_FIELDS']}
ns['BATCH_FIELDS'].update({name:label for name,label,lo,hi in vars['BATCH_NUMBER_FIELDS']})
exec(compile(ast.Module(body=[functions[k] for k in required], type_ignores=[]),str(P),'exec'),ns)

class TestTimeline(unittest.TestCase):
    def test_time_intervals_and_baseline(self):
        config={'timeline_enabled':True,'timeline_count':2,
                'timeline_0_title':'搬運','timeline_0_start':4.,'timeline_0_end':12.,
                'timeline_0_muscle':1,'timeline_0_force':2,
                'timeline_1_title':'放置','timeline_1_start':12.,'timeline_1_end':17.,
                'timeline_1_muscle':0,'timeline_1_force':1}
        rules=ns['rula_timeline_rules'](config,21.)
        self.assertEqual([x['name'] for x in rules],['搬運','放置'])
        for time,expected in [(0,(0,0)),(3.99,(0,0)),(4,(1,2)),(9.5,(1,2)),
                              (11.999,(1,2)),(12,(0,1)),(16.999,(0,1)),
                              (17,(0,1)),(17.1,(0,0)),(21,(0,0))]:
            self.assertEqual(ns['rula_timeline_at'](time,rules,0,0)[:2],expected)
        self.assertEqual(ns['rula_timeline_at'](11,[],1,3)[:2],(1,3))
        self.assertEqual(ns['rula_timeline_rules']({'timeline_enabled':False},2),[])

    def test_rejects_bad_intervals(self):
        x={'timeline_enabled':True,'timeline_count':2,
           'timeline_0_start':4.,'timeline_0_end':12.,'timeline_0_muscle':0,'timeline_0_force':1,
           'timeline_1_start':10.,'timeline_1_end':16.,'timeline_1_muscle':0,'timeline_1_force':1}
        for update,duration in [({},21),({'timeline_1_start':17,'timeline_1_end':24},21),
                                ({'timeline_0_end':4.},21),({'timeline_1_force':7},21),
                                ({'timeline_1_start':-1},21)]:
            with self.assertRaises(ValueError): ns['rula_timeline_rules']({**x,**update},duration)

    def test_camera_alignment_uses_side_native_clock(self):
        md={'Right':{'fps':30.,'count':630},'Left':{'fps':25.,'count':550},'Front':{'fps':30.,'count':630}}
        anchors={'Right':8.0,'Left':10.5,'Front':5.0}
        record=ns['v9_sample_row'](-2.5,md,anchors,['Front','Right','Left'],7,analysis_start=-8.0)
        self.assertEqual(record['Right影片時間(秒)'],5.5)
        self.assertEqual(record['Left影片時間(秒)'],8.0)
        self.assertEqual(record['Front影片時間(秒)'],2.5)
        # Right side is the user-visible time source in dual-view, not Front or the negative shared clock.
        rules=[{'start':4.,'end':12.,'name':'正在搬運','muscle':1,'force':2}]
        for side in ('右手側','左手側'):
            m,f,source=ns['rula_timeline_at'](record['Right影片時間(秒)'],rules,0,0)
            self.assertEqual((m,f),(1,2),side)
            self.assertIn('正在搬運',source)
        # Using another camera's time would be a bug at a boundary.
        self.assertEqual(ns['rula_timeline_at'](record['Front影片時間(秒)'],rules,0,0)[:2],(0,0))

    def test_used_scores_feed_production_rula_table(self):
        calc=ns['compute_rula_full']
        regular=calc(3,1,2,1,2,3,1,0,0)
        lifting=calc(3,1,2,1,2,3,1,1,2)
        self.assertNotEqual(regular[2:4],lifting[2:4])
        self.assertTrue(lifting[4]>=regular[4])

    def test_manual_photo_edit_overrides_staged_default(self):
        side='右手側';pre='右手'
        source_row={'樣本編號':2,'時間(秒)':5.5,'參考側影片時間(秒)':5.5,
            '作業條件設定來源':'影片時段：搬運（4～12秒）',
            '頸部角度':14.,'軀幹角度':24.,'肌肉使用加分':1,'負重施力加分':2,
            '腿部分數':1,'頸部側彎補正':'否','軀幹側彎補正':'否','軀幹扭轉補正':'否',
            '頸部扭轉補正':'否','軀幹向後伸展':'否','頸部後仰／伸展':'否',
            pre+'上臂角度':35.,pre+'前臂角度':95.,pre+'手腕角度':8.,
            pre+'上臂外展補正':'否',pre+'橫越中線補正':'否',pre+'前臂向外側工作補正':'否',
            pre+'肩部抬高補正':'否',pre+'上臂支撐／倚靠補正':'否',pre+'手腕偏移補正':'否',
            pre+'上臂向後伸展補正':'否',pre+'手腕扭轉分數':1}
        base=ns['batch_base_profile'](source_row,side)
        self.assertEqual((base['muscle'],base['force']),(1,2))
        edited,confirmed=ns['batch_update_profiles']({},set(),[(side,2)],{'force':0})
        effective=ns['batch_get_profile'](source_row,side,edited[(side,2)])
        self.assertEqual((effective['muscle'],effective['force']),(1,0))
        changed=ns['batch_recalculate_side'](source_row,side,effective)
        self.assertEqual((changed['肌肉使用加分'],changed['負重施力加分']),(1,0))
        self.assertIn((side,2),confirmed)

    def test_real_analysis_call_and_export_paths_are_wired(self):
        self.assertEqual(source.count('rula_timeline_at(local_ref_time,timeline_rules,m_val,f_val)'),2)
        self.assertIn('r_frame, pose, "右手側", front_mods_r, m_at, f_at',source)
        self.assertIn('l_frame, pose, "左手側", front_mods_l, m_at, f_at',source)
        self.assertIn('s_frame, pose, target_side, front_mods, m_at, f_at',source)
        self.assertIn("sync_samples[i+1][timeline_ref+'影片時間(秒)']",source)
        self.assertIn('評估結果/逐張時段與作業條件.csv',source)
        self.assertIn('研究驗證/人工評分比對空表.csv',source)

if __name__=='__main__':unittest.main()
