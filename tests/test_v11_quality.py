"""V11 pure-function quality smoke tests. Run: python tests/test_v11_quality.py
Does not require Streamlit, MediaPipe or the actual hand model.
"""
import ast
import math
import pathlib
import unittest
import numpy as np

SCRIPT=pathlib.Path(__file__).resolve().parents[1] / '222.py'
source=SCRIPT.read_text(encoding='utf-8')
parsed=ast.parse(source)
names={
 'v11_pt','v11_1_head_arm_proximity','v11_1_neck_crosscheck','v11_quality_from_snapshot','v11_normalized_points_from_landmarks',
 'v11_temporal_recheck','v11_update_qc',
}
mini=ast.Module(body=[node for node in parsed.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names],type_ignores=[])
namespace={'np':np, 'V11_PROBE_SECONDS':0.12, 'v11_cross_view_hint':lambda *args,**kw:{'跨視角診斷':'模擬','其他鏡頭腕肘最低可見度':0.9}}
exec(compile(ast.fix_missing_locations(mini),str(SCRIPT),'exec'), namespace)


def snapshot(wrist_vis=.98, elbow_vis=.98, hand_right=True, forearm=(130,100), other_wrist=(215,100)):
    xy={12:(100,100),14:forearm,16:(165,100),24:(100,220),15:other_wrist}
    if not hand_right:
        xy={11:xy[12],13:xy[14],15:xy[16],23:xy[24],16:other_wrist}
    vis={14 if hand_right else 13:elbow_vis,16 if hand_right else 15:wrist_vis}
    return {'影像寬':360,'影像高':280,'Pose影像像素座標':{str(i):{'x_px':a,'y_px':b,'visibility':vis.get(i,.98)} for i,(a,b) in xy.items()}}


class TestV11Quality(unittest.TestCase):
    def test_normal_right_passes(self):
        qc=namespace['v11_quality_from_snapshot'](snapshot(),'右手側',{'右手Hand用於計分':True})
        self.assertEqual(qc['品質狀態'],'通過')
        self.assertTrue(qc['品質可直接採信'])

    def test_normal_left_passes(self):
        qc=namespace['v11_quality_from_snapshot'](snapshot(hand_right=False),'左手側',{'左手Hand用於計分':True})
        self.assertEqual(qc['品質狀態'],'通過')

    def test_occlusion_plus_no_hand_quarantine(self):
        qc=namespace['v11_quality_from_snapshot'](snapshot(wrist_vis=.27),'右手側',{'右手Hand用於計分':False})
        self.assertEqual(qc['品質狀態'],'需人工複核')
        self.assertFalse(qc['品質可直接採信'])
        self.assertIn('wr',qc['品質嫌疑關節'])

    def test_high_flex_triggers_probe_but_not_auto_rejection(self):
        qc=namespace['v11_quality_from_snapshot'](snapshot(),'右手側',{'右手Hand用於計分':True,'右手前臂角度':177.7})
        self.assertEqual(qc['品質狀態'],'留意')
        self.assertTrue(qc['品質需時序複查'])

    def test_opposite_wrist_overlap_soft_warning(self):
        qc=namespace['v11_quality_from_snapshot'](snapshot(other_wrist=(167,99)),'右手側',{'右手Hand用於計分':True})
        self.assertEqual(qc['品質狀態'],'留意')

    def test_temporal_isolated_spike(self):
        middle=snapshot()
        # same body scale: wrist at 165 -> normalized wr 0.54 from shoulder
        middle['Pose影像像素座標']['16']['x_px']=245  # +80px => +0.67 torso lengths
        n1={'el':np.array([.25,0]),'wr':np.array([.54,0]),'visibility':{'el':.9,'wr':.9}}
        n2={'el':np.array([.25,.01]),'wr':np.array([.56,.01]),'visibility':{'el':.9,'wr':.9}}
        class Cap:
            def __init__(self):self.pos=22
            def get(self,_):return self.pos
            def set(self,_,x):self.pos=x
        cap=Cap()
        # patch helper in a cloned globals namespace
        fn=namespace['v11_temporal_recheck']
        old=fn.__globals__.get('v11_probe_neighbor_frame')
        fn.__globals__['v11_probe_neighbor_frame']=lambda *args: n1 if args[1] < 22 else n2
        fn.__globals__['cv2']=type('C',(),{'CAP_PROP_POS_FRAMES':1})
        try:
            result=fn(cap,22,30,None,'右手側',middle)
            self.assertEqual(result['時間重查判定'],'疑似孤立關鍵點錯位')
            self.assertEqual(cap.pos,22)
        finally:
            if old is None:fn.__globals__.pop('v11_probe_neighbor_frame',None)
            else:fn.__globals__['v11_probe_neighbor_frame']=old

    def test_hand_pose_difference_does_not_reject(self):
        qc=namespace['v11_quality_from_snapshot'](snapshot(),'右手側',{'右手Hand用於計分':True,'右手Pose2D與Hand2D差異':80})
        self.assertEqual(qc['品質狀態'],'通過')

    def test_rejected_record_emits_snapshot_cols(self):
        qc=namespace['v11_quality_from_snapshot'](snapshot(wrist_vis=.29),'右手側',{'右手Hand用於計分':False})
        self.assertIn('品質原因',qc)
        self.assertGreater(qc['品質軟訊號數'],0)


if __name__=='__main__':unittest.main(verbosity=2)
