"""Concrete OpenCV decoding and mapping test for optional Front camera."""
import ast
import os
import tempfile
from pathlib import Path
import cv2
import numpy as np

source=(Path(__file__).resolve().parents[1]/'222.py').read_text('utf-8')
tree=ast.parse(source)
funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
needed=['v9_frame_meta','v9_common_window','v9_frame_index','v9_camera_seconds','v9_make_sample_times','v9_sample_row']
ns={'np':np}
exec(compile(ast.Module(body=[funcs[n] for n in needed],type_ignores=[]),'<video>', 'exec'),ns)

with tempfile.TemporaryDirectory() as folder:
    specs={'Right':(20,80),'Left':(15,60),'Front':(10,50)}
    paths={}
    for cam,(fps,count) in specs.items():
        path=os.path.join(folder,cam+'.avi')
        writer=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'MJPG'),fps,(160,120))
        assert writer.isOpened(), 'MJPG writer not available'
        for i in range(count):
            img=np.full((120,160,3),i%200,dtype=np.uint8)
            cv2.putText(img,str(i),(15,55),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            writer.write(img)
        writer.release()
        paths[cam]=path
    for cams in (['Right'],['Left'],['Right','Left'],['Front','Right'],['Front','Left'],['Front','Right','Left']):
        caps={cam:cv2.VideoCapture(paths[cam]) for cam in cams}
        assert all(c.isOpened() for c in caps.values())
        meta={cam:ns['v9_frame_meta'](cap.get(cv2.CAP_PROP_FPS),cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cam,cap in caps.items()}
        anchors={cam:0.2*j for j,cam in enumerate(cams)}
        times,start,end=ns['v9_make_sample_times'](9,meta,anchors,cams)
        for i,t in enumerate(times):
            row=ns['v9_sample_row'](t,meta,anchors,cams,i+1,start)
            for cam,cap in caps.items():
                idx=row[cam+'影格索引']
                cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
                success,img=cap.read()
                assert success and img is not None,(cams,cam,i,t,idx)
            assert ('Front影格索引' in row) == ('Front' in cams)
        for cap in caps.values():cap.release()
        print('PASS synthetic video frames:',','.join(cams))
