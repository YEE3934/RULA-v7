"""Optional: download the official MediaPipe Hand Landmarker model.
Run from the repository root: python scripts/download_hand_model.py
No extra pip dependencies required.
"""
from pathlib import Path
from urllib.request import urlopen

URL=('https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
     'hand_landmarker/float16/latest/hand_landmarker.task')
root=Path(__file__).resolve().parents[1]
out=root/'models'/'hand_landmarker.task'
out.parent.mkdir(parents=True,exist_ok=True)
if out.exists() and out.stat().st_size>1_000_000:
    print('Model is already present:',out)
else:
    with urlopen(URL,timeout=50) as response, open(out.with_suffix('.partial'),'wb') as f:
        total=0
        while data:=response.read(128*1024):
            total+=len(data)
            if total>35_000_000:raise RuntimeError('Unexpected model size')
            f.write(data)
    part=out.with_suffix('.partial')
    if part.stat().st_size<1_000_000:raise RuntimeError('Downloaded model is incomplete')
    part.replace(out)
    print('Model downloaded:',out)
