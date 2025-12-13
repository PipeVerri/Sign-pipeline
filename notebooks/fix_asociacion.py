import os
import json
import cv2
from pathlib import Path

ROOT_DIR = Path("../")
PATH = Path("0-AsociacionCivil/")

files = [f for f in os.listdir(ROOT_DIR / "data" / "processed" / PATH) if f.endswith(".json")]

for f in files:
    with open(ROOT_DIR / "data" / "processed" / PATH / f,  "r") as bb_f:
        bounding_boxes = json.load(bb_f)

    cap = cv2.VideoCapture(ROOT_DIR / "data" / "raw" / PATH / f.replace(".json", ".mp4"))
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    for bb in bounding_boxes:
        bb["timestamp"] = bb["timestamp"] / fps_original

    with open(ROOT_DIR / "data" / "processed" / PATH / f, "w") as bb_f:
        bb_f.write(json.dumps(bounding_boxes))