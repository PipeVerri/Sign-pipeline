import os
import json
import cv2
from pathlib import Path

ROOT_DIR = Path("../")

def process_folder(VIDEOS_PATH, OUTPUT_PATH):
    files = [f for f in os.listdir(OUTPUT_PATH) if f.endswith(".json")]
    for f in files:
        with open(OUTPUT_PATH / f, "r") as bb_f:
            bounding_boxes = json.load(bb_f)

        cap = cv2.VideoCapture(VIDEOS_PATH / f.replace(".json", ".mp4"))
        fps_original = cap.get(cv2.CAP_PROP_FPS)
        for bb in bounding_boxes:
            bb["timestamp"] = bb["timestamp"] / fps_original

        with open(OUTPUT_PATH / f, "w") as bb_f:
            bb_f.write(json.dumps(bounding_boxes))

def process_both_folders(name):
    process_folder(ROOT_DIR / "data" / "raw" / name / "video", ROOT_DIR / "data" / "processed" / "bounding_boxes"/ name)
    process_folder(ROOT_DIR / "data" / "raw" / name / "unlabeled", ROOT_DIR / "data" / "processed" / "bounding_boxes" / name / "unlabeled")

process_both_folders("4-Locufre")