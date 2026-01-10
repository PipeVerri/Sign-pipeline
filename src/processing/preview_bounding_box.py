import cv2
import json
import pathlib

from src.shared.utils.video import frame_reader

ROOT_DIR = pathlib.Path(__file__).parents[2]
VIDEO_FILE = ROOT_DIR / "data" / "raw" / "0-AsociacionCivil" / "video" / "0-10.mp4"
JSON_FILE = ROOT_DIR / "data" / "processed" / "bounding_boxes" / "0-AsociacionCivil" / "0-10.json"

cap = cv2.VideoCapture(VIDEO_FILE)
with open(JSON_FILE, "r") as bb_f:
    bounding_boxes = json.load(bb_f)

current_bb_index = 0
for frame, timestamp in frame_reader(cap, fps=6, return_timestamp=True):
    # Update index: find the most recent bounding box for this timestamp
    while (current_bb_index + 1 < len(bounding_boxes) and
           timestamp >= bounding_boxes[current_bb_index + 1]["timestamp"]):
        current_bb_index += 1

    # Draw boxes if we have valid data
    if timestamp >= bounding_boxes[current_bb_index]["timestamp"]:
        for boxes in bounding_boxes[current_bb_index]["boxes"].values():
            cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (255, 0, 0), 2)

    cv2.imshow("Preview", frame)
    cv2.waitKey(1)