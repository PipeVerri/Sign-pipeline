import json
import cv2
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[2]

def get_scale_and_size(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    if orig_h <= 0:
        raise RuntimeError(f"Video {video_path} reports invalid height {orig_h}")
    scale = orig_h / 144.0
    return scale, orig_w, orig_h

def scale_and_clamp_box(box, scale, max_w, max_h):
    # box is [x0, y0, x1, y1] in the 144p coordinates
    x0 = int(round(box[0] * scale))
    y0 = int(round(box[1] * scale))
    x1 = int(round(box[2] * scale))
    y1 = int(round(box[3] * scale))
    # clamp to image bounds
    x0 = max(0, min(x0, max_w - 1))
    x1 = max(0, min(x1, max_w - 1))
    y0 = max(0, min(y0, max_h - 1))
    y1 = max(0, min(y1, max_h - 1))
    # ensure ordering
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]

def resize_file(file_path: Path, scale: float, video_w: int, video_h: int):
    print(f"Resizing {file_path}")
    with open(file_path, "r") as f:
        bounding_boxes = json.load(f)

    for frame in bounding_boxes:
        boxes = frame.get("boxes", {})
        # keys may be strings; iterate values
        for key, box in boxes.items():
            if not box or len(box) < 4:
                continue
            boxes[key] = scale_and_clamp_box(box[:4], scale, video_w, video_h)
            # if boxes had more attributes (score, class) adapt accordingly

    # write back safely
    with open(file_path, "w") as f:
        json.dump(bounding_boxes, f, indent=2)

def process_folder(video_path: Path, bb_path: Path, suffix=".mp4"):
    errors = []
    for video in sorted(video_path.glob(f"*{suffix}")):
        try:
            scale, w, h = get_scale_and_size(video)
            json_path = bb_path / (video.name.replace(suffix, ".json"))
            if not json_path.exists():
                raise FileNotFoundError(f"{json_path} not found")
            resize_file(json_path, scale, w, h)
        except Exception as e:
            errors.append((video.name, str(e)))
    return errors

def process_channel(name):
    video_path = ROOT_DIR / "data" / "raw" / name
    bb_path = ROOT_DIR / "data" / "processed" / "bounding_boxes" / name
    errs = process_folder(video_path / "video", bb_path)
    if (video_path / "unlabeled").exists():
        errs += process_folder(video_path / "unlabeled" / "video", bb_path / "unlabeled")
    return errs

if __name__ == "__main__":
    all_errors = []
    channels = sorted(os.listdir(ROOT_DIR / "data" / "raw"))
    for channel in channels:
        all_errors += process_channel(channel)
    for v, e in all_errors:
        print(f"Error processing {v}: {e}")
