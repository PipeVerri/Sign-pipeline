from ultralytics import YOLO
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
root_dir = Path(__file__).parent.parent.parent.parent.resolve()

VIDEOS_DIR = root_dir / "data" / "raw" / "0-AsociacionCivil"
UNLABELED = False

out_dir = root_dir / "data" / "processed" / "bounding_boxes"
out_dir.mkdir(exist_ok=True, parents=True) # Also create the parent dirs

def process_video(name):
    subs = name.split(".")[0] + ".vtt"
    with open(VIDEOS_DIR / subs, "r") as f:
        print(f.read())

process_video("0-0.mp4")