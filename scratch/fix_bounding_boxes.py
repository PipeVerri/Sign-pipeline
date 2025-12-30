# Get all the video heights and put them in a file
import cv2
import json
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

def get_video_height(video_path: Path) -> int | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return height


def collect_video_heights(video_dir: Path) -> dict:
    results = {}

    for video_path in sorted(video_dir.iterdir()):
        if video_path.suffix.lower() not in VIDEO_EXTS:
            continue

        height = get_video_height(video_path)
        if height is not None and height > 0:
            results[video_path.name] = height
        else:
            print(f"⚠️ Could not read height for {video_path.name}")

    return results


if __name__ == "__main__":
    video_dir = Path("path/to/videos")   # <-- change this
    output_json = Path("video_heights.json")

    data = collect_video_heights(video_dir)

    with output_json.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved heights for {len(data)} videos to {output_json}")
