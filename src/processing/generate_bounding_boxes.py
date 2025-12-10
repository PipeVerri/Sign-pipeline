import os

from torch.backends.mkl import verbose
from ultralytics import YOLO
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import webvtt
import cv2
import json

root_dir = Path(__file__).parent.parent.parent.resolve()
MAX_WORKERS = 24

def read_cap_segments(cap, start, end, fps=6):
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    skip_rate = int(round(fps_original/fps))
    start_fps = fps_original * start
    end_fps = fps_original * end
    frame_count = 0

    while cap.isOpened():
        if frame_count >= end_fps - start_fps:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_rate == 0:
            yield frame, start_fps + frame_count
        frame_count += 1

def process_folder(PATH):
    OUTPUT_PATH = root_dir / "data" / "processed" / PATH.name
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH / "unlabeled", exist_ok=True)

    def process_video(f):
        # Create a YOLO model
        model = YOLO(root_dir / "models" / "yolo" / "yolo11m.pt")
        # Get the video length
        cap = cv2.VideoCapture(PATH / "video" / f)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        length = fps_length / fps
        # Now process each caption clip
        video_output = []
        captions = webvtt.read(PATH / "subs" / f.replace(".mp4", ".vtt"))
        for idx, caption in enumerate(captions):
            print(f"Processing clip {idx} of {len(captions)}")
            # Convert to seconds
            parse_str = lambda arr: [int(float(x)) for x in arr.split(":")]
            convert_seconds = lambda arr: arr[0] * (60 * 60) + arr[1] * 60 + arr[2]
            start_seconds = convert_seconds(parse_str(caption.start))
            end_seconds = convert_seconds(parse_str(caption.end))
            # Now add padding and check the bounds
            start_seconds = max(start_seconds - 1.5, 0)
            end_seconds = min(end_seconds + 1.5, length)
            # Now process the cap using those bounds
            clip_output = []
            for frame, timestamp in read_cap_segments(cap, start_seconds, end_seconds):
                results = model.track(frame, persist=True, classes=[0], verbose=False) # Persist accross frames, only track humans
                r = results[0] # Only passed 1 frame, only grabbing the first result
                if r.boxes is not None and r.boxes.id is not None:
                    ids = r.boxes.id.cpu().tolist()
                    xyxy = r.boxes.xyxy.cpu().tolist()

                    boxes_dict = {}
                    for track_id, box in zip(ids, xyxy):
                        boxes_dict[track_id] = box

                    clip_output.append({
                        "timestamp": timestamp,
                        "people": boxes_dict
                    })

            video_output.append(clip_output)
        json.dump(video_output, open(OUTPUT_PATH / f"{f.replace('.mp4', '.json')}", "w"))
        cap.release()

    files = os.listdir(PATH / "video")
    process_video(files[0])
    #with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    #    executor.map(process_video, files)

if __name__ == "__main__":
    folders = ["0-AsociacionCivil", "1-videolibros_private", "2-videolibros_public"]
    for folder in folders:
        process_folder(root_dir / "data" / "raw" / folder)
        break