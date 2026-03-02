from utils.args import parse_args
import os
import json
import threading
from ultralytics import YOLO
import queue
from utils.video import frames_for_segment
from tqdm import tqdm

default_config = { "batch_size": 32, "batch_queue": 32 }
working, config = parse_args()
step_config = config["options"]["bounding_boxes"] | default_config

def process_folder(source_path):
    def process_video(video_path):
        output_dir = working / "processed" / "bounding_boxes" / video_path.name
        os.makedirs(output_dir, exist_ok=True)

        # Create a YOLO model
        model = YOLO(working / step_config["model_path"])
        batch_queue = queue.Queue(maxsize=step_config["batch_queue"])

        def producer():
            current_batch_frames = []
            current_batch_ts = []
            for frame, timestamp in frames_for_segment(video_path):
                current_batch_frames.append(frame)
                current_batch_ts.append(timestamp)
                if len(current_batch_frames) == step_config["batch_size"]:
                    batch_queue.put((current_batch_frames, current_batch_ts))
                    current_batch_frames = []
                    current_batch_ts = []
            if len(current_batch_frames) != 0:
                batch_queue.put((current_batch_frames, current_batch_ts))

        def consumer():
            video_output = []
            while True:
                item = batch_queue.get()
                if item is None:
                    break
                results = model.track(item[0], persist=True, classes=[0], verbose=False, batch=len(item[0]))
                for idx, r in enumerate(results):
                    if r.boxes is not None and r.boxes.id is not None:
                        ids = r.boxes.id.cpu().tolist()
                        xyxy = r.boxes.xyxy.cpu().tolist()
                        boxes_dict = {}
                        for track_id, box in zip(ids, xyxy):
                            boxes_dict[track_id] = box
                        video_output.append({
                            "timestamp": item[1][idx],
                            "boxes": boxes_dict
                        })
            out_file = output_dir / f"{video_path.stem}.json"
            json.dump(video_output, open(out_file, "w"))

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join()
        batch_queue.put(None)
        t2.join()

    files = os.listdir(source_path)
    for f in tqdm(files):
        process_video(source_path / f)

def process_source(source_name):
    process_folder(working / "videos" / source_name / "labeled/video")
    process_folder(working / "videos" / source_name / "unlabeled/video")

for source in config["sources"]:
    process_source(source["name"])