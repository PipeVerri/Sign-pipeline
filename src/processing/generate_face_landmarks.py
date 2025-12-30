# The previous version of this script didnt generate the face landmarks, this script is only for adding them back

import os, sys, contextlib
import logging

os.environ['GLOG_minloglevel'] = '3'
logging.getLogger('mediapipe').setLevel(logging.ERROR)
@contextlib.contextmanager
def mute_stderr_fd():
    fd = sys.stderr.fileno()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(fd)
    try:
        os.dup2(devnull, fd)   # redirect fd 2 -> /dev/null
        yield
    finally:
        os.dup2(saved, fd)     # restore
        os.close(saved)
        os.close(devnull)

with mute_stderr_fd():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

from pathlib import Path
import cv2
from sortedcontainers import SortedDict
from src.utils.video import read_cap_segments
from typing import Any
import numpy as np
from src.shared.lm_processing.landmarks import Landmarks, nn_parser, make_hip_centric
from src.shared.utils.mediapipe import mp_to_arr
import json
import h5py
from pathos.multiprocessing import ProcessPool
from tqdm import tqdm
from dataclasses import dataclass, field

ROOT_DIR = Path(__file__).resolve().parents[2]

face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=ROOT_DIR / "models/mediapipe/face_landmarker.task"
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)

MAX_CLIP_FRAME_SEPARATION = 1
BOUNDING_BOX_PADDING = 0.2
FPS = 6
MIN_CLIP_DURATION = 6 * FPS
MOVING_THRESHOLD = 0.25

class PersonResults:
    def __init__(self, id):
        self.id = id
        self.clips = []

    @dataclass
    class Clip:
        start: float
        end: float
        boxes: SortedDict[float, Any] = field(default_factory=SortedDict)
        max_box_size: dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})

    def add_bounding_box_frame(self, timestamp, bounding_box):
        x_size = bounding_box[2] - bounding_box[0]
        y_size = bounding_box[3] - bounding_box[1]
        if len(self.clips) > 0 and self.clips[-1].end + MAX_CLIP_FRAME_SEPARATION > timestamp:
            self.clips[-1].boxes[timestamp] = bounding_box
            self.clips[-1].end = timestamp
            self.clips[-1].max_box_size["x"] = max(self.clips[-1].max_box_size["x"], x_size)
            self.clips[-1].max_box_size["y"] = max(self.clips[-1].max_box_size["y"], y_size)
        else:
            to_add = PersonResults.Clip(start=timestamp, end=timestamp, boxes=SortedDict({timestamp: bounding_box}), max_box_size={"x": x_size, "y": y_size})
            self.clips.append(to_add)

def process_folder(PATH):
    os.makedirs(ROOT_DIR / "data" / "processed" / "landmarks" / PATH.name / "unlabeled", exist_ok=True)
    def process_file(f, unlabeled=False):
        adjusted_path = (PATH.name + "/unlabeled") if unlabeled else PATH.name
        try:
            with h5py.File(ROOT_DIR / "data" / "processed" / "landmarks" / adjusted_path / (f.replace(".mp4", ".h5")), "r") as output_f:
                if output_f.attrs.get("done", False) != 2: # Im gonna temporarily use 2 as a "done" marker for now
                    return
        except OSError:
            pass

        cap = cv2.VideoCapture(PATH / ("unlabeled" if unlabeled else "video") / f)

        with open(ROOT_DIR / "data" / "processed" / "bounding_boxes" / adjusted_path / f.replace(".mp4", ".json"), "r") as bb_file:
            bounding_boxes = json.load(bb_file)

        # Start by creating the clips for each person
        people = {}
        for entry in bounding_boxes:
            for person in entry["boxes"].keys():
                if person not in people:
                    people[person] = PersonResults(person)
                people[person].add_bounding_box_frame(entry["timestamp"], entry["boxes"][person])

        with h5py.File(ROOT_DIR / "data" / "processed" / "landmarks" / adjusted_path / f.replace(".mp4", ".h5"), "r+") as output_f: # Read and write, file must exist
            # Now process each person's clip
            for person in people.values():
                person_group = output_f.create_group(f"person_{person.id}")
                for clip_index, clip in enumerate(person.clips):
                    lm = Landmarks(max_frames_interpolation=12)
                    with vision.FaceLandmarker.create_from_options(face_options) as face_landmarker: # Reset the model for each clip
                        # Check at the start if the clip exists(to see if we should even bother processing it or not)
                        if not clip_index in person_group:
                            continue

                        for frame, timestamp in read_cap_segments(cap, fps=FPS, start=clip.start, end=clip.end):
                            # Start by getting the current frame's bounding box
                            bounding_box_idx = clip.boxes.bisect_left(timestamp)
                            bounding_box_idx = max(0, bounding_box_idx - 1) # If its too early, use the first bounding box
                            bounding_box = clip.boxes.peekitem(bounding_box_idx)[1]
                            # Get the bounding box's center
                            x_center = bounding_box[0] + (bounding_box[2] - bounding_box[0]) / 2
                            y_center = bounding_box[1] + (bounding_box[3] - bounding_box[1]) / 2
                            # Calculate the update distance with paddings
                            x_distance_center = (clip.max_box_size["x"] * (1 + BOUNDING_BOX_PADDING)) / 2
                            y_distance_center = (clip.max_box_size["y"] * (1 + BOUNDING_BOX_PADDING)) / 2
                            # Calculate the corners
                            x_start = max(0, x_center - x_distance_center)
                            x_end = x_center + x_distance_center
                            y_start = max(0, y_center - y_distance_center)
                            y_end = y_center + y_distance_center
                            # Crop the image and make it contiguous in memory
                            frame = frame[int(y_start):int(y_end), int(x_start):int(x_end)]
                            frame = np.ascontiguousarray(frame)
                            # Process the image
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                            face_result = face_landmarker.detect_for_video(mp_image, int(timestamp * 1000))

                            # Parse it
                            pose_landmarks = np.zeros((32, 3)) # Instead of passing a None pose, I will pass a non-representative one so that I can use the Landmarks class without modifications
                            face_landmarks = face_result.face_landmarks[0] if len(face_result.face_landmarks) > 0 else None

                            # Add it to Landmarks
                            lm.add(pose_landmarks, None, None, face_landmarks)

                            # No need for static image checks

                            # Show it
                            #cv2.imshow(f, frame)
                            #cv2.waitKey(1)

                        # Now go over each frame and append the face to the file's results
                        idx = 0
                        dataset = person_group[f"{clip_index}"]["landmarks"]
                        for _, _, _, face, frame_num in lm.get_landmarks(continuous=False, return_frame_number=True):
                            dataset[idx] = np.concatenate((dataset[idx], face.flatten()))
                            idx += 1

            # Set the new done status
            output_f.attrs["done"] = 2
        cap.release()

    files = sorted(os.listdir(PATH / "video"))
    unlabeled_files = sorted(os.listdir(PATH / "unlabeled"))

    def process_labeled(f):
        process_file(f, unlabeled=False)
    def process_unlabeled(f):
        process_file(f, unlabeled=True)

    pool = ProcessPool(nodes=20)
    for _ in tqdm(pool.imap(process_labeled, files),
                  total=len(files),
                  file=sys.stdout):
        pass

    for _ in tqdm(pool.imap(process_unlabeled, unlabeled_files),
                  total=len(unlabeled_files),
                  file=sys.stdout):
        pass

if __name__ == "__main__":
    for folder in ["1-videolibros_private", "2-videolibros_public", "3-CNSordos", "4-Locufre"]:
        process_folder(ROOT_DIR / "data" / "raw" / folder)
    cv2.destroyAllWindows()