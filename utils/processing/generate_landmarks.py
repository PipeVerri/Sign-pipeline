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
from video import read_cap_segments
from typing import Any
import numpy as np
from utils.shared.lm_processing.landmarks import Landmarks, nn_parser, make_hip_centric
from utils.shared.utils.mediapipe import mp_to_arr
import json
import h5py
from pathos.multiprocessing import ProcessPool
from tqdm import tqdm
from dataclasses import dataclass, field

ROOT_DIR = Path(__file__).resolve().parents[2]

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path= ROOT_DIR / "models/mediapipe/pose_landmarker_heavy.task"
    ),
    running_mode=vision.RunningMode.VIDEO,
)

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=ROOT_DIR / "models/mediapipe/hand_landmarker.task"
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

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
                if output_f.attrs.get("done", False):
                    return
        except OSError: # It will raise OSError if the file doesnt exist, which is fine
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

        with h5py.File(ROOT_DIR / "data" / "processed" / "landmarks" / adjusted_path / f.replace(".mp4", ".h5"), "w") as output_f:
            # Now process each person's clip
            for person in people.values():
                person_group = output_f.create_group(f"person_{person.id}")
                for clip_index, clip in enumerate(person.clips):
                    lm = Landmarks(max_frames_interpolation=12)
                    with vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
                        vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
                        vision.FaceLandmarker.create_from_options(face_options) as face_landmarker: # Reset the model for each clip

                        static_status = 0 # 0 not checked, 1 is moving, 2 is static
                        last_position = None
                        checked_frames = 0
                        max_accel = 0
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
                            pose_result = pose_landmarker.detect_for_video(mp_image, int(timestamp * 1000))
                            hand_result = hand_landmarker.detect_for_video(mp_image, int(timestamp * 1000))
                            face_result = face_landmarker.detect_for_video(mp_image, int(timestamp * 1000))

                            # Parse it
                            pose_landmarks = pose_result.pose_landmarks[0] if len(pose_result.pose_landmarks) > 0 else None
                            face_landmarks = face_result.face_landmarks[0] if len(face_result.face_landmarks) > 0 else None

                            left_hand_landmarks = None
                            right_hand_landmarks = None
                            for idx in range(len(hand_result.hand_landmarks)):
                                if idx < len(hand_result.handedness):
                                    hand_category = hand_result.handedness[idx][0].category_name
                                    if hand_category == "Right":
                                        right_hand_landmarks = hand_result.hand_landmarks[idx]
                                    else:
                                        left_hand_landmarks = hand_result.hand_landmarks[idx]

                            # Add it to Landmarks
                            lm.add(pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks)

                            # Check if this is just a static image and we should stop processing
                            if static_status == 0 and pose_landmarks is not None:
                                # Make it hip-based so that I dont interpret a moving image as a person
                                pose_arr = make_hip_centric(mp_to_arr(pose_landmarks))
                                # Now only take the arm vectors, and normalize it so I only store the direction data
                                pose_arr = pose_arr[12:23, :]
                                pose_arr = pose_arr / np.linalg.norm(pose_arr, axis=1, keepdims=True)
                                if last_position is not None:
                                    change = np.linalg.norm(pose_arr - last_position)
                                    max_accel = max(max_accel, change)
                                    if checked_frames >= MIN_CLIP_DURATION:
                                        if max_accel < MOVING_THRESHOLD:
                                            static_status = 2
                                            break
                                        else:
                                            static_status = 1
                                last_position = pose_arr.copy()
                                checked_frames += 1

                            # Show it
                            #cv2.imshow(f, frame)
                            #cv2.waitKey(1)

                        if static_status == 2 or checked_frames < FPS: # If it lasts less than a second or is static, discard it
                            #print("Discarded")
                            continue

                        # Now go over each landmark frame, pass it through nn_parser
                        parsed_landmarks = []
                        timestamps = []
                        for pose, left, right, face, frame_num in lm.get_landmarks(continuous=False, return_frame_number=True):
                            parsed = nn_parser(pose, left, right, face)
                            parsed_landmarks.append(parsed)
                            timestamps.append(clip.start + (frame_num / FPS))

                        # And save it
                        if len(parsed_landmarks) > 0:
                            clip_group = person_group.create_group(f"{clip_index}")
                            clip_group.attrs["start"] = clip.start
                            clip_group.attrs["end"] = clip.end
                            clip_group.create_dataset("landmarks", data=parsed_landmarks)
                            clip_group.create_dataset("timestamps", data=timestamps)
            output_f.attrs["fps"] = FPS
            output_f.attrs["done"] = True
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
    # TODO: Set back the original folders
    for folder in ["5-test_folder"]:
        process_folder(ROOT_DIR / "data" / "raw" / folder)
    cv2.destroyAllWindows()