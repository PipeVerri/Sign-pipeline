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
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)
        os.close(devnull)


with mute_stderr_fd():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

from pathlib import Path
import cv2
from sortedcontainers import SortedDict
from typing import Any
import numpy as np
from src.shared.utils.mediapipe import mp_to_arr
import json
import h5py
from pathos.multiprocessing import ProcessPool
from tqdm import tqdm
from dataclasses import dataclass, field

ROOT_DIR = Path(__file__).resolve().parents[2]

# Configuration constants
MAX_CLIP_FRAME_SEPARATION = 1
BOUNDING_BOX_PADDING = 0.2
FPS = 6
MIN_CLIP_DURATION = 6 * FPS
MOVING_THRESHOLD = 0.25

# Landmark dimensions
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
FACE_LANDMARKS = 478
LANDMARK_DIMS = 3  # x, y, z coordinates


def create_pose_options():
    return vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=ROOT_DIR / "models/mediapipe/pose_landmarker_heavy.task"
        ),
        running_mode=vision.RunningMode.VIDEO,
    )


def create_hand_options():
    return vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=ROOT_DIR / "models/mediapipe/hand_landmarker.task"
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2
    )


def create_face_options():
    return vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=ROOT_DIR / "models/mediapipe/face_landmarker.task"
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )


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
            to_add = PersonResults.Clip(
                start=timestamp,
                end=timestamp,
                boxes=SortedDict({timestamp: bounding_box}),
                max_box_size={"x": x_size, "y": y_size}
            )
            self.clips.append(to_add)


def check_if_already_processed(output_path):
    """Check if file has already been processed."""
    try:
        with h5py.File(output_path, "r") as f:
            return f.attrs.get("done", False)
    except OSError:
        return False


def load_bounding_boxes(bb_path):
    """Load bounding box data from JSON file."""
    with open(bb_path, "r") as f:
        return json.load(f)


def create_person_clips(bounding_boxes):
    """Create PersonResults clips from bounding box data."""
    people = {}
    for entry in bounding_boxes:
        for person_id in entry["boxes"].keys():
            if person_id not in people:
                people[person_id] = PersonResults(person_id)
            people[person_id].add_bounding_box_frame(entry["timestamp"], entry["boxes"][person_id])
    return people


def calculate_crop_region(bounding_box, clip_max_box_size):
    """Calculate the crop region for a frame based on bounding box."""
    x_center = bounding_box[0] + (bounding_box[2] - bounding_box[0]) / 2
    y_center = bounding_box[1] + (bounding_box[3] - bounding_box[1]) / 2

    x_distance_center = (clip_max_box_size["x"] * (1 + BOUNDING_BOX_PADDING)) / 2
    y_distance_center = (clip_max_box_size["y"] * (1 + BOUNDING_BOX_PADDING)) / 2

    x_start = max(0, x_center - x_distance_center)
    x_end = x_center + x_distance_center
    y_start = max(0, y_center - y_distance_center)
    y_end = y_center + y_distance_center

    return int(y_start), int(y_end), int(x_start), int(x_end)


def crop_and_prepare_frame(frame, crop_coords):
    """Crop frame and make it contiguous in memory."""
    y_start, y_end, x_start, x_end = crop_coords
    cropped = frame[y_start:y_end, x_start:x_end]
    return np.ascontiguousarray(cropped)


def detect_landmarks(frame, timestamp, pose_landmarker, hand_landmarker, face_landmarker):
    """Run all landmark detection on a frame."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    pose_result = pose_landmarker.detect_for_video(mp_image, int(timestamp * 1000))
    hand_result = hand_landmarker.detect_for_video(mp_image, int(timestamp * 1000))
    face_result = face_landmarker.detect_for_video(mp_image, int(timestamp * 1000))

    return pose_result, hand_result, face_result


def extract_pose_landmarks(pose_result):
    """Extract pose landmarks from detection result."""
    return pose_result.pose_landmarks[0] if len(pose_result.pose_landmarks) > 0 else None


def extract_face_landmarks(face_result):
    """Extract face landmarks from detection result."""
    return face_result.face_landmarks[0] if len(face_result.face_landmarks) > 0 else None


def extract_hand_landmarks(hand_result):
    """Extract left and right hand landmarks from detection result."""
    left_hand = None
    right_hand = None

    for idx in range(len(hand_result.hand_landmarks)):
        if idx < len(hand_result.handedness):
            hand_category = hand_result.handedness[idx][0].category_name
            if hand_category == "Right":
                right_hand = hand_result.hand_landmarks[idx]
            else:
                left_hand = hand_result.hand_landmarks[idx]

    return left_hand, right_hand


def landmarks_to_array(landmarks, expected_count):
    """Convert MediaPipe landmarks to numpy array, or NaN array if None."""
    if landmarks is None:
        return np.full((expected_count, LANDMARK_DIMS), np.nan)
    return mp_to_arr(landmarks)


def compute_pose_motion_features(pose_landmarks):
    """Compute normalized arm vectors for motion detection."""
    if pose_landmarks is None:
        return None

    pose_arr = mp_to_arr(pose_landmarks)
    # Assuming make_hip_centric uses landmarks, we'll do basic normalization
    # If you need the exact make_hip_centric behavior, you can add it back
    pose_arr = pose_arr[12:23, :]
    norms = np.linalg.norm(pose_arr, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return pose_arr / norms


def check_motion_status(pose_landmarks, last_position, checked_frames, max_accel):
    """Check if the person is moving or static."""
    if pose_landmarks is None:
        return last_position, checked_frames, max_accel, 0

    pose_features = compute_pose_motion_features(pose_landmarks)

    if pose_features is None:
        return last_position, checked_frames, max_accel, 0

    if last_position is not None:
        change = np.linalg.norm(pose_features - last_position)
        max_accel = max(max_accel, change)

        if checked_frames >= MIN_CLIP_DURATION:
            if max_accel < MOVING_THRESHOLD:
                return pose_features.copy(), checked_frames + 1, max_accel, 2  # Static
            else:
                return pose_features.copy(), checked_frames + 1, max_accel, 1  # Moving

    return pose_features.copy(), checked_frames + 1, max_accel, 0  # Keep checking


def should_discard_clip(static_status, checked_frames):
    """Determine if a clip should be discarded."""
    return static_status == 2 or checked_frames < FPS


def prepare_raw_landmarks(pose_lm_list, left_lm_list, right_lm_list, face_lm_list):
    """Convert landmark lists to numpy arrays with NaN for missing data."""
    pose_arrays = [landmarks_to_array(lm, POSE_LANDMARKS) for lm in pose_lm_list]
    left_arrays = [landmarks_to_array(lm, HAND_LANDMARKS) for lm in left_lm_list]
    right_arrays = [landmarks_to_array(lm, HAND_LANDMARKS) for lm in right_lm_list]
    face_arrays = [landmarks_to_array(lm, FACE_LANDMARKS) for lm in face_lm_list]

    return {
        'pose': np.array(pose_arrays),
        'left_hand': np.array(left_arrays),
        'right_hand': np.array(right_arrays),
        'face': np.array(face_arrays)
    }


def save_clip_to_h5(person_group, clip_index, clip, landmark_data, timestamps):
    """Save a single clip's data to HDF5."""
    clip_group = person_group.create_group(f"{clip_index}")
    clip_group.attrs["start"] = clip.start
    clip_group.attrs["end"] = clip.end

    # Save each landmark type separately
    clip_group.create_dataset("pose_landmarks", data=landmark_data['pose'])
    clip_group.create_dataset("left_hand_landmarks", data=landmark_data['left_hand'])
    clip_group.create_dataset("right_hand_landmarks", data=landmark_data['right_hand'])
    clip_group.create_dataset("face_landmarks", data=landmark_data['face'])
    clip_group.create_dataset("timestamps", data=timestamps)


def read_video_once_for_all_clips(cap, clips, fps):
    """
    Read video once sequentially and yield frames for each clip.
    Much faster than seeking to each clip start.
    """
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    skip_rate = int(round(fps_original / fps))

    # Sort clips by start time
    sorted_clips = sorted(clips, key=lambda c: c.start)

    # Create a mapping of which frames belong to which clips
    clip_frame_ranges = []
    for clip in sorted_clips:
        start_frame = int(clip.start * fps_original)
        end_frame = int(clip.end * fps_original)
        clip_frame_ranges.append((start_frame, end_frame, clip))

    current_clip_idx = 0
    frame_count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Seek only ONCE at the beginning

    while cap.isOpened() and current_clip_idx < len(clip_frame_ranges):
        ret, frame = cap.read()
        if not ret:
            break

        # Check if we're in the current clip's range
        start_frame, end_frame, clip = clip_frame_ranges[current_clip_idx]

        if frame_count < start_frame:
            # Haven't reached this clip yet, skip if we can
            if frame_count % skip_rate == 0:
                # Only process frames at our target fps to save time
                pass
            frame_count += 1
            continue

        if frame_count >= end_frame:
            # Finished this clip, move to next
            current_clip_idx += 1
            frame_count += 1
            continue

        # We're in the clip range
        if frame_count % skip_rate == 0:
            timestamp = frame_count / fps_original
            yield clip, frame, timestamp

        frame_count += 1


def process_person_clips_optimized(person, cap, output_f):
    """Process all clips for a single person using sequential reading."""
    person_group = output_f.create_group(f"person_{person.id}")

    pose_options = create_pose_options()
    hand_options = create_hand_options()
    face_options = create_face_options()

    # Create landmark collectors for each clip
    clip_data = []
    for clip_index, clip in enumerate(person.clips):
        clip_data.append({
            'index': clip_index,
            'clip': clip,
            'pose_lm': [],
            'left_lm': [],
            'right_lm': [],
            'face_lm': [],
            'timestamps': [],
            'static_status': 0,
            'last_position': None,
            'checked_frames': 0,
            'max_accel': 0,
        })

    with vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
            vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
            vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:

        # Read video once and process all clips
        for clip_obj, frame, timestamp in read_video_once_for_all_clips(cap, person.clips, FPS):
            # Find which clip_data entry this corresponds to
            clip_entry = None
            for cd in clip_data:
                if cd['clip'] is clip_obj:
                    clip_entry = cd
                    break

            if clip_entry is None or clip_entry['static_status'] == 2:
                continue  # Skip if clip already marked as static

            clip = clip_entry['clip']

            # Get bounding box for this timestamp
            bounding_box_idx = clip.boxes.bisect_left(timestamp)
            bounding_box_idx = max(0, bounding_box_idx - 1)
            bounding_box = clip.boxes.peekitem(bounding_box_idx)[1]

            # Crop frame
            crop_coords = calculate_crop_region(bounding_box, clip.max_box_size)
            cropped_frame = crop_and_prepare_frame(frame, crop_coords)

            # Detect landmarks
            pose_result, hand_result, face_result = detect_landmarks(
                cropped_frame, timestamp, pose_landmarker, hand_landmarker, face_landmarker
            )

            # Extract individual landmarks
            pose_landmarks = extract_pose_landmarks(pose_result)
            face_landmarks = extract_face_landmarks(face_result)
            left_hand, right_hand = extract_hand_landmarks(hand_result)

            # Store raw landmarks
            clip_entry['pose_lm'].append(pose_landmarks)
            clip_entry['left_lm'].append(left_hand)
            clip_entry['right_lm'].append(right_hand)
            clip_entry['face_lm'].append(face_landmarks)
            clip_entry['timestamps'].append(timestamp)

            # Check for static content
            if clip_entry['static_status'] == 0:
                clip_entry['last_position'], clip_entry['checked_frames'], clip_entry['max_accel'], clip_entry[
                    'static_status'] = \
                    check_motion_status(
                        pose_landmarks,
                        clip_entry['last_position'],
                        clip_entry['checked_frames'],
                        clip_entry['max_accel']
                    )

        # Now save all clips that passed the motion check
        for cd in clip_data:
            if should_discard_clip(cd['static_status'], cd['checked_frames']):
                continue

            if len(cd['timestamps']) > 0:
                landmark_data = prepare_raw_landmarks(
                    cd['pose_lm'],
                    cd['left_lm'],
                    cd['right_lm'],
                    cd['face_lm']
                )
                save_clip_to_h5(
                    person_group,
                    cd['index'],
                    cd['clip'],
                    landmark_data,
                    np.array(cd['timestamps'])
                )


def process_file(PATH, filename, unlabeled=False):
    """Process a single video file."""
    adjusted_path = (PATH.name + "/unlabeled/video") if unlabeled else PATH.name
    output_path = ROOT_DIR / "data" / "processed" / "landmarks" / adjusted_path / filename.replace(".mp4", ".h5")

    # Check if already processed
    if check_if_already_processed(output_path):
        return

    # Open video
    video_path = PATH / ("unlabeled/video" if unlabeled else "video") / filename
    cap = cv2.VideoCapture(str(video_path))

    # Load bounding boxes
    bb_path = ROOT_DIR / "data" / "processed" / "bounding_boxes" / adjusted_path / filename.replace(".mp4", ".json")
    bounding_boxes = load_bounding_boxes(bb_path)

    # Create person clips
    people = create_person_clips(bounding_boxes)

    # Process each person
    with h5py.File(output_path, "w") as output_f:
        for person in people.values():
            process_person_clips_optimized(person, cap, output_f)

        output_f.attrs["fps"] = FPS
        output_f.attrs["done"] = True

    cap.release()
    print(f"Processed {filename}")


def process_folder(PATH):
    """Process all videos in a folder."""
    os.makedirs(ROOT_DIR / "data" / "processed" / "landmarks" / PATH.name / "unlabeled", exist_ok=True)

    files = sorted(os.listdir(PATH / "video"))
    unlabeled_files = sorted(os.listdir(PATH / "unlabeled" / "video")) if os.path.exists(PATH / "unlabeled") else []

    def process_labeled(f):
        process_file(PATH, f, unlabeled=False)

    def process_unlabeled(f):
        process_file(PATH, f, unlabeled=True)

    pool = ProcessPool(nodes=10)
    list(tqdm(pool.imap(process_labeled, files), total=len(files), file=sys.stdout))
    list(tqdm(pool.imap(process_unlabeled, unlabeled_files), total=len(unlabeled_files), file=sys.stdout))


def main():
    """Main entry point."""
    for folder in sorted(os.listdir(ROOT_DIR / "data" / "raw")):
        process_folder(ROOT_DIR / "data" / "raw" / folder)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # To profile, run: python -m cProfile -o profile.stats script.py
    # Then analyze with: python -m pstats profile.stats
    main()