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
from sortedcontainers import SortedDict
import numpy as np
from utils.shared.utils.mediapipe import mp_to_arr
import json
import h5py
from pathos.multiprocessing import ProcessPool
from tqdm import tqdm
from dataclasses import dataclass, field
from utils.args import parse_args
from utils.gpu_reader import read_video_for_clips

ROOT_DIR = Path(__file__).resolve().parents[1]

FPS = 6
MAX_CLIP_FRAME_SEPARATION = 1
BOUNDING_BOX_PADDING = 0.2
MIN_CLIP_DURATION = 6 * FPS
MOVING_THRESHOLD = 0.25

FRAME_BATCH_SIZE = 80
WRITE_BUFFER_SIZE = 160

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
FACE_LANDMARKS = 478
LANDMARK_DIMS = 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class PersonResults:
    def __init__(self, id):
        self.id = id
        self.clips = []

    @dataclass
    class Clip:
        start: float
        end: float
        boxes: SortedDict = field(default_factory=SortedDict)
        max_box_size: dict = field(default_factory=lambda: {"x": 0, "y": 0})

    def add_bounding_box_frame(self, timestamp, bounding_box):
        x_size = bounding_box[2] - bounding_box[0]
        y_size = bounding_box[3] - bounding_box[1]
        if len(self.clips) > 0 and self.clips[-1].end + MAX_CLIP_FRAME_SEPARATION > timestamp:
            self.clips[-1].boxes[timestamp] = bounding_box
            self.clips[-1].end = timestamp
            self.clips[-1].max_box_size["x"] = max(self.clips[-1].max_box_size["x"], x_size)
            self.clips[-1].max_box_size["y"] = max(self.clips[-1].max_box_size["y"], y_size)
        else:
            self.clips.append(PersonResults.Clip(
                start=timestamp,
                end=timestamp,
                boxes=SortedDict({timestamp: bounding_box}),
                max_box_size={"x": x_size, "y": y_size},
            ))


# ---------------------------------------------------------------------------
# MediaPipe option factories (GPU delegate)
# ---------------------------------------------------------------------------

def create_pose_options(model_path):
    return vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.GPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
    )


def create_hand_options(model_path):
    return vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.GPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )


def create_face_options(model_path):
    return vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.GPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def calculate_crop_region(bounding_box, clip_max_box_size):
    x_center = bounding_box[0] + (bounding_box[2] - bounding_box[0]) / 2
    y_center = bounding_box[1] + (bounding_box[3] - bounding_box[1]) / 2
    x_half = (clip_max_box_size["x"] * (1 + BOUNDING_BOX_PADDING)) / 2
    y_half = (clip_max_box_size["y"] * (1 + BOUNDING_BOX_PADDING)) / 2
    return (
        int(max(0, y_center - y_half)), int(y_center + y_half),
        int(max(0, x_center - x_half)), int(x_center + x_half),
    )


def crop_and_prepare_frame(frame, crop_coords):
    y0, y1, x0, x1 = crop_coords
    return np.ascontiguousarray(frame[y0:y1, x0:x1])


def extract_pose_landmarks(pose_result):
    return pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None


def extract_face_landmarks(face_result):
    return face_result.face_landmarks[0] if face_result.face_landmarks else None


def extract_hand_landmarks(hand_result):
    left_hand = right_hand = None
    for idx in range(len(hand_result.hand_landmarks)):
        if idx < len(hand_result.handedness):
            if hand_result.handedness[idx][0].category_name == "Right":
                right_hand = hand_result.hand_landmarks[idx]
            else:
                left_hand = hand_result.hand_landmarks[idx]
    return left_hand, right_hand


def landmarks_to_array(landmarks, expected_count):
    if landmarks is None:
        return np.full((expected_count, LANDMARK_DIMS), np.nan)
    return mp_to_arr(landmarks)


# ---------------------------------------------------------------------------
# Motion check (from profiling version — no hip-centering, safe norm)
# ---------------------------------------------------------------------------

def check_motion_status(pose_landmarks, last_position, checked_frames, max_accel):
    if pose_landmarks is None:
        return last_position, checked_frames, max_accel, 0

    pose_arr = mp_to_arr(pose_landmarks)[12:23, :]
    norms = np.linalg.norm(pose_arr, axis=1, keepdims=True)
    pose_features = pose_arr / np.where(norms == 0, 1, norms)

    if last_position is not None:
        change = np.linalg.norm(pose_features - last_position)
        max_accel = max(max_accel, change)
        if checked_frames >= MIN_CLIP_DURATION:
            status = 2 if max_accel < MOVING_THRESHOLD else 1
            return pose_features.copy(), checked_frames + 1, max_accel, status

    return pose_features.copy(), checked_frames + 1, max_accel, 0


def should_discard_clip(static_status, checked_frames):
    return static_status == 2 or checked_frames < FPS


# ---------------------------------------------------------------------------
# ClipWriter — chunked raw-landmark H5 writer
# ---------------------------------------------------------------------------

class ClipWriter:
    """Accumulates raw per-frame landmarks and flushes to resizable H5 datasets
    in chunks to avoid memory overflow on long clips."""

    def __init__(self, h5_group, clip, chunk_size=WRITE_BUFFER_SIZE):
        self.h5_group = h5_group
        self.clip = clip
        self.chunk_size = chunk_size

        self.pose_buf = []
        self.left_buf = []
        self.right_buf = []
        self.face_buf = []
        self.ts_buf = []

        self.static_status = 0
        self.last_position = None
        self.checked_frames = 0
        self.max_accel = 0

        self.datasets_created = False
        self.total_frames = 0

    def add_frame(self, pose_lm, left_lm, right_lm, face_lm, timestamp):
        self.pose_buf.append(landmarks_to_array(pose_lm, POSE_LANDMARKS))
        self.left_buf.append(landmarks_to_array(left_lm, HAND_LANDMARKS))
        self.right_buf.append(landmarks_to_array(right_lm, HAND_LANDMARKS))
        self.face_buf.append(landmarks_to_array(face_lm, FACE_LANDMARKS))
        self.ts_buf.append(timestamp)

        self.last_position, self.checked_frames, self.max_accel, self.static_status = \
            check_motion_status(pose_lm, self.last_position, self.checked_frames, self.max_accel)

        if len(self.pose_buf) >= self.chunk_size:
            self._flush()

    def _flush(self):
        if not self.pose_buf:
            return

        pose_arr = np.array(self.pose_buf)
        left_arr = np.array(self.left_buf)
        right_arr = np.array(self.right_buf)
        face_arr = np.array(self.face_buf)
        ts_arr = np.array(self.ts_buf)

        if not self.datasets_created:
            self.h5_group.create_dataset("pose_landmarks", data=pose_arr,
                                         maxshape=(None, POSE_LANDMARKS, LANDMARK_DIMS), chunks=True)
            self.h5_group.create_dataset("left_hand_landmarks", data=left_arr,
                                         maxshape=(None, HAND_LANDMARKS, LANDMARK_DIMS), chunks=True)
            self.h5_group.create_dataset("right_hand_landmarks", data=right_arr,
                                         maxshape=(None, HAND_LANDMARKS, LANDMARK_DIMS), chunks=True)
            self.h5_group.create_dataset("face_landmarks", data=face_arr,
                                         maxshape=(None, FACE_LANDMARKS, LANDMARK_DIMS), chunks=True)
            self.h5_group.create_dataset("timestamps", data=ts_arr,
                                         maxshape=(None,), chunks=True)
            self.datasets_created = True
        else:
            for name, arr in [
                ("pose_landmarks", pose_arr),
                ("left_hand_landmarks", left_arr),
                ("right_hand_landmarks", right_arr),
                ("face_landmarks", face_arr),
                ("timestamps", ts_arr),
            ]:
                ds = self.h5_group[name]
                old = ds.shape[0]
                new = old + len(arr)
                ds.resize(new, axis=0)
                ds[old:new] = arr

        self.total_frames += len(self.pose_buf)
        self.pose_buf.clear()
        self.left_buf.clear()
        self.right_buf.clear()
        self.face_buf.clear()
        self.ts_buf.clear()

    def finalize(self):
        """Flush remaining data, write attrs. Returns True if clip should be kept."""
        self._flush()
        if self.datasets_created:
            self.h5_group.attrs["start"] = self.clip.start
            self.h5_group.attrs["end"] = self.clip.end
        return not should_discard_clip(self.static_status, self.checked_frames)


# ---------------------------------------------------------------------------
# Per-person processing
# ---------------------------------------------------------------------------

def process_person_clips(person, video_path, person_group, pose_opts, hand_opts, face_opts, batch_size):
    """Read the video once for this person's clips, detect landmarks in batches,
    and stream results to HDF5 via ClipWriter."""

    clip_writers = {}
    for clip_index, clip in enumerate(person.clips):
        grp = person_group.create_group(f"{clip_index}")
        clip_writers[id(clip)] = (clip_index, ClipWriter(grp, clip))

    with vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm, \
         vision.HandLandmarker.create_from_options(hand_opts) as hand_lm, \
         vision.FaceLandmarker.create_from_options(face_opts) as face_lm:

        frame_batch = []
        ts_batch = []
        clip_batch = []

        def flush_batch():
            for frame, ts, clip_obj in zip(frame_batch, ts_batch, clip_batch):
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                p_res = pose_lm.detect_for_video(mp_image, int(ts * 1000))
                h_res = hand_lm.detect_for_video(mp_image, int(ts * 1000))
                f_res = face_lm.detect_for_video(mp_image, int(ts * 1000))

                _, writer = clip_writers[id(clip_obj)]
                writer.add_frame(
                    extract_pose_landmarks(p_res),
                    *extract_hand_landmarks(h_res),
                    extract_face_landmarks(f_res),
                    ts,
                )
            frame_batch.clear()
            ts_batch.clear()
            clip_batch.clear()

        for clip_obj, frame, ts in read_video_for_clips(video_path, person.clips, FPS):
            _, writer = clip_writers[id(clip_obj)]
            if writer.static_status == 2:
                continue

            bb_idx = max(0, clip_obj.boxes.bisect_left(ts) - 1)
            bounding_box = clip_obj.boxes.peekitem(bb_idx)[1]
            crop_coords = calculate_crop_region(bounding_box, clip_obj.max_box_size)
            frame_batch.append(crop_and_prepare_frame(frame, crop_coords))
            ts_batch.append(ts)
            clip_batch.append(clip_obj)

            if len(frame_batch) >= batch_size:
                flush_batch()

        if frame_batch:
            flush_batch()

    # Finalize: flush buffers, set attrs, delete discarded clip groups
    for clip_id, (clip_index, writer) in clip_writers.items():
        if not writer.finalize():
            del person_group[f"{clip_index}"]


# ---------------------------------------------------------------------------
# Pipeline-integrated process_video
# ---------------------------------------------------------------------------

def process_video(video_path, is_labeled, working, step_config):
    video_path = Path(video_path)
    working = Path(working)

    source_name = video_path.parents[2].name  # .../videos/{source}/labeled|unlabeled/video/
    bb_file = working / "processed" / "bounding_boxes" / video_path.name / f"{video_path.stem}.json"
    subtitle_file = video_path.parents[2] / "labeled" / "subtitles" / f"{video_path.stem}.vtt"
    temp_h5 = working / "processed" / "landmarks" / "tmp" / source_name / f"{video_path.stem}.h5"

    try:
        with h5py.File(temp_h5, "r") as f:
            if f.attrs.get("done", False):
                return
    except OSError:
        pass

    if not bb_file.exists():
        return

    with open(bb_file, "r") as f:
        bounding_boxes = json.load(f)

    people = {}
    for entry in bounding_boxes:
        for person_id, box in entry["boxes"].items():
            if person_id not in people:
                people[person_id] = PersonResults(person_id)
            people[person_id].add_bounding_box_frame(entry["timestamp"], box)

    subtitles = ""
    if is_labeled and subtitle_file.exists():
        subtitles = subtitle_file.read_text(encoding="utf-8")

    pose_opts = create_pose_options(ROOT_DIR / step_config["model_path_pose"])
    hand_opts = create_hand_options(ROOT_DIR / step_config["model_path_hand"])
    face_opts = create_face_options(ROOT_DIR / step_config["model_path_face"])
    batch_size = step_config.get("frame_batch_size", FRAME_BATCH_SIZE)

    os.makedirs(temp_h5.parent, exist_ok=True)

    with h5py.File(temp_h5, "w") as output_f:
        video_group = output_f.create_group(video_path.stem)
        video_group.attrs["video_id"] = video_path.stem
        video_group.attrs["labeled"] = is_labeled
        video_group.attrs["subtitles"] = subtitles

        for person in people.values():
            person_group = video_group.create_group(f"person_{person.id}")
            process_person_clips(
                person, video_path, person_group,
                pose_opts, hand_opts, face_opts, batch_size,
            )

        output_f.attrs["fps"] = FPS
        output_f.attrs["done"] = True


# ---------------------------------------------------------------------------
# Merge + source orchestration
# ---------------------------------------------------------------------------

def merge_temp_files(temp_dir, output_h5_path):
    temp_dir = Path(temp_dir)
    output_h5_path = Path(output_h5_path)

    with h5py.File(output_h5_path, "a") as out_f:
        for temp_h5 in temp_dir.glob("*.h5"):
            with h5py.File(temp_h5, "r") as src_f:
                for key in src_f.keys():
                    if key not in out_f:
                        h5py.copy(src_f[key], out_f, name=key)
        out_f.attrs["fps"] = FPS


def process_source(source, working, config, step_config):
    source_name = source["name"]
    working = Path(working)

    labeled_dir = working / "videos" / source_name / "labeled" / "video"
    unlabeled_dir = working / "videos" / source_name / "unlabeled" / "video"
    temp_dir = working / "processed" / "landmarks" / "tmp" / source_name
    output_dir = working / "processed" / "landmarks"

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    labeled_videos = sorted(labeled_dir.glob("*.mp4")) if labeled_dir.exists() else []
    unlabeled_videos = sorted(unlabeled_dir.glob("*.mp4")) if unlabeled_dir.exists() else []

    pool = ProcessPool(nodes=os.cpu_count())

    if labeled_videos:
        def process_labeled(video_path):
            process_video(video_path, is_labeled=True, working=working, step_config=step_config)

        for _ in tqdm(pool.imap(process_labeled, labeled_videos),
                      total=len(labeled_videos), desc=f"[{source_name}] labeled", file=sys.stdout):
            pass

        merge_temp_files(temp_dir, output_dir / f"{source_name}_labeled.h5")

    if unlabeled_videos:
        def process_unlabeled(video_path):
            process_video(video_path, is_labeled=False, working=working, step_config=step_config)

        for _ in tqdm(pool.imap(process_unlabeled, unlabeled_videos),
                      total=len(unlabeled_videos), desc=f"[{source_name}] unlabeled", file=sys.stdout):
            pass

        merge_temp_files(temp_dir, output_dir / f"{source_name}_unlabeled.h5")

    pool.close()
    pool.join()


if __name__ == "__main__":
    working, config = parse_args()
    step_config = {"frame_batch_size": FRAME_BATCH_SIZE} | config["options"]["landmarks"]

    for source in config["sources"]:
        process_source(source, working, config, step_config)
