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
import subprocess

# Try to import GPU video decoder
try:
    import PyNvVideoCodec as nvc

    GPU_DECODE_AVAILABLE = True
except ImportError:
    GPU_DECODE_AVAILABLE = False
    print("WARNING: PyNvVideoCodec not available. Install with: pip install pynvvideocodec")
    print("Falling back to CPU decoding (slower)")

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
LANDMARK_DIMS = 3

# Optimized batch sizes for GPU pipeline
FRAME_BATCH_SIZE = 80  # Increased for GPU decoding
WRITE_BUFFER_SIZE = 160  # Larger buffer since we have more throughput
DECODE_BUFFER_SIZE = 100  # Pre-decode frames ahead
NUM_WORKERS = 8


class GPUVideoReader:
    """GPU-accelerated video reader using NVIDIA hardware decoder."""

    def __init__(self, video_path):
        self.video_path = str(video_path)
        self.use_gpu = GPU_DECODE_AVAILABLE

        if self.use_gpu:
            self._init_gpu_decoder()
        else:
            self._init_cpu_decoder()

    def _init_gpu_decoder(self):
        # Probe basic properties with OpenCV (fast)
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        cap.release()

        # Build ffmpeg command to use NVDEC (hwaccel cuda) and output raw RGB frames
        cmd = [
            "ffmpeg",
            "-nostdin", "-loglevel", "error",
            "-hwaccel", "cuda",  # use NVDEC
            "-i", str(self.video_path),
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo",
            "-"
        ]

        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 8)
        self.use_gpu = True
        self._ff_frame_bytes = self.width * self.height * 3
        self.current_frame = 0

    def _init_cpu_decoder(self):
        """Fallback to CPU decoder."""
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0

    def read(self):
        """Read next frame."""
        if self.use_gpu:
            return self._read_gpu()
        else:
            return self._read_cpu()

    def _read_gpu(self):
        """Read one RGB frame from ffmpeg stdout (decoded on GPU)."""
        if not hasattr(self, "proc"):
            return False, None

        raw = self.proc.stdout.read(self._ff_frame_bytes)
        if not raw or len(raw) < self._ff_frame_bytes:
            return False, None

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        self.current_frame += 1
        return True, frame

    def _read_cpu(self):
        """Read frame using CPU decoder."""
        ret, frame = self.cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame += 1
        return ret, frame

    def seek(self, frame_number):
        """Seek to specific frame."""
        if self.use_gpu:
            # GPU decoder seeks automatically in decode calls
            # For now, we use sequential reading which is faster anyway
            pass
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number

    def get_fps(self):
        return self.fps

    def get_total_frames(self):
        return self.total_frames

    def release(self):
        try:
            if getattr(self, "proc", None):
                self.proc.kill()
                self.proc.stdout.close()
                self.proc.stderr.close()
        except Exception:
            pass
        # also release CPU cap if present
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()


def create_pose_options():
    return vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(ROOT_DIR / "models/mediapipe/pose_landmarker_heavy.task"),
            delegate=mp.tasks.BaseOptions.Delegate.GPU
        ),
        running_mode=vision.RunningMode.VIDEO,
    )


def create_hand_options():
    return vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(ROOT_DIR / "models/mediapipe/hand_landmarker.task"),
            delegate=mp.tasks.BaseOptions.Delegate.GPU
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2
    )


def create_face_options():
    return vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(ROOT_DIR / "models/mediapipe/face_landmarker.task"),
            delegate=mp.tasks.BaseOptions.Delegate.GPU
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


def detect_landmarks_batch(frames, timestamps, pose_landmarker, hand_landmarker, face_landmarker):
    """Run landmark detection on a batch of frames for better GPU utilization."""
    pose_results = []
    hand_results = []
    face_results = []

    for frame, timestamp in zip(frames, timestamps):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        pose_result = pose_landmarker.detect_for_video(mp_image, int(timestamp * 1000))
        hand_result = hand_landmarker.detect_for_video(mp_image, int(timestamp * 1000))
        face_result = face_landmarker.detect_for_video(mp_image, int(timestamp * 1000))

        pose_results.append(pose_result)
        hand_results.append(hand_result)
        face_results.append(face_result)

    return pose_results, hand_results, face_results


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
    pose_arr = pose_arr[12:23, :]
    norms = np.linalg.norm(pose_arr, axis=1, keepdims=True)
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

    return pose_features.copy(), checked_frames + 1, max_accel, 0


def should_discard_clip(static_status, checked_frames):
    """Determine if a clip should be discarded."""
    return static_status == 2 or checked_frames < FPS


class ClipWriter:
    """Manages writing clip data to HDF5 in chunks to avoid RAM overflow."""

    def __init__(self, h5_group, clip, chunk_size=WRITE_BUFFER_SIZE):
        self.h5_group = h5_group
        self.clip = clip
        self.chunk_size = chunk_size

        # Buffers for accumulating data
        self.pose_buffer = []
        self.left_buffer = []
        self.right_buffer = []
        self.face_buffer = []
        self.timestamp_buffer = []

        # Motion tracking
        self.static_status = 0
        self.last_position = None
        self.checked_frames = 0
        self.max_accel = 0

        # Track if we've initialized datasets
        self.datasets_created = False
        self.total_frames = 0

    def add_frame(self, pose_lm, left_lm, right_lm, face_lm, timestamp):
        """Add a frame's landmarks to the buffer."""
        self.pose_buffer.append(landmarks_to_array(pose_lm, POSE_LANDMARKS))
        self.left_buffer.append(landmarks_to_array(left_lm, HAND_LANDMARKS))
        self.right_buffer.append(landmarks_to_array(right_lm, HAND_LANDMARKS))
        self.face_buffer.append(landmarks_to_array(face_lm, FACE_LANDMARKS))
        self.timestamp_buffer.append(timestamp)

        # Update motion status
        self.last_position, self.checked_frames, self.max_accel, self.static_status = \
            check_motion_status(pose_lm, self.last_position, self.checked_frames, self.max_accel)

        # Write to disk if buffer is full
        if len(self.pose_buffer) >= self.chunk_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Write buffered data to HDF5."""
        if len(self.pose_buffer) == 0:
            return

        pose_arr = np.array(self.pose_buffer)
        left_arr = np.array(self.left_buffer)
        right_arr = np.array(self.right_buffer)
        face_arr = np.array(self.face_buffer)
        timestamp_arr = np.array(self.timestamp_buffer)

        if not self.datasets_created:
            # Create resizable datasets
            self.h5_group.create_dataset(
                "pose_landmarks",
                data=pose_arr,
                maxshape=(None, POSE_LANDMARKS, LANDMARK_DIMS),
                chunks=True
            )
            self.h5_group.create_dataset(
                "left_hand_landmarks",
                data=left_arr,
                maxshape=(None, HAND_LANDMARKS, LANDMARK_DIMS),
                chunks=True
            )
            self.h5_group.create_dataset(
                "right_hand_landmarks",
                data=right_arr,
                maxshape=(None, HAND_LANDMARKS, LANDMARK_DIMS),
                chunks=True
            )
            self.h5_group.create_dataset(
                "face_landmarks",
                data=face_arr,
                maxshape=(None, FACE_LANDMARKS, LANDMARK_DIMS),
                chunks=True
            )
            self.h5_group.create_dataset(
                "timestamps",
                data=timestamp_arr,
                maxshape=(None,),
                chunks=True
            )
            self.datasets_created = True
        else:
            # Append to existing datasets
            for name, arr in [
                ("pose_landmarks", pose_arr),
                ("left_hand_landmarks", left_arr),
                ("right_hand_landmarks", right_arr),
                ("face_landmarks", face_arr),
                ("timestamps", timestamp_arr)
            ]:
                dataset = self.h5_group[name]
                old_size = dataset.shape[0]
                new_size = old_size + len(arr)
                dataset.resize(new_size, axis=0)
                dataset[old_size:new_size] = arr

        self.total_frames += len(self.pose_buffer)

        # Clear buffers
        self.pose_buffer.clear()
        self.left_buffer.clear()
        self.right_buffer.clear()
        self.face_buffer.clear()
        self.timestamp_buffer.clear()

    def finalize(self):
        """Flush remaining data and set clip attributes."""
        self._flush_buffer()

        if self.datasets_created:
            self.h5_group.attrs["start"] = self.clip.start
            self.h5_group.attrs["end"] = self.clip.end

        return not should_discard_clip(self.static_status, self.checked_frames)


def read_video_once_for_all_clips(reader, clips, fps):
    """Read video once sequentially and yield frames for each clip."""
    fps_original = reader.get_fps()
    skip_rate = int(round(fps_original / fps))

    sorted_clips = sorted(clips, key=lambda c: c.start)

    clip_frame_ranges = []
    for clip in sorted_clips:
        start_frame = int(clip.start * fps_original)
        end_frame = int(clip.end * fps_original)
        clip_frame_ranges.append((start_frame, end_frame, clip))

    current_clip_idx = 0
    frame_count = 0

    while current_clip_idx < len(clip_frame_ranges):
        ret, frame = reader.read()
        if not ret:
            break

        start_frame, end_frame, clip = clip_frame_ranges[current_clip_idx]

        if frame_count < start_frame:
            frame_count += 1
            continue

        if frame_count >= end_frame:
            current_clip_idx += 1
            frame_count += 1
            continue

        if frame_count % skip_rate == 0:
            timestamp = frame_count / fps_original
            yield clip, frame, timestamp

        frame_count += 1


def process_person_clips_optimized(person, reader, output_f):
    """Process all clips for a single person using batched GPU inference and streaming writes."""
    person_group = output_f.create_group(f"person_{person.id}")

    pose_options = create_pose_options()
    hand_options = create_hand_options()
    face_options = create_face_options()

    # Create ClipWriter for each clip
    clip_writers = {}
    for clip_index, clip in enumerate(person.clips):
        clip_group = person_group.create_group(f"{clip_index}")
        clip_writers[id(clip)] = ClipWriter(clip_group, clip)

    with vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
            vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
            vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:

        # Batch processing buffers
        frame_batch = []
        timestamp_batch = []
        clip_batch = []

        for clip_obj, frame, timestamp in read_video_once_for_all_clips(reader, person.clips, FPS):
            clip_writer = clip_writers[id(clip_obj)]

            # Skip if clip already marked as static
            if clip_writer.static_status == 2:
                continue

            # Get bounding box
            bounding_box_idx = clip_obj.boxes.bisect_left(timestamp)
            bounding_box_idx = max(0, bounding_box_idx - 1)
            bounding_box = clip_obj.boxes.peekitem(bounding_box_idx)[1]

            # Crop frame
            crop_coords = calculate_crop_region(bounding_box, clip_obj.max_box_size)
            cropped_frame = crop_and_prepare_frame(frame, crop_coords)

            # Add to batch
            frame_batch.append(cropped_frame)
            timestamp_batch.append(timestamp)
            clip_batch.append(clip_obj)

            # Process batch when full
            if len(frame_batch) >= FRAME_BATCH_SIZE:
                pose_results, hand_results, face_results = detect_landmarks_batch(
                    frame_batch, timestamp_batch, pose_landmarker, hand_landmarker, face_landmarker
                )

                # Process results and write to disk
                for i, (p_res, h_res, f_res, ts, c_obj) in enumerate(
                        zip(pose_results, hand_results, face_results, timestamp_batch, clip_batch)
                ):
                    pose_lm = extract_pose_landmarks(p_res)
                    face_lm = extract_face_landmarks(f_res)
                    left_lm, right_lm = extract_hand_landmarks(h_res)

                    clip_writers[id(c_obj)].add_frame(pose_lm, left_lm, right_lm, face_lm, ts)

                # Clear batches
                frame_batch.clear()
                timestamp_batch.clear()
                clip_batch.clear()

        # Process remaining frames in batch
        if len(frame_batch) > 0:
            pose_results, hand_results, face_results = detect_landmarks_batch(
                frame_batch, timestamp_batch, pose_landmarker, hand_landmarker, face_landmarker
            )

            for i, (p_res, h_res, f_res, ts, c_obj) in enumerate(
                    zip(pose_results, hand_results, face_results, timestamp_batch, clip_batch)
            ):
                pose_lm = extract_pose_landmarks(p_res)
                face_lm = extract_face_landmarks(f_res)
                left_lm, right_lm = extract_hand_landmarks(h_res)

                clip_writers[id(c_obj)].add_frame(pose_lm, left_lm, right_lm, face_lm, ts)

    # Finalize all clips
    for clip_id, writer in clip_writers.items():
        keep_clip = writer.finalize()
        if not keep_clip:
            # Remove clip group if it should be discarded
            clip_index = [i for i, c in enumerate(person.clips) if id(c) == clip_id][0]
            del person_group[f"{clip_index}"]


def process_file(PATH, filename, unlabeled=False):
    """Process a single video file."""
    adjusted_path = (PATH.name + "/unlabeled") if unlabeled else PATH.name
    output_path = ROOT_DIR / "data" / "processed" / "landmarks" / adjusted_path / filename.replace(".mp4", ".h5")

    if check_if_already_processed(output_path):
        return

    video_path = PATH / ("unlabeled/video" if unlabeled else "video") / filename
    reader = GPUVideoReader(video_path)

    bb_path = ROOT_DIR / "data" / "processed" / "bounding_boxes" / adjusted_path / filename.replace(".mp4", ".json")
    bounding_boxes = load_bounding_boxes(bb_path)

    people = create_person_clips(bounding_boxes)

    with h5py.File(output_path, "w") as output_f:
        for person in people.values():
            process_person_clips_optimized(person, reader, output_f)

        output_f.attrs["fps"] = FPS
        output_f.attrs["done"] = True

    reader.release()


def process_folder(PATH):
    """Process all videos in a folder."""
    os.makedirs(ROOT_DIR / "data" / "processed" / "landmarks" / PATH.name / "unlabeled", exist_ok=True)

    files = sorted(os.listdir(PATH / "video"))
    unlabeled_files = sorted(os.listdir(PATH / "unlabeled" / "video")) if os.path.exists(PATH / "unlabeled") else []

    def process_labeled(f):
        process_file(PATH, f, unlabeled=False)

    def process_unlabeled(f):
        process_file(PATH, f, unlabeled=True)

    # REDUCE parallel processes for GPU decoding (2-4 is optimal)
    pool = ProcessPool(nodes=NUM_WORKERS)  # Changed from 10 to 3
    list(tqdm(pool.imap(process_labeled, files), total=len(files), file=sys.stdout))
    list(tqdm(pool.imap(process_unlabeled, unlabeled_files), total=len(unlabeled_files), file=sys.stdout))


def main():
    """Main entry point."""
    for folder in sorted(os.listdir(ROOT_DIR / "data" / "raw")):
        process_folder(ROOT_DIR / "data" / "raw" / folder)


if __name__ == "__main__":
    main()
