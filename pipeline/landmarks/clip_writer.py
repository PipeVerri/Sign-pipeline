import numpy as np

from utils.shared.utils.mediapipe import mp_to_arr
from .config import (
    POSE_LANDMARKS, HAND_LANDMARKS, FACE_LANDMARKS, LANDMARK_DIMS,
    WRITE_BUFFER_SIZE, MIN_CLIP_DURATION, FPS, MOVING_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Landmark conversion: normalized (crop-relative) → absolute pixel coords
# ---------------------------------------------------------------------------

def landmarks_to_absolute_array(landmarks, expected_count, crop_coords):
    """Convert MediaPipe landmarks from crop-relative [0,1] to absolute pixel
    coordinates within the original full frame.

    crop_coords: (y0, y1, x0, x1) in pixels.
    z is scaled by crop width (MediaPipe convention for depth).
    """
    if landmarks is None:
        return np.full((expected_count, LANDMARK_DIMS), np.nan)

    y0, y1, x0, x1 = crop_coords
    crop_w = x1 - x0
    crop_h = y1 - y0

    arr = mp_to_arr(landmarks)
    arr[:, 0] = arr[:, 0] * crop_w + x0
    arr[:, 1] = arr[:, 1] * crop_h + y0
    arr[:, 2] = arr[:, 2] * crop_w
    return arr


# ---------------------------------------------------------------------------
# Motion detection (operates on crop-relative landmarks — relative is enough)
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
# ClipWriter — chunked H5 writer with absolute landmark coordinates
# ---------------------------------------------------------------------------

class ClipWriter:
    """Accumulates per-frame landmarks (in absolute pixel coords) and flushes
    to resizable HDF5 datasets in chunks to avoid memory overflow on long clips."""

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

    def add_frame(self, pose_lm, left_lm, right_lm, face_lm, timestamp, crop_coords):
        self.pose_buf.append(landmarks_to_absolute_array(pose_lm, POSE_LANDMARKS, crop_coords))
        self.left_buf.append(landmarks_to_absolute_array(left_lm, HAND_LANDMARKS, crop_coords))
        self.right_buf.append(landmarks_to_absolute_array(right_lm, HAND_LANDMARKS, crop_coords))
        self.face_buf.append(landmarks_to_absolute_array(face_lm, FACE_LANDMARKS, crop_coords))
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
        """Flush remaining data and write attrs. Returns True if clip should be kept."""
        self._flush()
        if self.datasets_created:
            self.h5_group.attrs["start"] = self.clip.start
            self.h5_group.attrs["end"] = self.clip.end
        return not should_discard_clip(self.static_status, self.checked_frames)
