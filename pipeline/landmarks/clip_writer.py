import numpy as np

from utils.config import LandmarksConfig

# Keypoint counts for COCO-WholeBody (rtmlib Wholebody3d)
BODY_LANDMARKS = 17
HAND_LANDMARKS = 21
FACE_LANDMARKS = 68
LANDMARK_DIMS = 3


# ---------------------------------------------------------------------------
# Motion detection (operates on absolute pixel coordinates)
# ---------------------------------------------------------------------------

def check_motion_status(body_landmarks, last_position, checked_frames, max_accel,
                        min_clip_duration, moving_threshold):
    """body_landmarks is a (17, 3) numpy array in absolute pixel coords, or None."""
    if body_landmarks is None:
        return last_position, checked_frames, max_accel, 0

    # Use torso + limbs (shoulders to ankles, indices 5-16) — analogous to
    # MediaPipe indices 12-22 used before.
    torso = body_landmarks[5:17, :]  # (12, 3)
    norms = np.linalg.norm(torso, axis=1, keepdims=True)
    pose_features = torso / np.where(norms == 0, 1, norms)

    if last_position is not None:
        change = np.linalg.norm(pose_features - last_position)
        max_accel = max(max_accel, change)
        if checked_frames >= min_clip_duration:
            status = 2 if max_accel < moving_threshold else 1
            return pose_features.copy(), checked_frames + 1, max_accel, status

    return pose_features.copy(), checked_frames + 1, max_accel, 0


def should_discard_clip(static_status, checked_frames, fps):
    return static_status == 2 or checked_frames < fps


# ---------------------------------------------------------------------------
# ClipWriter — chunked H5 writer
# ---------------------------------------------------------------------------

class ClipWriter:
    """Accumulates per-frame landmarks (absolute pixel coords from rtmlib) and
    flushes to resizable HDF5 datasets in chunks."""

    def __init__(self, h5_group, clip, cfg: LandmarksConfig):
        self.h5_group = h5_group
        self.clip = clip
        self.chunk_size = cfg.write_buffer_size
        self.min_clip_duration = cfg.min_clip_duration_frames
        self.moving_threshold = cfg.moving_threshold
        self.fps = cfg.fps

        self.body_buf = []
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

    def add_frame(self, body_lm, left_lm, right_lm, face_lm, timestamp):
        """body_lm: (17,3), left_lm/right_lm: (21,3), face_lm: (68,3) — absolute pixel coords."""
        self.body_buf.append(body_lm if body_lm is not None else np.full((BODY_LANDMARKS, LANDMARK_DIMS), np.nan))
        self.left_buf.append(left_lm if left_lm is not None else np.full((HAND_LANDMARKS, LANDMARK_DIMS), np.nan))
        self.right_buf.append(right_lm if right_lm is not None else np.full((HAND_LANDMARKS, LANDMARK_DIMS), np.nan))
        self.face_buf.append(face_lm if face_lm is not None else np.full((FACE_LANDMARKS, LANDMARK_DIMS), np.nan))
        self.ts_buf.append(timestamp)

        self.last_position, self.checked_frames, self.max_accel, self.static_status = \
            check_motion_status(body_lm, self.last_position, self.checked_frames, self.max_accel,
                                self.min_clip_duration, self.moving_threshold)

        if len(self.body_buf) >= self.chunk_size:
            self._flush()

    def _flush(self):
        if not self.body_buf:
            return

        body_arr = np.array(self.body_buf)
        left_arr = np.array(self.left_buf)
        right_arr = np.array(self.right_buf)
        face_arr = np.array(self.face_buf)
        ts_arr = np.array(self.ts_buf)

        if not self.datasets_created:
            self.h5_group.create_dataset("body_landmarks", data=body_arr,
                                         maxshape=(None, BODY_LANDMARKS, LANDMARK_DIMS), chunks=True)
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
                ("body_landmarks", body_arr),
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

        self.total_frames += len(self.body_buf)
        self.body_buf.clear()
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
        return not should_discard_clip(self.static_status, self.checked_frames, self.fps)
