import ctypes
import importlib.util
import os

import numpy as np

# ---------------------------------------------------------------------------
# Preload CUDA 12 runtime libs from nvidia pip packages before onnxruntime
# tries to find them by soname. This resolves the mismatch between
# onnxruntime-gpu (built for CUDA 12) and a system CUDA 13 installation.
# The dynamic linker caches loaded sonames, so subsequent dlopen("libcublas.so.12")
# calls from libonnxruntime_providers_cuda.so will reuse these handles.
# ---------------------------------------------------------------------------
def _preload_nvidia_cuda12_libs():
    _libs = [
        ("nvidia.cuda_runtime", "libcudart.so.12"),
        ("nvidia.cublas",       "libcublas.so.12"),
        ("nvidia.cublas",       "libcublasLt.so.12"),
        ("nvidia.cufft",        "libcufft.so.11"),
    ]
    for pkg, lib_name in _libs:
        spec = importlib.util.find_spec(pkg)
        if spec and spec.origin:
            lib_path = os.path.join(os.path.dirname(spec.origin), "lib", lib_name)
            if os.path.exists(lib_path):
                try:
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass

_preload_nvidia_cuda12_libs()

from rtmlib import Wholebody3d

# COCO-WholeBody 133-keypoint index ranges
BODY_SLICE = slice(0, 17)
FACE_SLICE = slice(23, 91)
LEFT_HAND_SLICE = slice(91, 112)
RIGHT_HAND_SLICE = slice(112, 133)


def create_wholebody3d(mode: str, backend: str, device: str) -> Wholebody3d:
    return Wholebody3d(mode=mode, backend=backend, device=device)

def run_pose(model: Wholebody3d, frame: np.ndarray, bbox: list) -> tuple[np.ndarray, np.ndarray]:
    """Run pose estimation with a pre-computed bounding box, bypassing internal detection.

    Returns (kpts, scores) where:
        kpts:   (133, 3) array with absolute pixel x,y and metric z depth
        scores: (133,)  confidence scores
    """
    keypoints, scores, _, keypoints_2d = model.pose_model(frame, bboxes=[bbox])
    # keypoints_2d: (1, 133, 2) absolute pixel coords
    # keypoints:    (1, 133, 3) where [:,:,2] is metric depth
    kpts = np.concatenate([keypoints_2d[0], keypoints[0, :, 2:3]], axis=-1)  # (133, 3)
    return kpts, scores[0]


def split_keypoints(kpts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split 133-keypoint array into body, left_hand, right_hand, face."""
    return (
        kpts[BODY_SLICE],       # (17, 3)
        kpts[LEFT_HAND_SLICE],  # (21, 3)
        kpts[RIGHT_HAND_SLICE], # (21, 3)
        kpts[FACE_SLICE],       # (68, 3)
    )
