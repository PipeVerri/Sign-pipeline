import os
import sys
import contextlib
import logging

os.environ['GLOG_minloglevel'] = '3'
logging.getLogger('mediapipe').setLevel(logging.ERROR)


@contextlib.contextmanager
def _mute_stderr_fd():
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


with _mute_stderr_fd():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision


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
