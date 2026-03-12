import os
import json
from pathlib import Path

import numpy as np
import h5py

from utils.gpu_reader import read_video_for_clips
from .config import FPS, BOUNDING_BOX_PADDING, FRAME_BATCH_SIZE
from .person import PersonResults
from .detectors import (
    mp, vision,
    create_pose_options, create_hand_options, create_face_options,
    extract_pose_landmarks, extract_hand_landmarks, extract_face_landmarks,
)
from .clip_writer import ClipWriter

ROOT_DIR = Path(__file__).resolve().parents[2]


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


def crop_frame(frame, crop_coords):
    y0, y1, x0, x1 = crop_coords
    return np.ascontiguousarray(frame[y0:y1, x0:x1])


# ---------------------------------------------------------------------------
# Per-person clip processing
# ---------------------------------------------------------------------------

def process_person_clips(person, video_path, person_group, pose_opts, hand_opts, face_opts, batch_size):
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
        crop_batch = []

        def flush_batch():
            for frame, ts, clip_obj, crop_coords in zip(frame_batch, ts_batch, clip_batch, crop_batch):
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
                    crop_coords,
                )
            frame_batch.clear()
            ts_batch.clear()
            clip_batch.clear()
            crop_batch.clear()

        for clip_obj, frame, ts in read_video_for_clips(video_path, person.clips, FPS):
            _, writer = clip_writers[id(clip_obj)]
            if writer.static_status == 2:
                continue

            bb_idx = max(0, clip_obj.boxes.bisect_left(ts) - 1)
            bounding_box = clip_obj.boxes.peekitem(bb_idx)[1]
            crop_coords = calculate_crop_region(bounding_box, clip_obj.max_box_size)

            frame_batch.append(crop_frame(frame, crop_coords))
            ts_batch.append(ts)
            clip_batch.append(clip_obj)
            crop_batch.append(crop_coords)

            if len(frame_batch) >= batch_size:
                flush_batch()

        if frame_batch:
            flush_batch()

    for clip_id, (clip_index, writer) in clip_writers.items():
        if not writer.finalize():
            del person_group[f"{clip_index}"]


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(video_path, is_labeled, working, step_config):
    video_path = Path(video_path)
    working = Path(working)

    source_name = video_path.parents[2].name
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
