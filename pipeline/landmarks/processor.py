import os
import json
from pathlib import Path

import h5py

from utils.gpu_reader import read_video_for_clips
from utils.config import LandmarksConfig
from .person import PersonResults
from .detectors import create_wholebody3d, run_pose, split_keypoints
from .clip_writer import ClipWriter


# ---------------------------------------------------------------------------
# Per-person clip processing
# ---------------------------------------------------------------------------

def process_person_clips(person, video_path, person_group, wholebody, cfg: LandmarksConfig):
    print(f"Processing {person.id}...")
    fps = cfg.fps

    clip_writers = {}
    for clip_index, clip in enumerate(person.clips):
        grp = person_group.create_group(f"{clip_index}")
        clip_writers[id(clip)] = (clip_index, ClipWriter(grp, clip, cfg))

    for clip_obj, frame, ts in read_video_for_clips(video_path, person.clips, fps):
        _, writer = clip_writers[id(clip_obj)]
        if writer.static_status == 2:
            continue

        bb_idx = max(0, clip_obj.boxes.bisect_left(ts) - 1)
        bbox = clip_obj.boxes.peekitem(bb_idx)[1]  # [x1, y1, x2, y2]

        kpts, scores = run_pose(wholebody, frame, bbox)
        body, left_hand, right_hand, face = split_keypoints(kpts)

        writer.add_frame(body, left_hand, right_hand, face, ts)

    for clip_id, (clip_index, writer) in clip_writers.items():
        if not writer.finalize():
            del person_group[f"{clip_index}"]


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(video_path, is_labeled, working, cfg: LandmarksConfig):
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
                people[person_id] = PersonResults(person_id, cfg.max_clip_frame_separation)
            people[person_id].add_bounding_box_frame(entry["timestamp"], box)

    subtitles = ""
    if is_labeled and subtitle_file.exists():
        subtitles = subtitle_file.read_text(encoding="utf-8")

    wholebody = create_wholebody3d(mode=cfg.mode, backend=cfg.backend, device=cfg.device)

    os.makedirs(temp_h5.parent, exist_ok=True)

    with h5py.File(temp_h5, "w") as output_f:
        video_group = output_f.create_group(video_path.stem)
        video_group.attrs["video_id"] = video_path.stem
        video_group.attrs["labeled"] = is_labeled
        video_group.attrs["subtitles"] = subtitles

        for person in people.values():
            person_group = video_group.create_group(f"person_{person.id}")
            process_person_clips(person, video_path, person_group, wholebody, cfg)

        output_f.attrs["fps"] = cfg.fps
        output_f.attrs["done"] = True
