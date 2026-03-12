import os
import sys
import h5py
from pathlib import Path
from pathos.multiprocessing import ProcessPool
from tqdm import tqdm

from utils.args import parse_args
from pipeline.landmarks.config import FPS, FRAME_BATCH_SIZE
from pipeline.landmarks.processor import process_video


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


def process_source(source, working, step_config):
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
        process_source(source, working, step_config)
