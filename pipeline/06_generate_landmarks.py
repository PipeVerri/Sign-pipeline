import os
import sys
import h5py
from pathlib import Path
from pathos.multiprocessing import ProcessPool
from tqdm import tqdm

from utils.args import parse_args
from pipeline.landmarks.processor import process_video

def merge_temp_files(temp_dir, output_h5_path, fps):
    temp_dir = Path(temp_dir)
    output_h5_path = Path(output_h5_path)
    with h5py.File(output_h5_path, "a") as out_f:
        for temp_h5 in temp_dir.glob("*.h5"):
            with h5py.File(temp_h5, "r") as src_f:
                for key in src_f.keys():
                    if key not in out_f:
                        src_f.copy(key, out_f, name=key)
        out_f.attrs["fps"] = fps


def process_source(source, working, cfg):
    source_name = source.name
    working = Path(working)

    labeled_dir = working / "videos" / source_name / "labeled" / "video"
    unlabeled_dir = working / "videos" / source_name / "unlabeled" / "video"
    temp_dir = working / "processed" / "landmarks" / "tmp" / source_name
    output_dir = working / "processed" / "landmarks"

    print(labeled_dir.exists())

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    labeled_videos = sorted(labeled_dir.glob("*.mp4")) if labeled_dir.exists() else []
    unlabeled_videos = sorted(unlabeled_dir.glob("*.mp4")) if unlabeled_dir.exists() else []

    pool = ProcessPool(nodes=cfg.num_workers)

    if labeled_videos:
        def process_labeled(video_path):
            process_video(video_path, is_labeled=True, working=working, cfg=cfg)

        for _ in tqdm(pool.uimap(process_labeled, labeled_videos),
                      total=len(labeled_videos), desc=f"[{source_name}] labeled", file=sys.stdout):
            pass
        merge_temp_files(temp_dir, output_dir / f"{source_name}_labeled.h5", cfg.fps)

    if unlabeled_videos:
        def process_unlabeled(video_path):
            process_video(video_path, is_labeled=False, working=working, cfg=cfg)

        for _ in tqdm(pool.uimap(process_unlabeled, unlabeled_videos),
                      total=len(unlabeled_videos), desc=f"[{source_name}] unlabeled", file=sys.stdout):
            pass
        merge_temp_files(temp_dir, output_dir / f"{source_name}_unlabeled.h5", cfg.fps)

    pool.close()
    pool.join()
    pool.clear()


if __name__ == "__main__":
    working, config = parse_args()

    for source in config.sources:
        process_source(source, working, config.options.landmarks)
