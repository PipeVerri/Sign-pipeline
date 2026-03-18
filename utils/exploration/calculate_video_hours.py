#!/usr/bin/env python3
import subprocess
import os
from pathlib import Path
from utils.args import parse_args

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Video extensions to count
VIDEO_EXTS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".mpg", ".mpeg", ".m4v"
}

def get_video_duration_seconds(path: Path) -> float:
    """
    Returns video duration in seconds using ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return float(out)
    except Exception:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            return float(out)
        except Exception:
            return 0.0


def calculate_source_stats(source_dir: Path, source_name: str):
    stats = {
        "labeled": {"count": 0, "seconds": 0.0},
        "unlabeled": {"count": 0, "seconds": 0.0},
        "other": {"count": 0, "seconds": 0.0}
    }

    if not source_dir.exists():
        return stats

    video_files = []
    for file in source_dir.rglob("*"):
        if file.suffix.lower() in VIDEO_EXTS:
            if "audio" in file.parts or "subtitles" in file.parts:
                continue
            video_files.append(file)

    if not video_files:
        return stats

    for file in tqdm(video_files, desc=f"Processing {source_name}", leave=False):
        duration = get_video_duration_seconds(file)
        
        # Categorize
        if "labeled" in file.parts:
            key = "labeled"
        elif "unlabeled" in file.parts:
            key = "unlabeled"
        else:
            key = "other"
            
        stats[key]["count"] += 1
        stats[key]["seconds"] += duration

    return stats


if __name__ == "__main__":
    working, config = parse_args()

    header = f"{'Source':<25} | {'Labeled (h)':<15} | {'Unlabeled (h)':<15} | {'Total (h)':<15}"
    print(f"\n{header}")
    print("-" * len(header))

    totals = {
        "labeled": {"count": 0, "hours": 0.0},
        "unlabeled": {"count": 0, "hours": 0.0},
        "other": {"count": 0, "hours": 0.0}
    }

    results = []

    for source in config.sources:
        source_dir = working / "videos" / source.name
        source_stats = calculate_source_stats(source_dir, source.name)
        
        row_total_count = sum(s["count"] for s in source_stats.values())
        row_total_hours = sum(s["seconds"] for s in source_stats.values()) / 3600
        
        l_h = source_stats["labeled"]["seconds"] / 3600
        u_h = source_stats["unlabeled"]["seconds"] / 3600
        o_h = source_stats["other"]["seconds"] / 3600
        
        # We merge "other" into unlabeled or just show it if it's there
        # For the sake of the requested sum, we show Labeled, Unlabeled, and Total
        # If 'other' exists (not yet processed), we'll group it with unlabeled for the summary table
        # but let's keep it distinct in the internal logic.
        
        disp_unlabeled_h = u_h + o_h
        disp_unlabeled_c = source_stats["unlabeled"]["count"] + source_stats["other"]["count"]
        
        print(f"{source.name:<25} | "
              f"{source_stats['labeled']['count']:>4} ({l_h:>6.2f}h) | "
              f"{disp_unlabeled_c:>4} ({disp_unlabeled_h:>6.2f}h) | "
              f"{row_total_count:>4} ({row_total_hours:>6.2f}h)")
        
        totals["labeled"]["count"] += source_stats["labeled"]["count"]
        totals["labeled"]["hours"] += l_h
        totals["unlabeled"]["count"] += disp_unlabeled_c
        totals["unlabeled"]["hours"] += disp_unlabeled_h

    print("-" * len(header))
    total_c = totals["labeled"]["count"] + totals["unlabeled"]["count"]
    total_h = totals["labeled"]["hours"] + totals["unlabeled"]["hours"]
    
    print(f"{'TOTAL':<25} | "
          f"{totals['labeled']['count']:>4} ({totals['labeled']['hours']:>6.2f}h) | "
          f"{totals['unlabeled']['count']:>4} ({totals['unlabeled']['hours']:>6.2f}h) | "
          f"{total_c:>4} ({total_h:>6.2f}h)")
