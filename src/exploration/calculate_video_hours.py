#!/usr/bin/env python3
import subprocess
from pathlib import Path

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
        print(f"⚠️  Failed to read duration: {path}")
        return 0.0


def cumulative_video_hours(folders):
    total_seconds = 0.0
    video_count = 0

    for folder in folders:
        folder = Path(folder)
        if not folder.exists():
            print(f"⚠️  Skipping missing folder: {folder}")
            continue

        for file in folder.rglob("*"):
            if file.suffix.lower() in VIDEO_EXTS:
                total_seconds += get_video_duration_seconds(file)
                video_count += 1

    total_hours = total_seconds / 3600
    return total_hours, video_count


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent.resolve()
    FOLDERS = [
        root_dir / "data" / "raw" / "0-AsociacionCivil",
        root_dir / "data" / "raw" / "1-videolibros_private",
        root_dir / "data" / "raw" / "2-videolibros_public",
        root_dir / "data" / "raw" / "3-CNSordos",
        root_dir / "data" / "raw" / "4-Locufre"
    ]

    hours, count = cumulative_video_hours(FOLDERS)

    print(f"\n🎬 Videos counted: {count}")
    print(f"⏱️  Total duration: {hours:.2f} hours")
