#!/usr/bin/env python3
"""
batch_resize_144p.py

Batch-resize videos so their height is 144 pixels (144p), preserving aspect ratio.
Runs ffmpeg concurrently on multiple files.

Usage examples:
  python3 batch_resize_144p.py /path/to/videos --outdir resized_144p --workers 8
  python3 batch_resize_144p.py video1.mp4 video2.mkv --outdir out --overwrite

Requirements:
  - ffmpeg (and optionally ffprobe) on PATH
  - Python 3.7+
  - tqdm (optional, for progress bar): pip install tqdm

Behavior:
  - Preserves aspect ratio, width is calculated to maintain even value (ffmpeg scale filter uses -2 to ensure even width).
  - Uses libx264 to encode video by default (change codec options if you want another encoder).
  - Copies audio stream by default.
  - Skips files when output already exists unless --overwrite is specified.

"""
from __future__ import annotations
import argparse
import concurrent.futures
import multiprocessing
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v"}


def find_video_files(inputs: List[str], recursive: bool) -> List[Path]:
    files: List[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            if recursive:
                for f in p.rglob("*"):
                    if f.suffix.lower() in VIDEO_EXTS and f.is_file():
                        files.append(f)
            else:
                for f in p.iterdir():
                    if f.suffix.lower() in VIDEO_EXTS and f.is_file():
                        files.append(f)
        else:
            # allow globs
            for f in sorted(Path('.').glob(inp)):
                if f.suffix.lower() in VIDEO_EXTS and f.is_file():
                    files.append(f)
    # de-duplicate and sort
    return sorted(list(dict.fromkeys(files)))


def ffprobe_height(path: Path) -> Optional[int]:
    """Return the height of the first video stream using ffprobe, or None if ffprobe is missing / fails."""
    if not shutil.which("ffprobe"):
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=height",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        s = out.decode().strip()
        if s == "":
            return None
        return int(s)
    except Exception:
        return None


def build_ffmpeg_cmd(infile: Path, outfile: Path, crf: int, preset: str, codec: str, copy_audio: bool, movflags_faststart: bool) -> List[str]:
    # Scale filter: width auto-calculated to keep aspect ratio, -2 ensures even width
    vf = "scale=-2:144"
    cmd = ["ffmpeg", "-y", "-i", str(infile), "-vf", vf]

    # video codec
    if codec:
        cmd += ["-c:v", codec]
    else:
        cmd += ["-c:v", "libx264"]

    # tuning quality
    if codec in (None, "libx264"):
        cmd += ["-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p"]

    # audio
    if copy_audio:
        cmd += ["-c:a", "copy"]

    if movflags_faststart:
        cmd += ["-movflags", "+faststart"]

    # container / output
    cmd.append(str(outfile))
    return cmd


def ensure_outpath(infile: Path, outdir: Path, out_ext: Optional[str]) -> Path:
    # preserve folder structure relative to current working directory
    try:
        rel = infile.resolve().relative_to(Path.cwd().resolve())
    except Exception:
        rel = infile.name
    out_path = outdir.joinpath(rel)
    out_path = out_path.with_suffix(out_ext if out_ext is not None else infile.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def process_file(
    infile: Path,
    outpath: Path,
    ffmpeg_args: dict,
    overwrite: bool,
    skip_if_same_height: bool,
) -> tuple[Path, bool, Optional[str]]:
    """Run ffmpeg to resize infile -> outpath.
    Returns (infile, success_bool, error_message_or_none)
    """
    if outpath.exists() and not overwrite:
        return infile, False, "exists"

    if skip_if_same_height:
        h = ffprobe_height(infile)
        if h is not None and h == 144:
            return infile, False, "already 144p"

    cmd = build_ffmpeg_cmd(infile, outpath, **ffmpeg_args)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return infile, True, None
    except subprocess.CalledProcessError as e:
        return infile, False, e.stderr.decode(errors='ignore') if e.stderr else "ffmpeg failed"
    except Exception as e:
        return infile, False, str(e)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-resize videos to height=144p (preserve aspect ratio).")
    p.add_argument("inputs", nargs="+", help="Files, directories, or glob patterns to process")
    p.add_argument("--outdir", default="resized_144p", help="Output base directory")
    p.add_argument("--workers", type=int, default=max(2, multiprocessing.cpu_count()), help="Number of concurrent ffmpeg processes")
    p.add_argument("--crf", type=int, default=23, help="CRF for libx264 (lower = higher quality).")
    p.add_argument("--preset", default="fast", help="x264 preset (ultrafast, superfast, veryfast, faster, fast, medium, slow...)")
    p.add_argument("--codec", default="libx264", help="Video codec to use (default libx264). Set to empty string to let ffmpeg choose by input/format.")
    p.add_argument("--copy-audio", action="store_true", help="Copy audio stream instead of re-encoding")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--recursive", action="store_true", help="If inputs contain directories, traverse recursively")
    p.add_argument("--out-ext", default=None, help="Output file extension (e.g. .mp4). Default: keep input extension")
    p.add_argument("--skip-if-same-height", action="store_true", help="Use ffprobe and skip processing if video already has height 144")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar even if available")
    p.add_argument("--faststart", dest='faststart', action='store_true', help="Add +faststart to mp4s to improve playback start")
    return p.parse_args()


def main():
    args = parse_args()
    inputs = args.inputs
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = find_video_files(inputs, recursive=args.recursive)
    if not files:
        print("No video files found for the given inputs.")
        return

    ffmpeg_args = {
        "crf": args.crf,
        "preset": args.preset,
        "codec": args.codec if args.codec != "" else None,
        "copy_audio": args.copy_audio,
        "movflags_faststart": args.faststart,
    }

    tasks = []
    for f in files:
        outpath = ensure_outpath(f, outdir, args.out_ext)
        tasks.append((f, outpath))

    use_tqdm = (tqdm is not None) and (not args.no_progress)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_file, f, out, ffmpeg_args, args.overwrite, args.skip_if_same_height): (f, out)
            for f, out in tasks
        }
        if use_tqdm:
            pbar = tqdm(total=len(futures), desc="Resizing")
        try:
            for fut in concurrent.futures.as_completed(futures):
                inp, outp = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    results.append((inp, False, str(e)))
                else:
                    results.append(r)
                if use_tqdm:
                    pbar.update(1)
        finally:
            if use_tqdm:
                pbar.close()

    # Summarize
    success = [r for r in results if r[1] is True]
    skipped = [r for r in results if r[1] is False]
    print(f"\nDone. {len(success)} succeeded, {len(skipped)} skipped/failed, total {len(results)}")
    if skipped:
        print("Examples of skipped/failed files:")
        for inp, ok, err in skipped[:10]:
            print(f" - {inp} : {err}")


if __name__ == "__main__":
    main()
