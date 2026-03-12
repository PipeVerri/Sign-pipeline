from args import parse_args
import subprocess
import os
from multiprocessing import Pool

working, config = parse_args()

def extract_audio(video_with_audio, output_path):
    if output_path.exists():
        return
    subprocess.run([
        "ffmpeg", "-i", str(video_with_audio),
        "-vn", "-acodec", "mp3",
        "-ar", "16000", "-ac", "1",
        str(output_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def extract_video(video_with_audio, output_path):
    if output_path.exists():
        return
    subprocess.run([
        "ffmpeg", "-i", str(video_with_audio),
        "-an", "-c:v", "copy",
        str(output_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def process_folder(path):
    files = [f for f in os.listdir(path) if f.endswith(".mp4")]

    audio_dir = path / "audio"
    video_dir = path / "video"
    subs_dir = path / "subtitles"
    audio_dir.mkdir(exist_ok=True)
    video_dir.mkdir(exist_ok=True)
    subs_dir.mkdir(exist_ok=True)

    # Prepare args for audio extraction
    audio_jobs = [
        (path / f, audio_dir / f.replace(".mp4", ".mp3"))
        for f in files
    ]

    # Prepare args for video extraction
    video_jobs = [
        (path / f, video_dir / f)
        for f in files
    ]

    with Pool(os.cpu_count()) as p:
        print("Extracting audio...")
        p.starmap(extract_audio, audio_jobs)
        print("Extracting video...")
        p.starmap(extract_video, video_jobs)

    print(f"Moving subtitles...")
    for sub in os.listdir(path):
        if sub.endswith(".vtt"):
            os.rename(path / sub, subs_dir / sub)

    if config["options"]["video_audio_separation"]["delete_original"]:
        print(f"Deleting original files...")
        for f in files:
            os.remove(path / f)

    print("Done!")

for folder in os.listdir(working / "videos"):
    print(f"\nProcessing {folder}...")
    process_folder(working / "videos" / folder)