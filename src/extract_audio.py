import subprocess
from pathlib import Path
import os
from multiprocessing import Pool

root_dir = Path(__file__).parent.parent.resolve()

def extract_audio(video_path, audio_path):
    subprocess.run([
        "ffmpeg", "-i", str(video_path),
        "-vn", "-acodec", "mp3",
        "-ar", "16000", "-ac", "1",
        str(audio_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

folder = "videolibros_private"
PATH = root_dir / "data" / "raw" / folder

files = [f for f in os.listdir(PATH) if f.endswith(".mp4")]

audio_dir = PATH / "audio"
audio_dir.mkdir(exist_ok=True)

# Prepare args for starmap
jobs = [
    (PATH / f, audio_dir / f.replace(".mp4", ".mp3"))
    for f in files
]

if __name__ == "__main__":
    with Pool(12) as p:
        p.starmap(extract_audio, jobs)
