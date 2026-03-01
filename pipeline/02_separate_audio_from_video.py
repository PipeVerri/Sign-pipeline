from src.args import parse_args
import subprocess

working, _ = parse_args()

def extract_audio(video_path, audio_path):
    if audio_path.exists():
        return
    subprocess.run([
        "ffmpeg", "-i", str(video_path),
        "-vn", "-acodec", "mp3",
        "-ar", "16000", "-ac", "1",
        str(audio_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def extract_video(video_path, video_output_path):
    if video_output_path.exists():
        return
    subprocess.run([
        "ffmpeg", "-i", str(video_path),
        "-an", "-c:v", "copy",
        str(video_output_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def process_folder(PATH):
    files = [f for f in os.listdir(PATH) if f.endswith(".mp4")]

    audio_dir = PATH / "audio"
    video_dir = PATH / "video"
    audio_dir.mkdir(exist_ok=True)
    video_dir.mkdir(exist_ok=True)

    # Prepare args for audio extraction
    audio_jobs = [
        (PATH / f, audio_dir / f.replace(".mp4", ".mp3"))
        for f in files
    ]

    # Prepare args for video extraction
    video_jobs = [
        (PATH / f, video_dir / f)
        for f in files
    ]

    with Pool(12) as p:
        print("Extracting audio...")
        p.starmap(extract_audio, audio_jobs)
        print("Extracting video...")
        p.starmap(extract_video, video_jobs)
    print("Done!")