from utils.args import parse_args
import os
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from tqdm import tqdm

load_dotenv()

working, config = parse_args()

print("Downloading Whisper model...")

whisper_cfg = config.options.whisper
model = WhisperModel(whisper_cfg.model, device=whisper_cfg.device)
language = whisper_cfg.language

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def transcribe(audio_path, vtt_path):
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True, beam_size=5, language=language)
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for segment in segments:
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            text = segment.text.strip()
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")


def process_source(source):
    source_dir = working / "videos" / source.name
    video_dir = source_dir / "video"
    audio_dir = source_dir / "audio"

    os.makedirs(source_dir / "labeled/video", exist_ok=True)
    os.makedirs(source_dir / "labeled/subtitles", exist_ok=True)

    videos = [v for v in os.listdir(video_dir) if v.endswith(".mp4")]
    for video_name in tqdm(videos, desc="Generating subtitles"):
        name = video_name.split(".")[0]
        audio_path = audio_dir / f"{name}.mp3"
        vtt_path = source_dir / "labeled/subtitles" / f"{name}.vtt"

        print(f"Transcribing {name}...")
        transcribe(audio_path, vtt_path)
        os.rename(video_dir / video_name, source_dir / "labeled/video" / video_name)
        print(f"Done {name}")


for source in config.sources:
    if source.generate_subs:
        print(f"\nProcessing {source.name}...")
        process_source(source)