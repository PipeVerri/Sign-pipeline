from args import parse_args
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

working, config = parse_args()

def delete_file(f):
    try:
        os.remove(f)
    except OSError:
        pass


def vad_process_video(source_dir, video_name):
    """Use VAD to detect speech. If none found, move to unlabeled."""
    name = video_name.split(".")[0]
    audio_path = source_dir / "audio" / f"{name}.mp3"
    video_path = source_dir / "video" / video_name

    model = load_silero_vad()
    wav = read_audio(str(audio_path))
    speech_timestamps = get_speech_timestamps(wav, model, return_seconds=False)

    if len(speech_timestamps) == 0:
        os.rename(video_path, source_dir / "unlabeled/video" / video_name)
        os.rename(audio_path, source_dir / "unlabeled/audio" / f"{name}.mp3")
        delete_file(source_dir / "subtitles" / f"{name}.vtt")
        return f"Moved {name} to unlabeled (no speech)"
    return f"{name} - speech detected"


def vad_filter_videos(source_dir, videos, max_workers=os.cpu_count()):
    """Run VAD on a list of unsubtitled videos in parallel."""
    os.makedirs(source_dir / "unlabeled/video", exist_ok=True)
    os.makedirs(source_dir / "unlabeled/audio", exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(vad_process_video, source_dir, v): v for v in videos}
        for future in as_completed(futures):
            video = futures[future]
            try:
                print(future.result())
            except Exception as e:
                print(f"Error with {video}: {str(e)}")


def process_subtitled(source):
    source_dir = working / "videos" / source["name"]
    os.makedirs(source_dir / "labeled/video", exist_ok=True)
    os.makedirs(source_dir / "labeled/subtitles", exist_ok=True)
    os.makedirs(source_dir / "unlabeled/video", exist_ok=True)

    # Rename subtitles: yt-dlp saves as name.lang.vtt, we want name.vtt
    for sub in os.listdir(source_dir / "subtitles"):
        dest_name = sub.split(".")[0] + ".vtt"
        if sub != dest_name:
            os.rename(source_dir / "subtitles" / sub, source_dir / "subtitles" / dest_name)

    # Sort each video by whether it has a matching subtitle
    unsubtitled = []
    for video in os.listdir(source_dir / "video"):
        sub_name = video.split(".")[0] + ".vtt"
        if os.path.exists(source_dir / "subtitles" / sub_name):
            os.rename(source_dir / "video" / video, source_dir / "labeled/video" / video)
            os.rename(source_dir / "subtitles" / sub_name, source_dir / "labeled/subtitles" / sub_name)
        else:
            unsubtitled.append(video)

    if not unsubtitled:
        return

    if source.get("auto_subs") or source.get("generate_subs"):
        # Keep videos with speech so step 04 can transcribe them (or VAD-filter if no generate_subs)
        print(f"Running VAD on {len(unsubtitled)} unsubtitled videos in {source['name']}...")
        vad_filter_videos(source_dir, unsubtitled)
    else:
        # Manual subs only, no generation: videos without a sub go straight to unlabeled
        for video in unsubtitled:
            os.rename(source_dir / "video" / video, source_dir / "unlabeled/video" / video)


def process_unsubtitled(source):
    source_dir = working / "videos" / source["name"]
    os.makedirs(source_dir / "unlabeled/video", exist_ok=True)
    for video in os.listdir(source_dir / "video"):
        os.rename(source_dir / "video" / video, source_dir / "unlabeled/video" / video)


for source in config["sources"]:
    if source.get("subs") or source.get("auto_subs"):
        # Has downloaded subs: sort labeled/unlabeled, VAD on unsubtitled if auto_subs
        process_subtitled(source)
    elif source.get("generate_subs"):
        # No downloaded subs but Whisper will run next: VAD-filter speechless videos only
        source_dir = working / "videos" / source["name"]
        videos = [v for v in os.listdir(source_dir / "video") if v.endswith(".mp4")]
        if videos:
            print(f"Running VAD on {len(videos)} videos in {source['name']}...")
            vad_filter_videos(source_dir, videos)
    else:
        process_unsubtitled(source)
