import os
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent.resolve()

def process_folder(PATH):
    # Start by renaming all the subtitles to get rid of .es-x.vtt
    subtitles = [f for f in os.listdir(PATH) if f.endswith(".vtt")]
    for sub in subtitles:
        name = sub.split(".")[0] + ".vtt"
        os.rename(PATH / sub, PATH / name)

    # Now move the videos that don't have subtitles into unlabeled
    os.makedirs(PATH / "unlabeled/video", exist_ok=True)
    os.makedirs(PATH / "unlabeled/audio", exist_ok=True)
    videos = [f for f in os.listdir(PATH) if f.endswith(".mp4")]
    for video in videos:
        sub = video.split(".")[0] + ".vtt"
        if not os.path.exists(PATH / sub):
            os.rename(PATH / "video" / video, PATH / "unlabeled/video" / video)
            os.rename(PATH / "audio" / video.replace(".mp4", ".mp3"), PATH / "unlabeled/audio" / video.replace(".mp4", ".mp3"))

if __name__ == "__main__":
    for folder in ("3-CNSordos", "4-Locufre"):
        print(f"\nProcessing folder: {folder}")
        process_folder(root_dir / "data" / "raw" / folder)