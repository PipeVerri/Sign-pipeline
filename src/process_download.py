import os
from pathlib import Path

root_dir = Path(__file__).parent.parent.resolve()
PATH = root_dir / "data" / "raw" / "3-CNSordos"

# Start by renaming all the subtitles to get rid of .es-x.vtt
subtitles = [f for f in os.listdir(PATH) if f.endswith(".vtt")]
for sub in subtitles:
    name = sub.split(".")[0] + ".vtt"
    os.rename(PATH / sub, PATH / name)

# Now delete the videos that dont have subtitles
os.makedirs(PATH / ".deleted", exist_ok=True)
videos = [f for f in os.listdir(PATH) if f.endswith(".mp4")]
for video in videos:
    sub = video.split(".")[0] + ".vtt"
    if not os.path.exists(PATH / sub):
        os.rename(PATH / video, PATH / ".deleted" / video)