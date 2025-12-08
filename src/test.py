from yt_dlp import YoutubeDL

url = "https://www.youtube.com/watch?v=QhNk5NNEbaA"

ydl_opts = {
    # don't download yet — just inspect
    "skip_download": True,
    "quiet": False,     # show progress / debug info
    "restrictfilenames": False,
}

with YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    print("keys in info:", list(info.keys()))
    print("subtitles:", info.get("subtitles"))
    print("automatic_captions:", info.get("automatic_captions"))
