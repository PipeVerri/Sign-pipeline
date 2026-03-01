from src.args import parse_args
import os
import yt_dlp

working, config = parse_args()

for source in config["download"]["sources"]:
    os.makedirs(working / "videos" / source["name"], exist_ok=True)
    ydl_opts = {
        "format": config["download"]["options"]["format"],
        "merge_output_format": "mp4",
        "subtitleslangs": config["download"]["options"]["sub_langs"],
        "outtmpl": str(working / "videos" / source["name"] / "%(id)s.%(ext)s"),
        "writesubtitles": True,
        "writeautomaticsub": source["auto_subs"],
        "remote_components": "ejs:github"
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        if source.get("url"):
            ydl.download([source["url"]])
        else:
            with open(working / source["path"], "r") as f:
                ydl.download(f.read().splitlines())