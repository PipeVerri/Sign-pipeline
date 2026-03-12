from utils.args import parse_args
import os
import yt_dlp
from tqdm import tqdm
working, config = parse_args()
for source in config["sources"]:
    os.makedirs(working / "videos" / source["name"], exist_ok=True)
    # Get total video count before downloading
    print(f"\nFetching video list for: {source['name']}...")

    flat_opts = {"quiet": True, "extract_flat": True, "ignoreerrors": True}
    browser = config["options"]["download"].get("cookies_from_browser")
    if browser is not None:
        flat_opts["cookies_from_browser"] = (browser, None, None, None)

    total = 0
    if source.get("url"):
        urls = source["url"]
        with yt_dlp.YoutubeDL(flat_opts) as ydl:
            info = ydl.extract_info(source["url"], download=False)
            if info:
                entries = info.get("entries")
                total += len(list(entries) if entries else 1)
    else:
        with open(working / source["path"], "r") as f:
            urls = f.read().split("\n")
            total += len(urls)
    
    state = {"completed": 0}
    # Pre-mark already downloaded videos in the progress bar
    archive_path = working / "downloaded.txt"
    already_done = 0
    if archive_path.exists():
        with open(archive_path) as f:
            already_done = min(sum(1 for _ in f), total)
    pbar = tqdm(
        total=total,
        initial=already_done,   # Start the bar at already-downloaded count
        unit="video",
        desc=f"{source['name']}",
        colour="green",
        dynamic_ncols=True,
    )
    def postprocessor_hook(d, state=state, pbar=pbar):
        # MoveFiles fires exactly once per completed video, after merging
        if d["status"] == "finished" and d["postprocessor"] == "MoveFiles":
            state["completed"] += 1
            title = d.get("info_dict", {}).get("title", "")
            pbar.update(1)
            pbar.set_postfix_str(title[:50])
    ydl_opts = {
        "format": config["options"]["download"]["format"],
        "merge_output_format": "mp4",
        "subtitleslangs": config["options"]["download"]["sub_langs"],
        "outtmpl": str(working / "videos" / source["name"] / "%(id)s.%(ext)s"),
        "writesubtitles": True,
        "writeautomaticsub": source["auto_subs"],
        "remote_components": ["ejs:github"],
        "download_archive": str(archive_path),
        "postprocessor_hooks": [postprocessor_hook],
        "quiet": True,
        "ignoreerrors": True,
    }
    if browser is not None:
        ydl_opts["cookies_from_browser"] = (browser, None, None, None)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(urls)
    pbar.close()
    print(f"{source['name']}: {state['completed']} downloaded, {already_done} already existed, {total} total\n")
