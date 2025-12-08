from yt_dlp import YoutubeDL
from pathlib import Path
import re

root_dir = Path(__file__).parent.parent.resolve()

def batch_download(urls, PATH, FAILED_LOG, check_subs=False):
    def has_spanish_subs(url):
        ydl_opts = {
            "skip_download": True
        }

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        subs = info.get("subtitles", {})
        return any(re.match(r'^es', lang) for lang in subs.keys())

    def download_video(url, index):
        ydl_opts = {
            "format": (
                "bestvideo[height<=240][ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/"
                "bestvideo[height<=360][ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/"
                "best[height<=240][ext=mp4]/"
                "best[ext=mp4]"
            ),
            "outtmpl": str(PATH) + f"%(autonumber)s.%(ext)s",
            "retries": 10,
            "fragment_retries": 10,
            "file_access_retries": 10,
            "socket_timeout": 30,
            "ignoreerrors": False,  # Necesario para detectar bien excepciones,
            "writesubtitles": check_subs,
            "subtitleslangs": ["es.*"],
            "allsubtitles": False,
            "subtitlesformat": "vtt",
            "remotecomponents": "ejs:github"
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return True

        except Exception as e:
            print(f"❌ Error en: {url}")
            print("   →", e)

            # Guardar en archivo
            with open(FAILED_LOG, "a", encoding="utf8") as f:
                f.write(url + "\n")

            return False

    with open(PATH / "archive.txt", "a+") as f:
        downloaded = f.read().split("\n")
        for idx, url in enumerate(urls):
            if not url in downloaded:
                if check_subs:
                    if has_spanish_subs(url):
                        download_video(url, idx)
                else:
                    download_video(url, idx)
                downloaded.append(url)
                #f.write(url + "\n")

def fetch_urls_from_channel(url):
    ydl_opts = {
        "skip_download": True,
        "extract_flat": True
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return [entry["url"] for entry in info["entries"]]

if __name__ == "__main__":
    #with open(root_dir / "data" / "info" / "videolibros_private.txt", "r") as f:
    #    urls = f.read().split("\n")

    urls = fetch_urls_from_channel("https://www.youtube.com/@CNSORDOSARGENTINA/videos")

    batch_download(
        ["https://www.youtube.com/@CNSORDOSARGENTINA/videos"],
        root_dir / "data" / "raw" / "3-CNSordos",
        root_dir / "failed_downloads.txt",
        check_subs=False,
    )