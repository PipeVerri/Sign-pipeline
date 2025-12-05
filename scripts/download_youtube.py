from pathlib import Path
from yt_dlp import YoutubeDL

root_dir = Path(__file__).parent.parent.resolve()
CHANNEL = "https://www.youtube.com/c/CanalesAsociaciónCivil/videos"
PATH = str(root_dir / "data" / "raw")

FAILED_LOG = root_dir / "failed_downloads.txt"

# 1. Extraer URLs del canal
def get_video_urls(channel_url):
    opts = {"extract_flat": True, "skip_download": True}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    return ["https://www.youtube.com/watch?v=" + e["id"] for e in info["entries"]]


# 2. Descargar cada video individualmente
def download_video(url):
    ydl_opts = {
        "format": (
            "bestvideo[height=240][vcodec!=none]+bestaudio/"
            "bestvideo[height>=240][vcodec!=none]+bestaudio/"
            "bestvideo[vcodec!=none][acodec=none]"
        ),
        "outtmpl": PATH + "/%(title)s.%(ext)s",
        "retries": 10,
        "fragment_retries": 10,
        "file_access_retries": 10,
        "socket_timeout": 30,
        "ignoreerrors": False,  # Necesario para detectar bien excepciones
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


# MAIN
if __name__ == "__main__":
    urls = get_video_urls(CHANNEL)

    print(f"Se encontraron {len(urls)} videos.")

    for url in urls:
        download_video(url)

    print("\n🎯 Terminado.")
    print(f"Las URLs fallidas quedaron guardadas en: {FAILED_LOG}")