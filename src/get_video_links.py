# For scraping from videolibros
import requests
from pathlib import Path
import time
from bs4 import BeautifulSoup

root_dir = Path(__file__).parent.parent.resolve()
videos_ids = set()

def save_videos():
    with open(root_dir / "data" / "info" / "videolibros_private.txt", "w") as f:
        for id in videos_ids:
            f.write(f"https://www.youtube.com/watch?v={id}" + "\n")

for i in range(106, 0, -1):
    res = requests.get(f'https://www.videolibros.org/video/{i}')
    if res.status_code == 500:  # Reached book limit
        break

    soup = BeautifulSoup(res.text, 'html.parser')
    for iframe in soup.find_all('iframe'):
        src = iframe.get('src')
        if src and ("youtube.com" in src):
            video_id = src.split("/embed/")[1].split("?")[0]
            videos_ids.add(video_id)
            print(f"Got video {i} with src {video_id}")

    save_videos()
    time.sleep(1)