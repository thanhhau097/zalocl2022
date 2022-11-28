from multiprocessing import Pool
import os
import subprocess
import pandas as pd

SAVE_FOLDER = "/home/anhph/youtube"


def download(url):
    video_id = url.split("watch?v=")[1].split("&")[0]
    if video_id + ".wav" in os.listdir(SAVE_FOLDER):
        return

    subprocess.run(f'youtube-dl -x --audio-format "wav" --audio-quality 0 -o "{SAVE_FOLDER}/%(id)s.%(ext)s" {url}', shell=True)

if __name__ == '__main__':
    df = pd.read_csv("music_youtube_url.csv", header=None)
    with Pool(32) as p:
        print(p.map(download, df[0].tolist()))
