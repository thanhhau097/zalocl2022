import datetime
import gc
import json
import os
from glob import glob

import IPython
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from joblib import Parallel, delayed
from tqdm import tqdm

import whisper
from dataset import get_audio_label_paths, load_wave
from stable_whisper import (
    group_word_timestamps,
    modify_model,
    results_to_word_srt,
    stabilize_timestamps,
)


def generate_word_timestamps(device_rank, world_size):
    dist.init_process_group(
        "nccl",
        init_method="env://",
        timeout=datetime.timedelta(0, 1800),
        rank=device_rank,
        world_size=world_size,
    )
    model = whisper.load_model("large")
    modify_model(model)
    model.to(f"cuda:{device_rank}")

    audio_paths = glob("/home/thanh/shared_disk/thanh/chunks/*.wav")
    audio_paths = audio_paths[device_rank::world_size]
    if device_rank == 0:
        audio_paths = tqdm(audio_paths)

    for audio_path in audio_paths:
        try:
            audio_fname = audio_path.split("/")[-1][:-4]
            label_path = os.path.join(
                "/home/thanh/shared_disk/thanh/chunk_labels", audio_fname + ".json"
            )
            cache_path = os.path.join(
                "/home/thanh/shared_disk/thanh/cache/", audio_fname + ".json"
            )
            results = model.transcribe(audio_path, language="vi", stab=True)
            json.dump(results, open(cache_path, "w"), ensure_ascii=False)

            results = group_word_timestamps(
                results, combine_compound=True, ts_key="word_timestamps"
            )

            labels = []

            for res in results:
                text = res["text"].strip()
                words = text.split(" ")
                timestamps = np.linspace(res["start"], res["end"], len(words) + 1)

                for i in range(len(timestamps) - 1):
                    label = {}
                    label["s"] = int(timestamps[i] * 1000)
                    label["e"] = int(timestamps[i + 1] * 1000)
                    label["d"] = words[i]
                    labels.append(label)

            json.dump(
                [{"s": 0, "e": 30000, "l": labels}],
                open(label_path, "w"),
                indent=4,
                ensure_ascii=False,
            )
            print(f"{audio_fname} processed successfully")
        except:
            print(f"Cannot generate word timestamps for {audio_fname}")


def main():
    world_size = 2
    mp.spawn(generate_word_timestamps, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
