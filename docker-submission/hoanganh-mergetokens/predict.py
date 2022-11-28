import argparse

import json
import os
from collections import defaultdict

import numpy as np
import torch
from time import time as current_time
from tqdm import tqdm

import whisper
from dataset import load_wave
from model import WhisperModel

parser = argparse.ArgumentParser()
SAMPLE_RATE = 16000

parser.add_argument('--data_folder', help='data folder', default="/data")
parser.add_argument('--weights_folder', help='data folder', default="./weights")
args = parser.parse_args()

TEST_AUDIO_FOLDER = os.path.join(args.data_folder, "songs")
TEST_TEMPLATES_FOLDER = os.path.join(args.data_folder, "json_lyrics")
SUBMISSION_FOLDER = "./data/submissions_{}/"

torch.set_grad_enabled(False)
woptions = whisper.DecodingOptions(language="vi", without_timestamps=True)
wtokenizer = whisper.get_tokenizer(True, language="vi", task=woptions.task)

print("WEIGHTS FOLDER", args.weights_folder)
dirs = args.weights_folder
checkpoints = os.listdir(dirs)
print("CHECKPOINTS:", checkpoints)

time_dict = defaultdict(lambda: 0)

for checkpoint_path in checkpoints:
    os.makedirs(SUBMISSION_FOLDER.format(checkpoint_path), exist_ok=True)
    weight = f"{dirs}/{checkpoint_path}"
    
    models = []
    alpha = 1.0

    model = WhisperModel("large")
    checkpoint = torch.load(weight, "cpu")
    model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()

    first_time = True
    for audio_name in tqdm(os.listdir(TEST_AUDIO_FOLDER)):
        audio_path = os.path.join(TEST_AUDIO_FOLDER, audio_name)
        name, ext = os.path.splitext(audio_name)
        template_path = os.path.join(TEST_TEMPLATES_FOLDER, name + ".json")

        # audio
        start = current_time()
        audio = load_wave(audio_path, sample_rate=SAMPLE_RATE)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)
        preprocessing_time = current_time() - start

        with open(template_path, "r") as f:
            template = json.load(f)

        words = []
        for segment in template:
            for chunk in segment["l"]:
                words.append(chunk["d"].lower())

        max_ms = 30000  # or len of audio file

        separated_tokens = []
        word_idxs = []

        for (word_idx, word) in enumerate(words):
            tokens = wtokenizer.encode(word)
            word_idxs += [word_idx] * len(tokens)
            separated_tokens += tokens

        separated_tokens = separated_tokens

        input_ids = torch.from_numpy(np.array(mel)).unsqueeze(0).cuda()
        dec_input_ids = torch.from_numpy(np.array(separated_tokens)).unsqueeze(0).cuda()
        word_idxs = torch.from_numpy(np.array(word_idxs)).unsqueeze(0).cuda()

        if first_time:
            model(input_ids, dec_input_ids, word_idxs)
            first_time = False

        model_start = current_time()
        out = model(input_ids, dec_input_ids, word_idxs)
        
        # word output
        out = out[0].tolist()
        for segment in template:
            for chunk in segment["l"]:
                s, e = out.pop(0)
                word = words.pop(0)
                if word != chunk["d"].lower():
                    print("Error:", chunk["d"])
                    continue

                chunk["s"] = int(s * 30000)
                chunk["e"] = int(e * 30000)

        # print(template)
        end = current_time()
        with open(os.path.join(SUBMISSION_FOLDER.format(checkpoint_path), name + ".json"), "w") as f:
            json.dump(template, f, indent=4)

        time_dict[audio_name] += end - model_start + preprocessing_time / len(checkpoints)
        print(audio_name, time_dict[audio_name])

with open("./data/time_submission.json", "w") as f:
    json.dump(time_dict, f)
