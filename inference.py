import json
import os
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

import whisper
from dataset import load_wave
from model import WhisperModel

SAMPLE_RATE = 16000
TEST_AUDIO_FOLDER = "./data/public_test/songs/"
TEST_LYRICS_FOLDER = "./data/public_test/lyrics/"
TEST_TEMPLATES_FOLDER = "./data/public_test/new_labels_json/"
SUBMISSION_FOLDER = "./data/submissions_merged_f5/"

torch.set_grad_enabled(False)
woptions = whisper.DecodingOptions(language="vi", without_timestamps=True)
wtokenizer = whisper.get_tokenizer(True, language="vi", task=woptions.task)
model = WhisperModel("large")
checkpoint = torch.load("wlarge_kfold_pseudo_spotify_merge_token_fold5/checkpoint-6000/pytorch_model.bin", "cpu")
model.load_state_dict(checkpoint)
model = model.cuda()
model.eval()


for audio_name in tqdm(os.listdir(TEST_AUDIO_FOLDER)):
    audio_path = os.path.join(TEST_AUDIO_FOLDER, audio_name)
    name, ext = os.path.splitext(audio_name)
    lyric_path = os.path.join(TEST_LYRICS_FOLDER, name + ".txt")
    template_path = os.path.join(TEST_TEMPLATES_FOLDER, name + ".json")

    # audio
    audio = load_wave(audio_path, sample_rate=SAMPLE_RATE)
    audio = whisper.pad_or_trim(audio.flatten())
    mel = whisper.log_mel_spectrogram(audio)

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
    with open(os.path.join(SUBMISSION_FOLDER, name + ".json"), "w") as f:
        json.dump(template, f, indent=4)
    # break