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
    checkpoint = torch.load(f"{weight}", "cpu")
    model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()

    for audio_name in tqdm(os.listdir(TEST_AUDIO_FOLDER)):
        audio_path = os.path.join(TEST_AUDIO_FOLDER, audio_name)
        name, ext = os.path.splitext(audio_name)
        template_path = os.path.join(TEST_TEMPLATES_FOLDER, name + ".json")

        start = current_time()
        audio = load_wave(audio_path, sample_rate=SAMPLE_RATE, augment=False)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)

        preprocessing_time = current_time() - start

        out = 0
        # audio
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
        word_idxs = [word_idxs]

        model_start = current_time()
        out += model(input_ids, dec_input_ids)
        print("model prediction time", current_time() - model_start)

        for i, res in enumerate(out):
            prediction = {}
            words_dict = defaultdict(list)
            word_ids_dict = defaultdict(list)

            for token_id, word_id, (s, e) in zip(dec_input_ids[i], word_idxs[i], res):
                token = wtokenizer.decode(token_id)
                prediction[str(word_id) + token] = [s.item(), e.item()]
                words_dict[word_id].append(token)
                word_ids_dict[word_id].append(token_id)

            words = list(words_dict.values())
            word_ids = list(word_ids_dict.values())
            output_dict = {}
            output_words = []
            output_times = []

            for wid, word_tokens, word_id in zip(words_dict.keys(), words_dict.values(), word_ids):
                starts = []
                ends = []

                pred = [float("inf"), 0]
                for token in word_tokens:
                    # prediction
                    s, e = prediction[str(wid) + token]
                    # print(s, e)
                    if s < pred[0]:
                        pred[0] = s
                    if e > pred[1]:
                        pred[1] = e

                # word = "".join(word_tokens)
                word = wtokenizer.decode(word_id)
                time = [int(pred[0] * max_ms), int(pred[1] * max_ms)]
                # print(word, time)
                output_dict[word] = time
                output_words.append(word)
                output_times.append(time)

        # convert output_dict to submission format
        for segment in template:
            min_s, max_e = float("inf"), 0
            for chunk in segment["l"]:
                s, e = output_times.pop(0)
                word = output_words.pop(0)

                if word != chunk["d"].lower():
                    chunk_words = chunk["d"].split()
                    try:
                        chunk_word = chunk_words.pop(0)
                    except:
                        output_words.insert(0, word)
                        break
                    print("Error:", chunk["d"])
                    while chunk_words:
                        try:
                            word = output_words.pop(0)
                            last_s, last_e = output_times.pop(0)
                            chunk_word = chunk_words.pop(0)
                            e = last_e
                            if chunk_word != word:
                                import pdb

                                pdb.set_trace()
                        except:
                            break

                chunk["s"] = s
                chunk["e"] = e

                if s < min_s:
                    min_s = s
                if e > max_e:
                    max_e = e

            segment["s"] = min_s
            segment["e"] = max_e

        end = current_time()
        with open(os.path.join(SUBMISSION_FOLDER.format(checkpoint_path), name + ".json"), "w") as f:
            json.dump(template, f, indent=4)

        time_dict[audio_name] += end - model_start + preprocessing_time / len(checkpoints)
        print(audio_name, time_dict[audio_name])


with open("./data/time_submission.json", "w") as f:
    json.dump(time_dict, f)