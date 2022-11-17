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
TEST_TEMPLATES_FOLDER = "./data/public_test/json_lyrics/"
SUBMISSION_FOLDER = "./submissions/"

torch.set_grad_enabled(False)
woptions = whisper.DecodingOptions(language="vi", without_timestamps=True)
wtokenizer = whisper.get_tokenizer(True, language="vi", task=woptions.task)

models = []
for name, ckpt_path in zip(["medium", "large"], ["wmedium.bin", "wlarge.bin"]):
    model = WhisperModel(name)
    checkpoint = torch.load(ckpt_path, "cpu")
    model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()
    models.append(model)
weights = [0.4, 0.6]


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
    word_idxs = [word_idxs]

    # out = model(input_ids, dec_input_ids)

    # out1 = models[0](input_ids, dec_input_ids)
    # out2 = models[1](input_ids, dec_input_ids)
    # starts = torch.stack([out1[..., 0], out2[..., 0]], -1)
    # starts = torch.max(starts, dim=-1)[0]
    # ends = torch.stack([out1[..., 1], out2[..., 1]], -1)
    # ends = torch.min(ends, dim=-1)[0]
    # out = torch.stack([starts, ends], -1)

    out = 0
    for weight, model in zip(weights, models):
        out += model(input_ids, dec_input_ids) * weight

    for i, res in enumerate(out):
        prediction = {}
        words_dict = defaultdict(list)
        word_ids_dict = defaultdict(list)

        for token_id, word_id, (s, e) in zip(dec_input_ids[i], word_idxs[i], res):
            token = wtokenizer.decode(token_id)
            prediction[str(word_id) + token] = [s.item(), e.item()]
            # print(token, s.item() * max_ms, e.item() * max_ms)
            words_dict[word_id].append(token)
            word_ids_dict[word_id].append(token_id)

        # print(prediction)
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

    # print(output_dict)
    # print(output_words)
    # print(output_times)
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

                # import pdb; pdb.set_trace()
                # raise Exception()

            chunk["s"] = s
            chunk["e"] = e

            if s < min_s:
                min_s = s
            if e > max_e:
                max_e = e

        segment["s"] = min_s
        segment["e"] = max_e

    # print(template)
    with open(os.path.join(SUBMISSION_FOLDER, name + ".json"), "w") as f:
        json.dump(template, f, indent=4)
    # break
