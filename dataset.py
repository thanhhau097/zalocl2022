import json
import os
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as at

import whisper
from whisper.tokenizer import Tokenizer


def get_audio_label_paths(audio_folder: str, label_folder: str) -> Tuple[List[str], List[str]]:
    audio_files = os.listdir(audio_folder)
    label_files = os.listdir(label_folder)

    audio_paths = []
    label_paths = []

    for file in audio_files:
        name, ext = os.path.splitext(file)
        if name + ".json" in label_files:
            audio_paths.append(os.path.join(audio_folder, file))
            label_paths.append(os.path.join(label_folder, name + ".json"))

    return audio_paths, label_paths


def load_wave(wave_path, sample_rate: int = 16000, augment=False) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    # if augment:
    #     transform = Compose(transforms=transforms)
    #     transformed_audio =  transform(audio)

    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform


class LyricDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        audio_paths: List[str],
        label_paths: List[str],
        tokenizer: Tokenizer,
        sample_rate: int,
        is_training: bool = False,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.is_training = is_training
        self.audio_paths, self.label_paths = audio_paths, label_paths

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label_path = self.label_paths[idx]

        # audio
        audio = load_wave(audio_path, sample_rate=self.sample_rate, augment=self.is_training)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)

        with open(label_path, "r") as f:
            label = json.load(f)

        words = []
        starts = []
        ends = []
        for segment in label:
            for ann in segment["l"]:
                words.append(ann["d"].lower())
                starts.append(ann["s"])
                ends.append(ann["e"])

        max_ms = 30000  # or len of audio file

        separated_tokens = []
        separated_starts = []
        separated_ends = []
        word_idxs = []

        for (word_idx, word), s, e in zip(enumerate(words), starts, ends):
            tokens = self.tokenizer.encode(word)
            word_idxs += [word_idx] * len(tokens)
            separated_tokens += tokens
            separated_starts += [s / max_ms] * len(tokens)
            separated_ends += [e / max_ms] * len(tokens)

        separated_tokens = separated_tokens
        starts = separated_starts
        ends = separated_ends
        return {
            "input_ids": mel,
            "dec_input_ids": separated_tokens,
            "starts": starts,
            "ends": ends,
            "word_idxs": word_idxs,
        }


class DataCollatorWithPadding:
    def __call__(sefl, features):
        input_ids, labels, dec_input_ids, word_idxs = [], [], [], []

        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(np.array([f["starts"], f["ends"]]).transpose())
            dec_input_ids.append(f["dec_input_ids"])
            word_idxs.append(f["word_idxs"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        word_idxs_length = [len(w) for w in word_idxs]
        max_label_len = max(label_lengths + dec_input_ids_length)

        labels = [
            np.concatenate([lab, np.ones((max(max_label_len - lab_len, 0), 2)) * -100])
            for lab, lab_len in zip(labels, label_lengths)
        ]
        dec_input_ids = [
            np.pad(e, (0, max_label_len - e_len), "constant", constant_values=50257)
            for e, e_len in zip(dec_input_ids, dec_input_ids_length)
        ]  # 50257 is eot token id
        word_idxs = [
            np.pad(w, (0, max_label_len - w_len), "constant", constant_values=-100)
            for w, w_len in zip(word_idxs, word_idxs_length)
        ]

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "input_ids": input_ids,
            "word_idxs": word_idxs,
        }
        batch = {k: torch.from_numpy(np.array(v)) for k, v in batch.items()}

        return batch
