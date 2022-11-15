from typing import Dict

import torch
import torch.nn as nn

import whisper


class CustomLoss(nn.Module):
    def forward(self, inputs, targets, word_idxs=None):
        return nn.functional.l1_loss(inputs[targets != -100], targets[targets != -100])
        losses = []
        for i in range(inputs.shape[0]):
            word_idx = word_idxs[i]
            input_i = inputs[i][word_idx != -100]
            target = targets[i][word_idx != -100]
            word_idx = word_idx[word_idx != -100]
            unique_idxs = torch.unique(word_idx)

            start = torch.stack([input_i[word_idx == idx][:, 0].min() for idx in unique_idxs])
            end = torch.stack([input_i[word_idx == idx][:, 1].max() for idx in unique_idxs])
            pred_word_timestamps = torch.stack([start, end], dim=1)
            gt_word_timestamps = torch.stack([target[word_idx == idx][-1] for idx in unique_idxs])
            losses.append(nn.functional.l1_loss(pred_word_timestamps, gt_word_timestamps))
        return torch.stack(losses).mean()


class WhisperModel(nn.Module):
    def __init__(self, model_name: str = "base"):
        super().__init__()

        self.model = whisper.load_model(model_name, device="cpu")
        # for p in self.model.encoder.parameters():
        #     p.requires_grad = False
        self.linear = nn.Sequential(nn.Linear(self.model.dims.n_text_state, 2), nn.Sigmoid())

    def forward(self, input_ids, dec_input_ids, starts=None, ends=None, word_idxs=None):
        audio_features = self.model.encoder(input_ids)
        decoder_features = self.model.decoder(dec_input_ids, audio_features)
        out = self.linear(decoder_features)
        return out
