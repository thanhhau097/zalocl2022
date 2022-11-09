from typing import Dict

import torch
import torch.nn as nn

import whisper


class CustomLoss(nn.Module):
    def forward(self, inputs, targets):
        return nn.functional.l1_loss(inputs[targets != -100], targets[targets != -100])


class WhisperModel(nn.Module):
    def __init__(self, model_name: str = "base"):
        super().__init__()

        self.model = whisper.load_model(model_name)
        # for p in self.model.encoder.parameters():
        #     p.requires_grad = False
        self.linear = nn.Sequential(nn.Linear(self.model.dims.n_text_state, 2), nn.Sigmoid())

    def forward(self, input_ids, dec_input_ids, starts=None, ends=None, word_idxs=None):
        audio_features = self.model.encoder(input_ids)
        decoder_features = self.model.decoder(dec_input_ids, audio_features)
        out = self.linear(decoder_features)
        return out
