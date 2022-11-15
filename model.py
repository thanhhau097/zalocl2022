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


def mask_logits(target, mask):
    return target * mask


class BalancedBCE(nn.Module):
    def forward(self, inputs, targets):
        loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        targets_inv = (targets != 1).float()
        loss_p = mask_logits(loss, targets).sum() / targets.sum()
        loss_n = mask_logits(loss, targets_inv).sum() / targets_inv.sum()
        loss = loss_p + loss_n
        return loss.mean()


class WhisperModel(nn.Module):
    def __init__(self, model_name: str = "base", bce_aux=False):
        super().__init__()

        self.model = whisper.load_model(model_name, device="cpu")
        self.linear = nn.Sequential(nn.Linear(self.model.dims.n_text_state, 2), nn.Sigmoid())
        self.bce_aux = bce_aux
        if self.bce_aux:
            self.linear_bce = nn.Linear(self.model.dims.n_text_state, 3000)

    def forward(
        self,
        input_ids,
        dec_input_ids,
        starts=None,
        ends=None,
        word_idxs=None,
        separated_multiclass=None,
    ):
        audio_features = self.model.encoder(input_ids)
        decoder_features = self.model.decoder(dec_input_ids, audio_features)
        out = self.linear(decoder_features)
        if self.bce_aux:
            out_bce = self.linear_bce(decoder_features)
            return out, out_bce
        return out
