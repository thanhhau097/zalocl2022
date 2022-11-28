from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

from model import CustomLoss, WhisperModel


class CustomTrainer(Trainer):
    def compute_loss(self, model: WhisperModel, inputs: Dict, return_outputs=False):
        outputs = model(inputs["input_ids"], inputs["dec_input_ids"], inputs["word_idxs"])
        loss_fct = CustomLoss()
        
        labels = inputs.get("labels")
        loss = loss_fct(outputs, labels)
        if return_outputs:
            return (loss, outputs)
        return loss

    def create_optimizer(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)
        outputs = nested_detach(outputs)
        inputs.pop("input_ids")
        return loss, outputs, inputs


def compute_metrics(eval_preds, wtokenizer):
    ious = compute_word_iou(
        eval_preds.predictions,
        wtokenizer,
        eval_preds.label_ids["dec_input_ids"],
        eval_preds.label_ids["word_idxs"],
        eval_preds.label_ids["labels"],
    )
    return {"IoU": np.mean(ious)}


def compute_word_iou(out, wtokenizer, dec_input_ids, word_idxs, labels):
    batch_ious = []
    for i, res in enumerate(out):
        max_idx = word_idxs[i].max() + 1
        ious = []
        for pred, gt_ in zip(out[i][:max_idx], labels[i][:max_idx]):
            if (max(pred[1], gt_[1]) - min(pred[0], gt_[0])) == 0:
                iou = 0
            else:
                iou = (min(pred[1], gt_[1]) - max(pred[0], gt_[0])) / (
                    max(pred[1], gt_[1]) - min(pred[0], gt_[0])
                )
                iou = np.clip(iou, 0, 1)
            ious.append(iou)
        batch_ious.append(np.mean(ious))

    return batch_ious
