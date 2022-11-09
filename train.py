import logging
import os
import sys
from functools import partial

import torch
import transformers
from sklearn.model_selection import train_test_split
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import AutoTokenizer

import whisper
from data_args import DataArguments
from dataset import DataCollatorWithPadding, LyricDataset, get_audio_label_paths
from engine import CustomTrainer, compute_metrics
from model import WhisperModel
from model_args import ModelArguments

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        # transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    # logger.info(f"Training/evaluation parameters {training_args}")
    # Set seed before initializing model.
    set_seed(training_args.seed)

    woptions = whisper.DecodingOptions(language="vi", without_timestamps=True)
    # wtokenizer = whisper.get_tokenizer(True, language="vi", task=woptions.task)
    wtokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-vi-en")
    audio_paths, label_paths = get_audio_label_paths(
        data_args.audio_folder, data_args.label_folder
    )
    train_audios, val_audios, train_labels, val_labels = train_test_split(
        audio_paths, label_paths, test_size=0.05, random_state=42
    )
    train_dataset = LyricDataset(
        train_audios, train_labels, wtokenizer, data_args.sample_rate, is_training=True
    )
    val_dataset = LyricDataset(val_audios, val_labels, wtokenizer, data_args.sample_rate)

    # Initialize trainer
    model = WhisperModel(model_args.model_name)
    if last_checkpoint is None and model_args.resume is not None:
        logger.info(f"Loading {model_args.resume} ...")
        checkpoint = torch.load(model_args.resume, "cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint.pop("state_dict")
        model.load_state_dict(checkpoint)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(),
        compute_metrics=partial(compute_metrics, wtokenizer=wtokenizer),
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        # max_train_samples = (
        #     data_args.max_train_samples
        #     if data_args.max_train_samples is not None
        #     else len(train_dataset)
        # )
        # metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        # max_val_samples = (
        #     data_args.max_val_samples
        #     if data_args.max_val_samples is not None
        #     else len(val_dataset)
        # )
        # metrics["eval_samples"] = min(max_val_samples, len(val_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
