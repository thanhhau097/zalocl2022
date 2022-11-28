import logging
import os
import sys
from functools import partial

import torch
import transformers
from sklearn.model_selection import train_test_split, KFold
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process
import pandas as pd

import whisper
from data_args import DataArguments
from dataset import DataCollatorWithPadding, LyricDataset, get_audio_label_paths
from engine import CustomTrainer, compute_metrics
from model import WhisperModel
from model_args import ModelArguments

logger = logging.getLogger(__name__)

def freeze_model(frozen_layers, model):
    """
    Freeze some or all parts of the model.
    """
    if len(frozen_layers) > 0:
        for name, parameter in model.named_parameters():
            if any([name.startswith(layer) for layer in frozen_layers]):
                print(f"{name} is freezed")
                parameter.requires_grad_(False)

def main(current_fold):
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    training_args.output_dir = training_args.output_dir + f'_fold{current_fold}'
    # Pseudo folds
    model_args.resume = model_args.resume + f'_fold{current_fold}/'
    if current_fold == 7:
        model_args.resume += 'checkpoint-10584/'
    elif current_fold == 9:
        model_args.resume += 'checkpoint-10836/'
    elif current_fold == 5:
        model_args.resume += 'checkpoint-10710/'
    elif current_fold == 2:
        model_args.resume += 'checkpoint-10710/'
    elif current_fold == 15:
        model_args.resume += 'checkpoint-10206/'
    model_args.resume += 'pytorch_model.bin'

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
    wtokenizer = whisper.get_tokenizer(True, language="vi", task=woptions.task)
    
    df = pd.read_csv("data/train.csv")
    df.lyrics = df.lyrics.apply(lambda x: x.replace('data/train/labels/', 'data/train/labels_2/'))
    train_audios = df[df.fold != current_fold].audio.tolist()
    train_labels = df[df.fold != current_fold].lyrics.tolist()
    # train_audios = df.audio.tolist()
    # train_labels = df.lyrics.tolist()
    val_audios = df[df.fold == current_fold].audio.tolist()
    val_labels = df[df.fold == current_fold].lyrics.tolist()
    # audio_paths, label_paths = get_audio_label_paths(
    #     data_args.audio_folder, data_args.label_folder
    # )
    # train_audios, val_audios, train_labels, val_labels = train_test_split(
    #     audio_paths, label_paths, test_size=0.05, random_state=42
    # )

    print(f"ZALO Dataset FOLD{current_fold}: Train {len(train_audios)}, Val {len(val_audios)}")
    
    # audio_paths_2, label_paths_2 = get_audio_label_paths(
    #     "data/DALI_V1/song_segments/", "data/DALI_V1/labels/"
    # )
    # print(f"DALI Dataset: Train {len(audio_paths_2)}")

    audio_paths_2, label_paths_2 = get_audio_label_paths(
        "data/segments/", "data/spotify_segment_pseudo_result_2/"
    )
    print(f"Pseudo Dataset: Train {len(audio_paths_2)}")
    train_audios  = train_audios + audio_paths_2
    train_labels = train_labels + label_paths_2

    # train_audios  = audio_paths_2
    # train_labels = label_paths_2
    train_dataset = LyricDataset(
        train_audios, train_labels, wtokenizer, data_args.sample_rate, is_training=True
    )
    val_dataset = LyricDataset(val_audios, val_labels, wtokenizer, data_args.sample_rate)

    # Initialize trainer
    model = WhisperModel(model_args.model_name, bce_aux=False)
    frozen_layers = ['model.encoder.conv1', 'model.encoder.conv2'] + [f'model.encoder.blocks.{i}.' for i in range(15)]
    # frozen_layers = ['model.']
    freeze_model(frozen_layers, model)
    if last_checkpoint is None and model_args.resume is not None:
        logger.info(f"Loading {model_args.resume} ...")
        checkpoint = torch.load(model_args.resume, "cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint.pop("state_dict")
        model.load_state_dict(checkpoint, strict=False)
        # model.load_state_dict(checkpoint)
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
    # for i in [7, 9, 5, 2]:
    for i in [15]:
    # for i in [5, 9, 7, 2]:
        main(i)
    # main(0)