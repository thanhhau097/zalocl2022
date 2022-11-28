from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments relating to data.
    """

    audio_folder: str = field(default="data/train/songs/", metadata={"help": "Audio folder"})
    label_folder: str = field(
        default="data/train/labels/",
        metadata={"help": "Label folder"},
    )
    sample_rate: int = field(default=16000, metadata={"help": "Audio sample rate"})
    max_length: int = field(default=480000, metadata={"help": "Audio max length"})
