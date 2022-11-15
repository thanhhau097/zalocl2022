from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments relating to model.
    """

    model_name: str = field(default="base", metadata={"help": "Whisper model name"})
    bce_aux: Optional[bool] = field(default=False, metadata={"help": "Use auxiliary BCE loss"})
    resume: Optional[str] = field(default=None, metadata={"help": "Path of model checkpoint"})
