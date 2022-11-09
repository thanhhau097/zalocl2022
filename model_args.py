from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    """
    Arguments relating to model.
    """

    model_name: str = field(default="base", metadata={"help": "Whisper model name"})
