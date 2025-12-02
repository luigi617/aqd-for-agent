from dataclasses import dataclass, field
from trl import KTOConfig
from typing import List

@dataclass
class LoggerArguments:
    run_project: str = field(
        default="aqd-for-agent",
        metadata={"help": "project name for wandb"},
    )
    run_group: str = field(
        default="kto",
        metadata={"help": "group name for wandb"},
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen3-8B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: str = field(
        default="/home/yunanlu/code/aqd-for-agent/data/train_dataset.json",
        metadata={"help": "Path for cached train dataset"},
    )
    eval_file: str = field(
        default='/home/yunanlu/code/aqd-for-agent/data/eval_dataset.json',
        metadata={"help": "Path for cached eval dataset"},
    )
    test_file: str = field(
        default='/home/yunanlu/code/aqd-for-agent/data/test_dataset.json',
        metadata={"help": "Path for cached test dataset"},
    )



@dataclass
class EvaluationArguments:
    target_model: str = field(
        default="gpt-5-mini",
        metadata={"help": "path to target model"},
    )
    evaluation_models: List[str] = field(
        default_factory=lambda: ['gpt-5-mini', 'gemini-2.5-flash', 'deepseek-reasoner'],
        metadata={"help": "paths to evaluation (local/api) models"},
    )
    preferred_definition: str = field(
        default="/local/data/shared/aqd/input_data/preferred_definition.json",
        metadata={"help": "path topreferred definition w/ criteria"},
    )
    unpreferred_definition: str = field(
        default="/local/data/shared/aqd/input_data/unpreferred_definition.json",
        metadata={"help": "path to unpreferred definition w/ criteria"},
    )
    n_evaluation: int = field(
        default=2,
        metadata={"help": "number of evaluations per response"},
    )

@dataclass
class KTOTrainingArguments(KTOConfig):
    output_dir: str = field(
        default="/home/yunanlu/code/aqd-for-agent/results/kto",
        metadata={"help": "Path for output directory"},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training."}
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run eval on the dev set."}
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy: steps or epoch"},
    )
    eval_steps: int = field(
        default=40,
        metadata={"help": "Evaluation steps if eval_strategy is steps"},
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "beta for KTO"},
    )
    desirable_weight: float = field(
        default=1.0,
        metadata={"help": "weight for desirable"},
    )
    undesirable_weight: float = field(
        default=1.0,
        metadata={"help": "weight for undesirable"},
    )
    learning_rate: float = field(
        default=5e-7,
        metadata={"help": "learning rate for KTO"},
    )
    per_device_train_batch_size: int = field(
        default=2,
        metadata={"help": "batch size for KTO"},
    )
    per_device_eval_batch_size: int = field(
        default=2,
        metadata={"help": "batch size for KTO evaluation"},
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "gradient accumulation steps for KTO"},
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "logging steps for KTO"},
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "max length for KTO"},
    )
    max_prompt_length: int = field(
        default=512,
        metadata={"help": "max prompt length for KTO"},
    )
