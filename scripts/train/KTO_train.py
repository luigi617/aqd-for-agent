import json
import collections
from typing import List, Tuple
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
)
from trl import KTOTrainer, KTOConfig
from transformers import TrainerCallback
from tqdm import tqdm
import random
import wandb
from peft import LoraConfig

from src.trainers.evaluator import EvaluatorConfig
from src.trainers.trainer_kto import CustomizedKTOTrainer
from src.trainers.arguments import ModelArguments, EvaluationArguments, DataArguments, KTOTrainingArguments, LoggerArguments
from src.aqd.query_refinement.evaluate_responses import get_preference_criteria_map, get_preference_criteria_to_key_map, get_preference_criteria_str, get_majority
from src.models.model import get_model
from src.aqd.query_refinement.evaluate_responses import get_schema, EVAL_PROMPT


def main(training_args, data_args, model_args, eval_args, logger_args):
    # 1. Setup Model & Tokenizer
    model_id = model_args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load Model in 4-bit (QLoRA) to save memory, as KTO uses more memory than SFT
    # (KTO effectively runs the forward pass twice: once for policy, once for reference)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_4bit=True, # Requires bitsandbytes
        torch_dtype=torch.float16,
    )

    train_raw = Dataset.from_dict(json.load(open(data_args.train_file)))
    eval_raw = Dataset.from_dict(json.load(open(data_args.eval_file)))
    test_raw = Dataset.from_dict(json.load(open(data_args.test_file)))

    # KTO requires the dataset to have columns: 'prompt', 'completion', 'label'
    def format_kto(example):
        # Chat template formatting
        prompt_messages = [{"role": "user", "content": example['query']}]
        
        # We strictly separate Prompt and Completion
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        
        # Map Score to Boolean Label
        # Score 2 -> True (Desirable)
        # Score 0 -> False (Undesirable)
        label = True if float(example['score']) == 2 else False
        
        return {
            "prompt": prompt_text,
            "completion": example['response'],
            "label": label
        }

    # Filter out Score 1 (Neutral) to keep the signal clean
    train_dataset = train_raw.map(format_kto) # train_dataset.data.fields('label') to check if the label is correct 
    eval_dataset = eval_raw.map(format_kto)
    test_dataset = test_raw.map(format_kto)

    eval_config = EvaluatorConfig(
        target_model=eval_args.target_model,
        evaluation_models=eval_args.evaluation_models,
        preferred_definition=eval_args.preferred_definition,
        unpreferred_definition=eval_args.unpreferred_definition,
        n_evaluation=eval_args.n_evaluation,
        batch_size=training_args.per_device_eval_batch_size
    )

    all_args = {
        'model_args': vars(model_args),
        'data_args': vars(data_args),
        'training_args': training_args.to_dict(),
    }

    run = wandb.init(
        project=logger_args.run_project,
        name=training_args.output_dir.split("/")[-1] or None,
        group=logger_args.run_group,
        config=all_args,
    )

    # LoRA adapters are required to fine-tune quantized (4-bit/8-bit) models
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    trainer = CustomizedKTOTrainer(
        model=model,
        ref_model=None, # None = create a copy of 'model' as reference automatically
        training_args=training_args,
        data_args=data_args,
        eval_config=eval_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if training_args.do_eval and not training_args.do_train:
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser(
        dataclass_types=(ModelArguments, DataArguments, KTOTrainingArguments, LoggerArguments, EvaluationArguments),
        description="KTO evaluation training script"
    )
    model_args, data_args, training_args, logger_args, eval_args = parser.parse_args_into_dataclasses()

    print('received model_args:')
    print(json.dumps(vars(model_args), indent=2, sort_keys=True))
    print('received data_args:')
    print(json.dumps(vars(data_args), indent=2, sort_keys=True))
    print('received eval_args:')
    print(json.dumps(vars(eval_args), indent=2, sort_keys=True))
    # print('received training_args:')
    # print(json.dumps(training_args.to_dict(), indent=2, sort_keys=True))
    print('received logger_args:')
    print(json.dumps(vars(logger_args), indent=2, sort_keys=True))
    print('--------------------------------')

    main(training_args, data_args, model_args, eval_args, logger_args)
