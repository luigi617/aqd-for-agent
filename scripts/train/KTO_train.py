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


class GenerativeJudgeCallback(TrainerCallback):
    def __init__(self, eval_args, eval_dataset, tokenizer, num_samples=50):
        self.eval_args = eval_args
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.num_samples = num_samples
    
    def evaluate_responses(
        self,
        query,
        response,
        preferred_definition,
        unpreferred_definition,
        preference_score_map,
        evaluation_model_paths,
        n_evaluation = 2,
    ) -> float:
        """
        Evaluate response
        """
        preference_criteria_map = get_preference_criteria_map(preferred_definition, unpreferred_definition)
        preference_criteria_to_key_map = get_preference_criteria_to_key_map(preference_criteria_map)
        preference_criteria_str = get_preference_criteria_str(preference_criteria_map)
        

        evaluations = []
        for model_path in evaluation_model_paths:
            evaluate_response_chain = get_model(model_path, get_schema(preference_criteria_map), retries=3)
            
            # get violated criteria
            for _ in range(n_evaluation):
                evaluation = evaluate_response_chain(
                    EVAL_PROMPT.format(
                        query=query,
                        response=response,
                        criteria=preference_criteria_str,
                    )
                )
                evaluations.append(evaluation)

        criteria_count = collections.defaultdict(int)
        for evaluation in evaluations:
            for criterion in evaluation['criteria']:
                criteria_count[criterion] += 1
        preference_criterion_to_category_map = {}
        for category, criteria in {**preferred_definition, **unpreferred_definition}.items():
            for criterion in criteria:
                preference_criterion_to_category_map[criterion] = category
        
        chosen_criteria = []
        threshold = get_majority(len(evaluation_model_paths), n_evaluation)
        for criterion, count in criteria_count.items():
            if count >= threshold:
                category = preference_criterion_to_category_map[criterion]
                chosen_criteria.append((category, criterion))
        
        score = sum(preference_score_map[criterion] for criterion, _ in chosen_criteria) / len(chosen_criteria)
        return score

    def external_evaluator(self, query, response):
        """
        Returns a score: 0, 1, or 2
        """
        preferred_definition = {}
        unpreferred_definition = {}
        preference_score_map = {}
        for criterion, info in self.eval_args.preferred_definition.items():
            preferred_definition[criterion] = info["description"]
            preference_score_map[criterion] = info["score"]

        for criterion, info in self.eval_args.unpreferred_definition.items():
            unpreferred_definition[criterion] = info["description"]
            preference_score_map[criterion] = info["score"]

        return self.evaluate_responses(query, response, preferred_definition, unpreferred_definition, preference_score_map, self.eval_args.evaluation_models, self.eval_args.n_evaluation)

    def on_evaluate(self, args, state, control, model, **kwargs):
        """
        Event called after the standard evaluation phase.
        We will hijack this to run our generation & scoring.
        """
        print("\n--- Running Generative Evaluation with External Judge ---")
        
        # 1. Select a random subset to save time
        # We ensure we don't pick more than the dataset size
        n = min(self.num_samples, len(self.eval_dataset))
        indices = random.sample(range(len(self.eval_dataset)), n)
        subset = self.eval_dataset.select(indices)
        
        scores = []
        
        # 2. Put model in eval mode
        model.eval()
        
        # 3. Batched generation and decoding
        batch_size = 2
        for start in tqdm(range(0, len(subset), batch_size), desc="Generating & Scoring"):
            end = min(start + batch_size, len(subset))
            batch_examples = [subset[i] for i in range(start, end)]
            batch_prompts = [ex["prompt"] for ex in batch_examples]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                )
                generated_ids = outputs[:, inputs.input_ids.shape[1]:]
                batch_generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
            # 4. Score with External Evaluator
            for prompt_text, generated_text in zip(batch_prompts, batch_generated_texts):
                score = self.external_evaluator(prompt_text, generated_text)
                scores.append(score)

        # 5. Calculate Average
        avg_score = sum(scores) / len(scores)
        print(f"\n>>> Average Generation Score: {avg_score:.4f} / 2.0\n")

        # 6. Log to WandB / Tensorboard
        # We access the existing _log function of the trainer from kwargs if available
        # Or simpler: just append to the state's log history if you want straightforward logging
        # Ideally, use: trainer.log({"eval_gen_score": avg_score}) inside the loop, 
        # but inside a Callback, we can modify the logs indirectly or print them.
        
        # NOTE: To log properly to WandB from a callback:
        if state.is_world_process_zero:
             # This injects the metric into the trainer's logs
             # (Requires you to pass 'trainer' to this callback or access it differently)
             pass 

        # Return model to train mode
        model.train()


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

    judge_callback = GenerativeJudgeCallback(
        eval_args=eval_args,
        eval_dataset=eval_dataset, 
        tokenizer=tokenizer,
        num_samples=50 # Only evaluate 50 samples per epoch to save time
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
        callbacks=[judge_callback],
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
