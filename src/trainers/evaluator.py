import json
from dataclasses import dataclass, fields, field
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from tqdm.auto import tqdm
from collections import defaultdict

import math
import torch

from src.aqd.budget import Budget
from src.aqd.schema import QueryNode
from src.aqd.query_refinement.query_refinement import QueryRefinement

@dataclass
class EvaluatorConfig:
    """controls what metrics to compute during evaluation, as well as other eval related model configs
    """
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
    batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for evaluation"},
    )
    response_score: bool = field(
        default=True,
        metadata={"help": "Whether to compute score for response"},
    )

class RunningMetic(ABC):
    """this metric computes a score for each bach input (using update), and returns an overall score (using compute) at the very end e.g., by averaging

    Args:
        ABC (_type_): _description_
    """
    @abstractmethod
    def update(self, *args, **kwargs):
        return
    
    @abstractmethod
    def compute(self):
        return
    
    @abstractmethod
    def reset(self):
        return


class ScoreMetric(RunningMetic):
    def __init__(self, name="score"):
        self.name = name
        self.state = {
            "scores": []
        }
        self.reset()
        return
    
    def update(self, eval_batch, config: EvaluatorConfig):
        scores = []
        for example in eval_batch:
            score = self._compute(example, config)
            self.state["scores"].append(score)
            scores.append(score)
        return scores
    
    def _compute(self, data, config: EvaluatorConfig):

        response = data["response"]
        preferred_definition_raw = json.load(open(config.preferred_definition))
        unpreferred_definition_raw = json.load(open(config.unpreferred_definition))
        preferred_definition = {}
        unpreferred_definition = {}
        preference_score_map = {}
        for criterion, info in preferred_definition_raw.items():
            preferred_definition[criterion] = info["description"]
            preference_score_map[criterion] = info["score"]

        for criterion, info in unpreferred_definition_raw.items():
            unpreferred_definition[criterion] = info["description"]
            preference_score_map[criterion] = info["score"]

        budget = Budget(10)
        query_node = QueryNode(query=response, product=data["product_attributes"], attributes=data["selected_attributes"])

        query_refinement = QueryRefinement(
            query_node=query_node,
            target_model=config.target_model,
            filtering_model=config.target_model,
            refinement_model=config.target_model,
            evaluation_models=config.evaluation_models,
            preferred_definition=preferred_definition,
            unpreferred_definition=unpreferred_definition,
            preference_score_map=preference_score_map,
            filtering_policy=None,
            query_refinement_strategies=None,
            n_query_refiment_beam_search_size=2,
            budget=budget,
        )
        query_node = query_refinement.evaluate_responses(query_node)
        score = query_node.get_score(preference_score_map)
        return score
    
    def compute(self):
        return sum(self.state["scores"]) / len(self.state["scores"])
    
    def reset(self):
        self.state["scores"] = []
        return

METRICS = {
    "score": ScoreMetric
}


class Evaluator(ABC):
    def __init__(
        self,
        config: EvaluatorConfig,
        test_data: List[Dict],
        tokenizer
    ):
        self.config = config
        self.test_data = test_data
        self.tokenizer = tokenizer
        return
    
    def _get_data_iterator(self):
        batch = defaultdict(list)
        batch['__len__'] = 0
        for example in self.test_data:
            for k, v in example.items():
                batch[k].append(v)
            batch['__len__'] += 1
            if batch['__len__'] == self.config.batch_size:
                yield batch
                batch = defaultdict(list)
                batch['__len__'] = 0
        if batch['__len__'] > 0:
            yield batch

    def _flatten_performance(self, metric, prefix, metric_type):
        metric_performance = {}
        score = metric.compute()
        if isinstance(score, dict):
            for k, v in score.items():
                metric_performance[f"{prefix}_{metric_type}/{metric.name}/{k}"] = v
        else:
            metric_performance[f"{prefix}_{metric_type}/{metric.name}"] = score
        return metric_performance

    @abstractmethod
    def evaluate(self, wrapped_model, prefix='eval') -> Tuple[Dict[str, Any], List[Dict]]:
        raise NotImplementedError
    

class ResponseEvaluator(Evaluator):
    def __init__(self,
        config: EvaluatorConfig,
        test_data: List[Dict],
        tokenizer
        ):
        super().__init__(config, test_data, tokenizer)
        self.response_metrics = self.init_metrics(metric_type="response")
        return
    
    def init_metrics(self, metric_type):
        metrics = []
        for f in fields(self.config):
            if f.name.startswith(f"{metric_type}_") and getattr(self.config, f.name):
                metric_name = f.name.replace(f"{metric_type}_", "")
                metric = METRICS[metric_name]()
                metrics.append(metric)
        return metrics
    
    def reset_all_metrics(self):
        for metric in self.response_metrics:
            metric.reset()
        return

    def compute_performance(self, prefix):
        performance = {}
        for metric in self.response_metrics:
            # flattent so that trainer can still track nested metrics
            metric_performance = self._flatten_performance(metric, prefix, metric_type="response")
            performance.update(metric_performance)
        return performance

    def evaluate(self, wrapped_model, prefix='eval') -> Tuple[Dict[str, Any], List[Dict]]:
        test_data_iterator = self._get_data_iterator()
        self.reset_all_metrics()

        performance = {}
        predictions = []
        
        num_batches = math.ceil(len(self.test_data) / self.config.batch_size)
        for batch in tqdm(test_data_iterator, total=num_batches, desc="Evaluating"):
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                for q in batch["query"]
            ]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
            # Move tensors to the same device as the wrapped model
            model_device = next(wrapped_model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            with torch.no_grad():
                # Align activation dtype with model weights to avoid dtype mismatches (e.g., float vs bfloat16)
                param_dtypes = {p.dtype for p in wrapped_model.parameters() if p.is_floating_point()}
                preferred_dtype = torch.bfloat16 if torch.bfloat16 in param_dtypes else (torch.float16 if torch.float16 in param_dtypes else None)
                if model_device.type == "cuda" and preferred_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=preferred_dtype):
                        outputs = wrapped_model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                        )
                else:
                    outputs = wrapped_model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                    )
                generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
                batch_responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # batch_responses: List[str] = wrapped_model.generate(batch["prompt"])
            # reformat the batch
            # Build a list of example dicts, skipping the special '__len__' key
            batch_size = batch['__len__']
            eval_batch = [
                {k: v[i] for k, v in batch.items() if k != '__len__'}
                for i in range(batch_size)
            ]
            for i in range(batch_size):
                eval_batch[i]["response"] = batch_responses[i]

            for metric in self.response_metrics:
                scores = metric.update(eval_batch, self.config)
                for i in range(len(eval_batch)):
                    eval_batch[i]["score"] = scores[i]
            
            predictions.extend(eval_batch)

        performance = self.compute_performance(prefix)
        return performance, predictions
