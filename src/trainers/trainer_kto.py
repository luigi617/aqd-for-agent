import inspect
import os
import json
import random
import textwrap
import warnings
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState, logging
from accelerate.utils import tqdm
from datasets import Dataset, concatenate_datasets
from torch import autocast
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    TrainingArguments,
    is_comet_available,
    is_wandb_available,
)
from transformers.trainer_utils import EvalLoopOutput, has_length
from transformers.utils import is_peft_available
from trl import KTOTrainer, KTOConfig

from src.trainers.arguments import KTOTrainingArguments, DataArguments
from src.trainers.evaluator import ResponseEvaluator, EvaluatorConfig



class CustomizedKTOTrainer(KTOTrainer):
    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str = None,
        ref_model: PreTrainedModel | nn.Module | str | None = None,
        training_args: KTOTrainingArguments = None,
        data_args: DataArguments = None,
        eval_config: EvaluatorConfig = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        data_collator: DataCollator | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: dict | None = None,
        compute_metrics: Callable[[EvalLoopOutput], dict] | None = None,
        model_adapter_name: str | None = None,
        ref_adapter_name: str | None = None,
    ):
        super().__init__(
            model=model,
            ref_model=None, # None = create a copy of 'model' as reference automatically
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            peft_config=peft_config,
        )
        self.evaluator_config = eval_config
        self.data_args = data_args
        self.tokenizer = processing_class
        return
    
    # def wrap_model_for_eval(
    #     self,
    #     documents,
    #     embeddings,
    #     index_path,
    # ) -> FaissRetriever:
    #     wrapped_model = FaissRetriever(
    #         documents,
    #         embeddings=embeddings,
    #         index_path=index_path
    #     )
    #     return wrapped_model
    
    # def _load_all_docs(self, document_path) -> List[Document]:
    #     with open(document_path, "rb") as f:
    #         documents = pickle.load(f)
    #     return documents
    
    def _load_eval_data(self, eval_data_path) -> List[Dict]:
        eval_data = json.load(open(eval_data_path))
        final_data = []
        # change the {"key1": ["value1", "value2", "value3"], "key2": ["value4", "value5", "value6"]} to [{"key1": "value1", "key2": "value4"}, {"key1": "value2", "key2": "value5"}, {"key1": "value3", "key2": "value6"}]
        for key, values in eval_data.items():
            if not final_data:
                final_data = [{key: value} for value in values]
            else:
                final_data = [dict(final_data[i], **{key: value}) for i, value in enumerate(values)]

        # just use 2 samples for evaluation
        final_data = final_data[:2]
        return final_data
    
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only = None,
        ignore_keys = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        is_test_only_mode = (not self.args.do_train) and self.args.do_eval

        if is_test_only_mode:
            model = self.model
            output = EvalLoopOutput(predictions=[], label_ids=None, metrics={}, num_samples=len(dataloader.dataset))
        else:
            model = self._wrap_model(self.model, training=False, dataloader=dataloader)
            output = super().evaluation_loop(
                dataloader,
                description=description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix
            )

        loaded_eval_data = self._load_eval_data(self.data_args.eval_file)
        
        evaluator = ResponseEvaluator(
            config=self.evaluator_config,
            test_data=loaded_eval_data,
            tokenizer=self.tokenizer
        )
        performance, predictions = evaluator.evaluate(wrapped_model=model, prefix=metric_key_prefix)
        output.metrics.update(performance)

        if self.args.write_predictions:
            save_name = f'step-{self.state.global_step}-predictions.json'
            save_path = os.path.join(self.args.output_dir, save_name)
            json.dump(predictions, open(save_path, 'w'), indent=4)
        return output
    
    
    # def _save(self, output_dir: Optional[str] = None, state_dict=None):
    #     TRAINING_ARGS_NAME = "training_args.bin"
    #     # If we are executing this function, we are the process zero, so we don't check for that.
    #     output_dir = output_dir if output_dir is not None else self.args.output_dir
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     self.model.save_pretrained(
    #         output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
    #     )

    #     if self.tokenizer is not None:
    #         self.tokenizer.save_pretrained(output_dir)

    #     # Good practice: save your training arguments together with the trained model
    #     torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    #     # save to sentence transformers
    #     word_embedding_model = models.Transformer(output_dir)
    #     pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=self.args.pooling_type)
    #     model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
    #     model.save(output_dir)