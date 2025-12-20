from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from gepa import optimize
from gepa.core.adapter import DataInst, EvaluationBatch, ProposalFn, RolloutOutput, Trajectory, GEPAAdapter
from gepa.core.state import GEPAState
from tqdm import tqdm
from src.aqd.budget import Budget
from src.config import Config
from src.gepa.gepa_reflect import refine_prompt
from src.aqd.query_refinement.evaluate_responses import evaluate_responses
from src.aqd.query_refinement.generate_response import generate_response
from src.aqd.schema import PromptNode, QueryNode
from src.aqd.prompt_search.generate_queries import generate_queries_baseline
from src.aqd.prompt_search.unpreferred_response_feedback import unpreferred_response_feedback
import json
import pickle
from pathlib import Path

def save_path(config: Config) -> Path:
    return config.output_path.with_name('gepa_run.pkl')

class EcommerceGEPAAdapater(GEPAAdapter):
    def __init__(self, config: Config):
        self.config = config
        self.prompt_log = []
        self.budget_log = {}
        self.num_prompts = 0
        self.prompt_log_path = config.output_path.with_name('gepa_prompt_log.json')
        self.budget_log_path = config.output_path.with_name('gepa_budget_log.json')
        self.save_path = save_path(config)
        self.prompt_to_node: Dict[str, PromptNode] = {}

    def save(self):
        self.config.budget.dump(self.budget_log_path)
        with self.prompt_log_path.open('w') as f:
          json.dump(self.prompt_log, f, indent=4)
        with self.save_path.open('wb') as f:
          pickle.dump(self, f)

    def propose_new_texts(
          self,
          candidate: dict[str, str],
          reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
          components_to_update: list[str],
      ) -> dict[str, str]:
          """
          GEPA's function of modifying prompts (no need to use reflective dataset since prompt node already stores
          reflections in the format we use)
          """
          print('refining prompt...')
          new_component = {'prompt': refine_prompt(
              self.prompt_to_node[candidate['prompt']],
              self.config.generate_model,
              self.config.budget,
              self.config.preferred_definition,
              self.config.unpreferred_definition,
              list(set(self.prompt_to_node.keys())-set([candidate['prompt']]))
          )}
          self.save()
          return new_component
    
    def _process_example_product(self, prompt_node: PromptNode, example: dict, idx: int):
        """function for processing product and resulting query for multithreading"""

        # print('generating query...')
        query_node = QueryNode(generate_queries_baseline(
            prompt_node,
            self.config.generate_model,
            self.config.budget,
            example
        ), 
        {
          'product': example['all_attributes'],
          'name': example['product_name'],
          'id': example['product_id']
        })

        # print('generating response...')
        query_node.response = generate_response(query_node, self.config.target_model, self.config.budget) 

        # print('evaluating response...')
        evaluation, reasoning, individual_evaluations = evaluate_responses(
          query_node = query_node,
          evaluation_model_paths = self.config.evaluation_models,
          generative_model_path = self.config.generate_model,
          budget = self.config.budget,
          preferred_definition = self.config.preferred_definition,
          unpreferred_definition = self.config.unpreferred_definition
        )

        query_node.evaluation = evaluation
        query_node.evaluation_reason = reasoning
        query_node.individual_evaluations = individual_evaluations
        query_node.get_score(self.config.preference_score_map)
        query_node.set_is_unpreferred(evaluation, self.config.unpreferred_definition)

        prompt_node.queries.append(query_node)

        # update logs
        log = [{
              "id": idx,
              "query": query_node.query,
              "query_generation_extra_info": {
                  'product': query_node.query_generation_extra_info['name'],
                  'attribute': query_node.query_generation_extra_info['product'],
                  'product_id': query_node.query_generation_extra_info['id']
              },
              'response': query_node.response,
              'individual_evaluations': query_node.individual_evaluations,
              'evaluation': query_node.evaluation,
              'score': query_node.score
        }]

        # turn score negative because GEPA optimizes towards higher score, but lower is better for us
        return query_node.query, -query_node.score, reasoning, log

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        """
        Creates scores and reflections for GEPA pipeline based on candidate
        Our traces are the evaluation reasonings from the model.
        We ignore capture_traces and always provide
        reasoning to ensure the GEPA baseline and our model operate on even footing, 
        allowing both to benefit from chain-of-thought.
        """
        # initializes return values GEPA expects
        outputs = []
        scores = []
        trajectories = []

        print(candidate)
        # check if prompt log already exists (e.g. during validation prompts are reused)
        new_prompt_log = None
        for log in self.prompt_log:
          if log['prompt'] == candidate['prompt']:
              new_prompt_log = log
        if not new_prompt_log:
            new_prompt_log = {'id': self.num_prompts, 'prompt': candidate['prompt'], 'queries': []}
            self.prompt_log.append(new_prompt_log)
            self.num_prompts += 1

        
        # creates prompt node
        prompt_node = PromptNode(candidate['prompt'])
        if candidate['prompt'] in self.prompt_to_node:
            prompt_node = self.prompt_to_node[candidate['prompt']]
        else:
            self.prompt_to_node[candidate['prompt']] = prompt_node

        # gets queries from prompt and logs information for output json
        query_logs = []
        with ThreadPoolExecutor(max_workers = self.config.n_threads) as pool:
          futures = [pool.submit(partial(self._process_example_product, prompt_node=prompt_node), example=example, idx=idx) 
                     for idx,example in enumerate(batch)]

          for fut in tqdm(as_completed(futures), total=len(futures), desc=f"GEPA Evaluation"):
              try:
                  output, score, reasoning, log = fut.result()
                  outputs.append(output)
                  scores.append(score)
                  trajectories.append(reasoning)
                  query_logs.append(log)
              except Exception as e:
                  pool.shutdown(wait=False, cancel_futures=True)
                  raise            

        # add queries to existing log and save
        new_prompt_log['queries'].extend(query_logs)
        self.save()

        return EvaluationBatch(outputs, scores, trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """
        Generates feedback for GEPA in accordance with its API
        This gets called before propose_new_texts
        """
        prompt_node = self.prompt_to_node[candidate['prompt']]
        print('generating feedback...')
        reasoning, suggestions = unpreferred_response_feedback(
            prompt_node,
            self.config.generate_model,
            self.config.budget,
            self.config.unpreferred_definition
        )
        prompt_node.unpreferred_response_reason = reasoning
        prompt_node.unpreferred_response_improvement_suggestions = suggestions

        self.save()

        return {
            'Inputs': candidate,
            'Generated Outputs': '\n'.join(eval_batch.outputs),
            'Feedback': '\n'.join(suggestions)
        }

def run_gepa(config: Config):
    # Load product dataset, TODO: put this in config
    all_data = config.products
    n = len(all_data)
    train_data = all_data[:n//2]
    val_data = all_data[n//2:]
    
    # load save
    if config.prev_result:
        with open(save_path(config), 'rb') as f:
          adapter = pickle.load(f)
    else:
        adapter = EcommerceGEPAAdapater(config)

    def stop_callback(gepa_state: GEPAState) -> bool:
        if gepa_state.i > config.n_prompt_refinement_iterations:
            return False
        return True

    # GEPA optimization process
    gepa_result = optimize(
        seed_candidate = {'prompt': config.initial_prompt[0]},
        trainset = train_data,
        valset = val_data,
        adapter = adapter,
        stop_callbacks=stop_callback,
        reflection_lm = config.generate_model,
        reflection_minibatch_size = n//4,
        run_dir = config.prev_result.parent if config.prev_result else None,
    )

    print(f'{config.n_prompt_refinement_iterations} Iterations reached... terminating')