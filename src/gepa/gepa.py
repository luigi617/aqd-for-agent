from typing import Any, List, Mapping, Sequence, Tuple
from gepa import optimize
from gepa.core.adapter import DataInst, EvaluationBatch, ProposalFn, RolloutOutput, Trajectory, GEPAAdapter
from tqdm import tqdm
from src.aqd.budget import Budget
from src.config import Config
from src.aqd.query_refinement.evaluate_responses import evaluate_responses
from src.aqd.query_refinement.generate_response import generate_response
from src.aqd.schema import PromptNode, QueryNode
from src.aqd.prompt_search.generate_queries import generate_queries_baseline
from src.aqd.prompt_search.unpreferred_response_feedback import unpreferred_response_feedback
import json
from pathlib import Path

class EcommerceProposalFn(ProposalFn):
    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """
        - Given the current `candidate`, a reflective dataset (as returned by
          `GEPAAdapter.make_reflective_dataset`), and a list of component names to update,
          return a mapping component_name -> new component text (str). This allows the user
          to implement their own instruction proposal logic. For example, the user can use
          a different LLM, implement DSPy signatures, etc. Another example can be situations
          where 2 or more components need to be updated together (coupled updates).

        Returns
        - Dict[str, str] mapping component names to newly proposed component texts.
        """
        ...

class EcommerceGEPAAdapater(GEPAAdapter):
    def __init__(self, config: Config):
        self.config = config
        self.prompt_log = []
        self.budget_log = {}
        self.num_prompts = 0
        self.prompt_log_path = config.output_path.with_name('gepa_prompt_log.json')
        self.budget_log_path = config.output_path.with_name('gepa_budget_log.json')

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        """
        Run the program defined by `candidate` on a batch of data.

        Parameters
        - batch: list of task-specific inputs (DataInst).
        - candidate: mapping from component name -> component text. You must instantiate
          your full system with the component text for each component, and execute it on the batch.
        - capture_traces: when True, you must populate `EvaluationBatch.trajectories`
          with a per-example trajectory object that your `make_reflective_dataset` can
          later consume. When False, you may set trajectories=None to save time/memory.
          capture_traces=True is used by the reflective mutation proposer to build a reflective dataset.

        Returns
        - EvaluationBatch with:
          - outputs: raw per-example outputs (opaque to GEPA).
          - scores: per-example floats, length == len(batch). Higher is better.
          - trajectories:
              - if capture_traces=True: list[Trajectory] with length == len(batch).
              - if capture_traces=False: None.

        Scoring semantics
        - The engine uses sum(scores) on minibatches to decide whether to accept a
          candidate mutation and average(scores) over the full valset for tracking.
        - Prefer to return per-example scores, that can be aggregated via summation.
        - If an example fails (e.g., parse error), use a fallback score (e.g., 0.0).

        Correctness constraints
        - len(outputs) == len(scores) == len(batch)
        - If capture_traces=True: trajectories must be provided and len(trajectories) == len(batch)
        - Do not mutate `batch` or `candidate` in-place. Construct a fresh program
          instance or deep-copy as needed.
        """
        outputs = []
        scores = []
        trajectories = []

        print(candidate)
        new_prompt_log = None
        for log in self.prompt_log:
          if log['prompt'] == candidate['prompt']:
              new_prompt_log = log
        if not new_prompt_log:
            new_prompt_log = {'id': self.num_prompts, 'prompt': candidate['prompt'], 'queries': []}
            self.prompt_log.append(new_prompt_log)
            self.num_prompts += 1

        prompt_node = PromptNode(candidate['prompt'])

        query_logs = []
        num_queries = 0

        for example in tqdm(batch, total=len(batch), desc="Processing GEPA batch"):
            print('generating query...')
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

            print('generating response...')
            query_node.response = generate_response(query_node, self.config.target_model, self.config.budget) 

            print('evaluating response...')
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

            outputs.append(query_node.query)
            scores.append(query_node.score)
            trajectories.append(reasoning)

            # update logs
            query_logs.append([{
                  "id": num_queries,
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
            }])
            num_queries += 1

        new_prompt_log['queries'].extend(query_logs)
        json.dump(self.prompt_log, self.prompt_log_path.open('w'), indent=4)
        self.config.budget.dump(self.budget_log_path)

        return EvaluationBatch(outputs, scores, trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """
        Build a small, JSON-serializable dataset (per component) to drive instruction
        refinement by a teacher LLM.

        Parameters
        - candidate: the same candidate evaluated in evaluate().
        - eval_batch: The result of evaluate(..., capture_traces=True) on
          the same batch. You should extract everything you need from eval_batch.trajectories
          (and optionally outputs/scores) to assemble concise, high-signal examples.
        - components_to_update: subset of component names for which the proposer has
          requested updates. At a time, GEPA identifies a subset of components to update.

        Returns
        - A dict: component_name -> list of dict records (the "reflective dataset").
          Each record should be JSON-serializable and is passed verbatim to the
          instruction proposal prompt. A recommended schema is:
            {
              "Inputs": Dict[str, str],             # Minimal, clean view of the inputs to the component
              "Generated Outputs": Dict[str, str] | str,  # Model outputs or raw text
              "Feedback": str                       # Feedback on the component's performance, including correct answer, error messages, etc.
            }
          You may include additional keys (e.g., "score", "rationale", "trace_id") if useful.

        Determinism
        - If you subsample trace instances, use a seeded RNG to keep runs reproducible.
        """
        return {
            'Inputs': candidate,
            'Generated Outputs': '\n'.join(eval_batch.outputs),
            'Feedback': '\n'.join(eval_batch.trajectories)
        }

    propose_new_texts: ProposalFn | None = EcommerceProposalFn

def run_gepa(config: Config):
    # Load product dataset, TODO: put this in config
    all_data = json.load(Path('input_data/product_info_1.json').open())
    train_data = all_data[:10]
    val_data = all_data[10:13]

    adapter = EcommerceGEPAAdapater(config)

    # GEPA optimization process
    gepa_result = optimize(
        seed_candidate = {'prompt': config.initial_prompt},
        trainset = train_data,
        valset = val_data,
        adapter = adapter,
        max_metric_calls = 150,
        reflection_lm = config.generate_model,
        reflection_minibatch_size=20,
    )

    print("GEPA Optimized Prompt:", gepa_result.best_candidate['system_prompt'])