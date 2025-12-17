from typing import Any, List, Mapping, Sequence, Tuple
from gepa import optimize
from gepa.core.adapter import DataInst, EvaluationBatch, ProposalFn, RolloutOutput, Trajectory, GEPAAdapter
from aqd.budget import Budget
from src.config import Config
from src.aqd.query_refinement.evaluate_responses import evaluate_responses
from src.aqd.schema import PromptNode, QueryNode
from src.aqd.prompt_search.generate_queries import generate_queries_w_attr
import json

class EcommerceGEPAAdapater(GEPAAdapter):
    def __init__(self, config: Config):
        self.config = config

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
        trajectories = []

        prompt_node = PromptNode(candidate['instruction_text'])

        for example in batch:
            generate_queries_w_attr(
                prompt_node,
                self.config.generate_model,
                self.config.budget,
                
            )

            chosen_criteria, reasoning, evaluations = evaluate_responses(
                query_node = QueryNode(),
                evaluation_model_paths = self.config.evaluation_models,
                generative_model_path = self.config.generate_model,
                budget = self.config.budget
                preferred_definition = self.config.preferred_definition,
                unpreferred_definition = self.config.unpreferred_definition
            )

            outputs.append(evaluations)
            trajectories.append(reasoning)

        return [trajectories, outputs]

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
        ...

    propose_new_texts: ProposalFn | None = None

def run_gepa(config: Config):
    # Load product dataset, TODO: put this in config
    trainset = json.load('input_data/product_info_1.json')

    adapter = EcommerceGEPAAdapater(config)

    # GEPA optimization process
    gepa_result = optimize(
        seed_candidate = config.initial_prompt,
        trainset = trainset,
        adapter = adapter,
        task_lm = config.generate_model,
        max_metric_calls = 150,
        reflection_lm = config.generate_model,
    )

    print("GEPA Optimized Prompt:", gepa_result.best_candidate['system_prompt'])
