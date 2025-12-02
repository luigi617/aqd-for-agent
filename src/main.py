
import os
import random
import json
from pathlib import Path

from dotenv import load_dotenv
from src.aqd.budget import BudgetExceededError, Budget
from src.args_parser import Argument, argument_parser
from src.aqd.query_refinement.query_refinement import QueryRefinement
from src.aqd.schema import QueryNode
from src.aqd.prompt_search.prompt_search import PromptSearch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch (if available).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    print(f"✅ Random seed set to {seed}")

def save_result(result: list, output_path):
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Save prompt search result to {output_path}")

def main():
    load_dotenv(override=True)

    # Parse arguments
    args: Argument = argument_parser()
    config_path = args.output_path.with_name("config.json")
    save_result(args.config, config_path)

    if args.seed is not None:
        set_seed(args.seed)

    preferred_definition = {}
    unpreferred_definition = {}
    preference_score_map = {}
    for criterion, info in args.preferred_definition.items():
        preferred_definition[criterion] = info["description"]
        preference_score_map[criterion] = info["score"]

    for criterion, info in args.unpreferred_definition.items():
        unpreferred_definition[criterion] = info["description"]
        preference_score_map[criterion] = info["score"]

    budget = None
    if isinstance(args.budget, float) and args.budget >= 0:
        budget = Budget(limit_usd=args.budget)


    save_iter_path = args.output_path
    print(f"PROMPT OPTIMIZATION: START")
    try:
        ps = PromptSearch(
            prompt=args.initial_prompt,
            products=args.products,
            n_queries=args.n_queries,
            generate_model=args.generate_model,
            refinement_model=args.refinement_model,
            target_model=args.target_model,
            filtering_model=args.filtering_model,
            evaluation_models=args.evaluation_models,
            preferred_definition=preferred_definition,
            unpreferred_definition=unpreferred_definition,
            preference_score_map=preference_score_map,
            filtering_policy=args.filtering_policy,
            query_refinement_strategies=args.query_refinement_strategies,

            n_prompt_refinemnet_children=args.n_prompt_refinemnet_children,
            n_prompt_refinemnet_beam_search_size=args.n_prompt_refinemnet_beam_search_size,
            n_query_refinemnet_iterations=args.n_query_refinemnet_iterations,
            n_query_refiment_beam_search_size=args.n_query_refiment_beam_search_size,
            n_thread_queries=args.n_thread_queries,
            n_thread_query_refiment=args.n_thread_query_refiment,
            
            base_depth_output_path=save_iter_path,
            budget=budget,
        )
        if args.prev_result:
            ps.load_prev_result(args.prev_result)
        prompt_search_res = ps.run(max_depth=args.n_prompt_refinemnet_iterations)
        prompt_search_res_json = ps.result_to_json(prompt_search_res)
        
        save_result(prompt_search_res_json, save_iter_path)

        print("Spent:", budget.spent_usd, "remaining:", budget.remaining_usd)
    except Exception as e:
        print("Pipeline stopped:", e)
        prompt_search_res_json = ps.result_to_json(ps.root)
        error_save_path = save_iter_path.with_name("stopped_result.json")
        save_result(prompt_search_res_json, error_save_path)

    print(f"PROMPT OPTIMIZATION: DONE")

if __name__ == "__main__":
    main()




