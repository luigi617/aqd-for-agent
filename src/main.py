
import os
import random
import json
from pathlib import Path

from dotenv import load_dotenv
from args_parser import Argument, argument_parser
from aqd.query_refinement.query_refinement import QueryRefinement
from aqd.schema import QueryNode
from aqd.prompt_search.prompt_search import PromptSearch


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
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)

def main():
    load_dotenv(override=True)

    # Parse arguments
    args: Argument = argument_parser()

    if args.seed is not None:
        set_seed(args.seed)

    result = []
    for index, init_prompt in enumerate(args.initial_prompt_pool):
        ps = PromptSearch(
            prompt=init_prompt,
            products=args.products,
            n_queries=args.n_queries,
            generate_model=args.generate_model,
            refinement_model=args.refinement_model,
            target_model=args.target_model,
            filtering_model=args.filtering_model,
            evaluation_model=args.evaluation_model,
            preferred_definition=args.preferred_definition,
            unpreferred_definition=args.unpreferred_definition,
            filtering_policy=args.filtering_policy,

            n_prompt_refinemnet_children=args.n_prompt_refinemnet_children,
            n_query_refiment_children=args.n_query_refiment_children,
            n_thread_queries=args.n_thread_queries,
            n_thread_query_refiment=args.n_thread_query_refiment,
        )

        prompt_search_res = ps.run(max_depth=args.n_prompt_refinemnet_iterations)
        prompt_search_res_json = ps.result_to_json(prompt_search_res)
        save_iter_path = args.output_path.with_stem(args.output_path.stem + f'_iter{index}')
        save_result(prompt_search_res_json, save_iter_path)

        result.extend(prompt_search_res_json)

    save_result(result, args.output_path)

if __name__ == "__main__":
    main()




