
import os
import random
import json
from pathlib import Path

from aqd.query_refinement.filter_queries import filter_queries
from aqd.query_refinement.misc_nodes import init_state, join_node, stopping_condition
from aqd.query_refinement.generate_queries import generate_queries
from aqd.query_refinement.generate_response import generate_response
from aqd.query_refinement.evaluate_responses import evaluate_responses
from aqd.query_refinement.modify_responses import modify_responses
from aqd.query_refinement.reflective_mutation_workflow import reflective_mutation
from aqd.query_refinement.refine_prompt_pool import refine_prompt_pool

from dotenv import load_dotenv
from args_parser import argument_parser
from aqd.query_refinement.query_refinement import QueryRefinement


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

def main():
    load_dotenv(override=True)

    # Parse arguments
    args = argument_parser()

    if args.seed is not None:
        set_seed(args.seed)

    initial_prompt_pool = json.load(args.input_pool.open())
    prev_results = json.load(args.from_prev.open()) if args.from_prev and Path.exists(args.from_prev) else {}
    products = json.load(args.products.open())
    preferred_definition = json.load(args.preferred_definition.open())
    unpreferred_definition = json.load(args.unpreferred_definition.open())
    filtering_policy = args.filtering_policy.open().read()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    QueryRefinement("what is the price of the tv?")


if __name__ == "__main__":
    main()




