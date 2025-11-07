
import os
import random
import json
from pathlib import Path

from dotenv import load_dotenv
from args_parser import argument_parser
from aqd.query_refinement.query_refinement import QueryRefinement
from aqd.schema import QueryNode


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

    target_model = args.target_model
    filtering_model = args.generate_model
    refinement_model = args.generate_model
    evaluation_model = args.generate_model

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    qn = QueryNode("what is the price of the vacuum?", products[0], ["manufacturer_suggested_retail_price"])
    a = QueryRefinement(
        qn,
        target_model,
        filtering_model,
        refinement_model,
        evaluation_model,
        preferred_definition,
        unpreferred_definition,
        filtering_policy
    )
    a.run(max_depth=1)


if __name__ == "__main__":
    main()




