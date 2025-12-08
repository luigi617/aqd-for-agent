
import os
import random
import json

from dotenv import load_dotenv
from src.config import Config
from src.args_parser import argument_parser
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
    config: Config = argument_parser()
    config_path = config.output_path.with_name("config.json")
    save_result(config.to_dict(), config_path)

    if config.seed is not None:
        set_seed(config.seed)

    save_iter_path = config.output_path
    print(f"PROMPT OPTIMIZATION: START")
    try:
        ps = PromptSearch(
            config=config,
            base_depth_output_path=save_iter_path,
        )
        if config.prev_result:
            ps.load_prev_result(config.prev_result)
        prompt_search_res = ps.run(max_depth=config.n_prompt_refinemnet_iterations)
        prompt_search_res_json = ps.result_to_json(prompt_search_res)
        
        save_result(prompt_search_res_json, config.output_path)

        print("Spent:", config.budget.spent_usd, "remaining:", config.budget.remaining_usd)
    except Exception as e:
        print("Pipeline stopped:", e)
        prompt_search_res_json = ps.result_to_json(ps.root)
        error_save_path = save_iter_path.with_name("stopped_result.json")
        save_result(prompt_search_res_json, error_save_path)

    print(f"PROMPT OPTIMIZATION: DONE")

if __name__ == "__main__":
    main()




