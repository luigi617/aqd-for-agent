
import os
import random
import json

from dotenv import load_dotenv
from src.ace.ace import run_ace
from src.gepa.gepa import run_gepa
from src.config import Config
from src.args_parser import argument_parser
from src.aqd.prompt_search.prompt_search import PromptSearch
from src.tap.tap import TreeOfAttackWithPruning


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


def run_ours(config: Config):
    print(f"PROMPT OPTIMIZATION: START")
    save_iter_path = config.output_path
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


def run_tap(config: Config):
    print(f"Tree of Attack with Pruning: START")
    # The annotation mapping between baseline and our method is as follows:
    # 1. prompt P -> local queries q_l
    # 2. Query Q -> global query q_g
    # 3. Target model T -> qwen3-8B
    # 4. Off-topic evaluator: Off-Topic(P,Q), decide whether the q_l is semantically different from q_g -> off-topic evaluator
    # 5. Judge Evaluator: Judge(Q, R), decide whether the response R break the criteria for query Q. -> response evaluator
    tap = TreeOfAttackWithPruning(
        config=config
    )
    tap_res = tap.run()
    tap_res_json = tap.result_to_json(tap_res)
    save_result(tap_res_json, config.output_path)
    print(f"Tree of Attack with Pruning: DONE")



def run_pair(config: Config):
    raise NotImplementedError("Pair is not implemented yet")

def main():
    load_dotenv(override=True)

    # Parse arguments
    config: Config = argument_parser()
    config_path = config.output_path.with_name("config.json")
    save_result(config.to_dict(), config_path)

    if config.seed is not None:
        set_seed(config.seed)

    if config.task == "ours":
        run_ours(config)
    elif config.task == "tap":
        run_tap(config)
    elif config.task == "pair":
        run_pair(config)
    elif config.task == "gepa":
        run_gepa(config)
    elif config.task == "ace":
        run_ace(config)
    else:
        raise ValueError(f"Invalid task: {config.task}")

    
if __name__ == "__main__":
    main()




