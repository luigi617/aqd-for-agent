from pipeline.misc.graph_common import GraphState, PromptState
from pathlib import Path
import json
from tqdm import tqdm
import time

def init_state(_: GraphState, pool: list[str], prev: dict) -> GraphState:
    """initializes if no previous otherwise update empty state with previous information"""
    
    state = GraphState.get_default()
    state["prompt_pool"] = pool

    if not prev: return state

    if isinstance(prev.get("iterations_passed"), int):
        state["iterations_passed"] = prev["iterations_passed"]
    if isinstance(prev.get("prompt_pool"), list):
        state["prompt_pool"] = list(prev["prompt_pool"])
    if isinstance(prev.get("prompts"), dict):
        state["prompts"] = dict(prev["prompts"])

    return state



def join_node(state: GraphState, output_path: Path, save_iter: bool=False):
    """used for joining parallel nodes"""
    with open(output_path, 'w') as f:
        json.dump(state, f, indent=4)
    if save_iter:
        save_iter_path = output_path.with_stem(output_path.stem + f'_iter{state["iterations_passed"]}')
        with open(save_iter_path, 'w') as f:
            json.dump(state, f, indent=4)
    return state

def stopping_condition(state: GraphState, n_iterations):
    """condition for edge that resets graph back to beginning after each iteration"""
    if state["iterations_passed"] >= n_iterations:
        return "end"
    return "continue"