import os
from collections import defaultdict
import json

dir_path = "gpt5-mini-not-good"
RESULTS_PATH = f'output_data/{dir_path}/results_seedprompt0.json'
ANALYSIS_PATH = f'output_data/{dir_path}/analysis/'
ANALYSIS_FILE_PATH = ANALYSIS_PATH + 'analysis_strategies.json'
os.makedirs(ANALYSIS_PATH, exist_ok=True)

with open(RESULTS_PATH) as f:
    results = json.load(f)

def add_avg_score_diff(d):
    for key, value in d.items():
        if isinstance(value, dict):
            # If this dict has score_diff, compute its average
            if "score_diff" in value and isinstance(value["score_diff"], list):
                sd = value["score_diff"]
                value["avg_score_diff"] = sum(sd) / len(sd) if sd else "None"
                value.pop("score_diff")

            # Recurse into nested dictionaries
            add_avg_score_diff(value)

def move_child_counts(data):
    def process_node(node: dict):
        # Find child dicts first (before touching their counts)
        child_items = [(k, v) for k, v in node.items() if isinstance(v, dict)]

        if child_items:
            # Parent: set its count from any child that has a 'count'
            child_with_count = next(
                (child for _, child in child_items if "count" in child),
                None
            )
            if child_with_count is not None:
                node["count"] = child_with_count["count"]
            else:
                # No child has a count -> remove parent's count if exists
                node.pop("count", None)

            # Now recurse into children
            for _, child in child_items:
                process_node(child)
        else:
            # Leaf node: remove its count
            node.pop("count", None)

    # Apply to each top-level value
    for _, v in data.items():
        if isinstance(v, dict):
            process_node(v)


dataset = []

for res in results:
    prompt_id = res["id"]
    prompt_parent_id = res["parent_id"]
    prompt = res["prompt"]
    queries = res["queries"]
    strategies_analysis = {}
    for query_tree in queries:
        id_to_query = {}
        for query in query_tree:
            query_id = query["id"]
            id_to_query[query_id] = query

            strategy = query["query_refinement_strategy"]
            score = query["score"]
            if strategy is None or score == "None":
                continue
            parent_score = id_to_query[query["parent_id"]]["score"]
            
            strategy_map = strategies_analysis
            
            cur_query = query
            path = []
            while cur_query["parent_id"] is not None:
                cur_query = id_to_query[cur_query["parent_id"]]
                if cur_query["query_refinement_strategy"] is not None:
                    path.append(cur_query["query_refinement_strategy"])
            for i in range(len(path)-1, -1, -1):
                strategy_map = strategy_map[path[i]]

            strategy_map[strategy] = strategy_map.get(strategy, {})
            strategy_map[strategy]["score_diff"] = strategy_map[strategy].get("score_diff", [])
            strategy_map[strategy]["count"] = strategy_map[strategy].get("count", 0)

            strategy_map[strategy]["count"] += 1
            strategy_map[strategy]["score_diff"].append(float(score) - float(parent_score))

    add_avg_score_diff(strategies_analysis)
    move_child_counts(strategies_analysis)
    row = {
        "id": prompt_id,
        "parent_id": prompt_parent_id,
        "prompt": prompt,
        "strategies_analysis": strategies_analysis,
    }


    dataset.append(row)

with open(ANALYSIS_FILE_PATH, 'w') as f:
    json.dump(dataset, f, indent=4)
