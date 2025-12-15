
import json

dir_path = "lucy_res"
RESULTS_PATH = f'output_data/{dir_path}/results.json'
DATASET_PATH = f'output_data/{dir_path}/dataset.json'

with open(RESULTS_PATH) as f:
    results = json.load(f)

dataset = []

for res in results:
    prompt = res["prompt"]
    queries = res["queries"]
    for query_tree in queries:
        depth_map = {None: -1}
        for query in query_tree:
            depth = depth_map[query["parent_id"]] + 1
            depth_map[query["id"]] = depth

            row = {
                "prompt": prompt,
                "query": query["query"],
                "depth": depth
            }
            if query["is_filtered_out"] == True:
                row["is_filtered_out"] = query["is_filtered_out"]
                row["filtering_reason"] = query["filtering_reason"]
            else:
                row["response"] = query["response"]
                # row["evaluation"] = query["evaluation"]
                row["evaluation_reason"] = query["evaluation_reason"]
                row["score"] = query["score"]

            dataset.append(row)

with open(DATASET_PATH, 'w') as f:
    json.dump(dataset, f, indent=4)
