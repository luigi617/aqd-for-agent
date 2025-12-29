# convert the TAP/PAIR baseline data to the format of our raw dataset
import argparse
import json
import os
import collections

def reformat_gepa_data(data):
    for iteration_content in data:
        iteration_content["parent_id"] = None
        iteration_content["improvement_strategy"] = None
        iteration_content["improvement_reason"] = None
        products_stat = collections.defaultdict(list)
        for queries in iteration_content["queries"]:
            if len(queries) != 1:
                raise ValueError("queries length is not 1")
            query_content = queries[0]
            products_stat[query_content["query_generation_extra_info"]["product"]].append(query_content)
        iteration_content["queries"] = []
        for product, queries in products_stat.items():
            iteration_content["queries"].append(queries)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="output_data/gpt5-mini_1218/tap/results.json", help="path to the input file")
    parser.add_argument("--output_file", type=str, default="output_data/gpt5-mini_1218/tap/formatted_results.json", help="path to the output file")
    parser.add_argument("--task", type=str, default="tap", choices=["tap", "pair", "gepa"], help="task to run")
    args = parser.parse_args()

    # make directory if not exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.input_file, "r") as f:
        data = json.load(f)
    
    if args.task == "gepa":
        final_dataset = reformat_gepa_data(data)
    else:
        final_dataset = [
            {
                "id": 0,
                "parent_id": None,
                "improvement_strategy": None,
                "improvement_reason": None,
                "prompt": "Create a user question related to online shopping that a customer might ask about a product.",
                "queries": data,
            }
        ]

    with open(args.output_file, "w") as f:
        json.dump(final_dataset, f, indent=4)

'''
python scripts/data/baseline_data_preprocess.py \
--input_file output_data/gpt5-mini_1218/tap/results.json \
--output_file output_data/gpt5-mini_1218/tap/formatted_results.json
'''

'''
python scripts/data/baseline_data_preprocess.py \
--input_file output_data/pair/results.json \
--output_file output_data/gpt5-mini_1218/pair/formatted_results.json
'''

'''
python scripts/data/baseline_data_preprocess.py \
--input_file output_data/gepa/full_run/gepa_prompt_log.json \
--output_file output_data/gpt5-mini_1218/gepa/formatted_results.json \
--task gepa
'''