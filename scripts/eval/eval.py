import argparse
import json
import os
import collections
import random

from scripts.eval.diversity_analysis import evaluate_diversity

random.seed(42)

def extract_data(args):
    """
    extract the data for evaluation
    """
    with open(args.product_file, "r") as f:
        products = json.load(f)
    valid_ids = set()
    valid_product_names = set()
    for product in products:
        valid_ids.add(product["product_id"])
        valid_product_names.add(product["product_name"])

    data_for_eval = []
    with open(args.input_file, "r") as f:
        data = json.load(f)
    for item in data:
        if not item.get("query_generation_extra_info").get("product_id"):
            if item.get("query_generation_extra_info").get("product") in valid_product_names:
                data_for_eval.append(item)
        else:
            if item["query_generation_extra_info"]["product_id"] in valid_ids:
                data_for_eval.append(item)

    budget_for_eval = []
    with open(args.budget_file, "r") as f:
        budget = json.load(f)
    if not budget.get("history"):
        budget["history"] = []
    # #### legacy code
    # for item in budget["history"]:
    #     if not item.get("budget_extra_info"):
    #         continue
    #     for product_name in valid_product_names:
    #         if product_name in item["budget_extra_info"]["desc"]:
    #             budget_for_eval.append(item)
    #             break
    # ####
    if not budget_for_eval:
        budget_for_eval = budget["history"]
    # # save the ids to a json file
    # with open(os.path.join(args.output_dir, "ours_ids.json"), "w") as f:
    #     json.dump(ours_products, f, indent=4)
    return data_for_eval, budget_for_eval


def calculate_statistics(args, data_for_eval, budget_for_eval):
    """
    calculate the statistics of the data: number of data points, number of products, number of queries per product, 
    metrics: percentage of unhelpful queries
    metrics: token cost per query
    """

    if args.task == "real":
        unhelpful_query_count = 0
        for item in data_for_eval:
            if item["label"] != "helpful":
                unhelpful_query_count += 1
        statistics = {
            "total_query_count": len(data_for_eval),
            "unhelpful_query_count": unhelpful_query_count,
            "percentage_of_unhelpful_queries": unhelpful_query_count / len(data_for_eval)
        }
        with open(os.path.join(args.output_dir, "statistics.json"), "w") as f:
            json.dump(statistics, f, indent=4)
        return
    
    statistics = {}
    total_query_count = len(data_for_eval)
    product_count = len(
        set(
            [
                item["query_generation_extra_info"].get("product_id") if item["query_generation_extra_info"].get("product_id") else item["query_generation_extra_info"].get("product")
                for item in data_for_eval
            ]
        )
    )
    query_per_product_count = total_query_count / product_count

    unhelpful_query_count = 0
    scored_query_count = 0
    for item in data_for_eval:
        if item["score"]:
            scored_query_count += 1
            if (round(item["score"]) == 0 or round(item["score"]) == 1):
                unhelpful_query_count += 1
    percentage_of_unhelpful_queries = unhelpful_query_count / scored_query_count

    completion_token_cost = 0
    prompt_token_cost = 0
    cached_prompt_token_cost = 0
    for item in budget_for_eval:
        completion_token_cost += item["completion_tokens"]
        prompt_token_cost += item["prompt_tokens"]
        cached_prompt_token_cost += item["cached_prompt_tokens"]
    average_completion_token_cost_per_query = completion_token_cost / total_query_count
    average_prompt_token_cost_per_query = prompt_token_cost / total_query_count
    average_cached_prompt_token_cost_per_query = cached_prompt_token_cost / total_query_count

    statistics["total_query_count"] = total_query_count
    statistics["product_count"] = product_count
    statistics["query_per_product_count"] = query_per_product_count
    statistics["percentage_of_unhelpful_queries"] = percentage_of_unhelpful_queries
    statistics["average_completion_token_cost_per_query"] = average_completion_token_cost_per_query
    statistics["average_prompt_token_cost_per_query"] = average_prompt_token_cost_per_query
    statistics["average_cached_prompt_token_cost_per_query"] = average_cached_prompt_token_cost_per_query
    with open(os.path.join(args.output_dir, "statistics.json"), "w") as f:
        json.dump(statistics, f, indent=4)


def extract_unhelpful_data(args, data_for_eval):
    """
    extract the unhelpful data from the data for evaluation
    """
    unhelpful_data = []
    if args.task == "real":
        extracted_data_unhelpful = []
        data_per_product = collections.defaultdict(list)
        for item in data_for_eval:
            if item.get("label") != "helpful":
                data_per_product[item.get("product_category")].append(item)
        for product_category, data in data_per_product.items():
            if len(data) > 25:
                extracted_data_unhelpful.extend(random.sample(data, 25))
            else:
                extracted_data_unhelpful.extend(data)
        print(f"Extracted {len(extracted_data_unhelpful)} unhelpful real data.")
        return extracted_data_unhelpful
    
    filtered_out_data_count = 0
    none_score_count = 0
    confidence_score = collections.Counter()
    length_distribution = collections.Counter()
    for item in data_for_eval:
        if args.task == "ours":
            if item.get("is_filtered_out"):
                filtered_out_data_count += 1
            else:
                confidence_score[item.get("filtering_confidence_score")] += 1
            if not item.get("is_filtered_out") and item.get("filtering_confidence_score") > 0.9:
                if item.get("score") is None:
                    none_score_count += 1
                    # breakpoint()
                if item.get("score") is not None and (round(item["score"]) == 0 or round(item["score"]) == 1):
                    length_distribution[len(item["query"].split())] += 1
                    unhelpful_data.append(item)
        else:
            if item.get("score") is not None and (round(item["score"]) == 0 or round(item["score"]) == 1):
                unhelpful_data.append(item)
    with open(os.path.join(args.output_dir, "unhelpful_data.json"), "w") as f:
        json.dump(unhelpful_data, f, indent=4)
    print(f"Extracted {len(unhelpful_data)} unhelpful data, {filtered_out_data_count} filtered out data, {confidence_score.most_common(10)}")
    print(f"None score count: {none_score_count}")
    print(f"Length distribution: {length_distribution.most_common(10)}")
    return unhelpful_data

def extract_1_data_per_product(args, data_for_eval):
    """
    extract 1 data per product from the unhelpful data
    """
    extracted_data = []
    data_per_product = collections.defaultdict(list)


    if args.task == "real":
        for item in data_for_eval:
            product_category = item.get("product_category")
            data_per_product[product_category].append(item)
        for product_category, data in data_per_product.items():
            if len(data) > 25:
                extracted_data.extend(random.sample(data, 25))
            else:
                extracted_data.extend(data)
    else:
        for item in data_for_eval:
            product_name = item.get("query_generation_extra_info").get("product") if item.get("query_generation_extra_info").get("product") else item.get("query_generation_extra_info").get("product_name")
            data_per_product[product_name].append(item)

        for product, data in data_per_product.items():
            if len(data) > 1:
                extracted_data.append(random.sample(data, 1)[0])
    print(f"Extracted {len(extracted_data)} data.")
    return extracted_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="/home/yunanlu/code/aqd-for-agent/output_data/gpt5-mini_1126/cleaned_data.json",
        help="path to the input file",
    )
    parser.add_argument(
        "--model_path", type=str, default="gpt-5-mini", help="path to the model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/yunanlu/code/aqd-for-agent/output_data/gpt5-mini_1126",
        help="path to the output file",
    )
    parser.add_argument(
        "--product_file",
        type=str,
        default="/home/yunanlu/code/aqd-for-agent/input_data/product_info_1.json",
        help="only use the data generated from the products in the product file",
    )
    parser.add_argument(
        "--budget_file",
        type=str,
        default="/local/data/shared/aqd/lucy/budget.json",
        help="path to the budget file",
    )
    parser.add_argument(
        "--embedding_model_path",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="path to the embedding model",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="ours",
        choices=["ours", "tap", "pair", "gepa", "real", "ace"],
        help="task to run",
    )
    parser.add_argument(
        "--data_sample",
        type=str,
        default="1_data_per_product",
        help="data sample to use",
        choices=["1_data_per_product", "unhelpful", "all"],
    )
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)
    if args.budget_file:
        with open(args.budget_file, "r") as f:
            budget = json.load(f)
    else:
        budget = {}

    # create the output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        print(f"Output directory {args.output_dir} already exists")

    if args.task == "ours":
        data_for_eval, budget_for_eval = extract_data(args)
    else:
        data_for_eval = data
        budget_for_eval = budget.get("history", [])

    if args.data_sample == "1_data_per_product":
        extracted_data = extract_unhelpful_data(args, data_for_eval)
        with open(os.path.join(args.output_dir, "unhelpful_data.json"), "w") as f:
            json.dump(extracted_data, f, indent=4)
        extracted_data = extract_1_data_per_product(args, extracted_data)
        # save the extracted data to a json file
        with open(os.path.join(args.output_dir, "extracted_data.json"), "w") as f:
            json.dump(extracted_data, f, indent=4)
    elif args.data_sample == "unhelpful":
        extracted_data = extract_unhelpful_data(args, data_for_eval)
        # save the extracted data to a json file
        with open(os.path.join(args.output_dir, "unhelpful_data.json"), "w") as f:
            json.dump(extracted_data, f, indent=4)
    elif args.data_sample == "all":
        extracted_data = data_for_eval
    # calculate statistics of the data
    calculate_statistics(args, data_for_eval, budget_for_eval)
    # evaluate percentage of unhelpful queries
    evaluate_diversity(args, extracted_data)


# ours data
"""
python -m scripts.eval.eval \
--input_file output_data/evaluation/gpt5/pqr/cleaned_dataset.json \
--output_dir output_data/evaluation/gpt5/pqr/eval_1_data_per_product \
--product_file '<your domain knowledge>' \
--embedding_model_path Qwen/Qwen3-Embedding-0.6B \
--budget_file output_data/gpt5/pqr/budget.json \
--task ours \
--data_sample 1_data_per_product
"""
