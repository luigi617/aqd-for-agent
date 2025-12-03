import os
import json
import numpy as np
import copy
import argparse
from Levenshtein import distance
from scipy import stats


def edit_ratio(a, b):
    """
    Compute edit distance ratio = dist / avg(len(a), len(b)).
    Returns 0 when both strings are empty.
    """
    dist = distance(a, b)
    avg_len = (len(a) + len(b)) / 2
    return dist / avg_len if avg_len > 0 else 0


def group_strings(filtered_data_items, threshold=0.2):
    """
    Group responses by normalized edit distance threshold.
    Within each group, the first encountered ID is used as the representative.
    Returns (groups, reduced_ids)
      - groups: list of lists of IDs that are similar
      - reduced_ids: list of representative IDs (first ID of each group)
    """
    groups = []
    used = set()

    for i, s in enumerate(filtered_data_items["queries"]):
        if i in used:
            continue

        current_id = s["id"]
        response_l = s.get("response", "") or ""
        group = [current_id]
        used.add(i)

        for j in range(i + 1, len(filtered_data_items["queries"])):  # compare with subsequent items
            if j in used:
                continue
            compared_item = filtered_data_items["queries"][j]
            response_content = compared_item.get("response", "") or ""
            response_id = compared_item["id"]

            # if no response content, mark as used and skip
            if not response_content:
                used.add(j)
                continue

            if edit_ratio(response_l, response_content) < threshold:
                group.append(response_id)
                used.add(j)

        groups.append(group)

    reduced = [g[0] for g in groups]
    return groups, reduced


def reformat(clean_dataset, products_filepath):
    """add product attributes and selected attributes to the clean dataset

    Args:
        clean_dataset (list): The clean dataset
    """
    final_dataset = []
    with open(products_filepath, "r") as f:
        products = json.load(f)

    # create a name to attribute mapping
    name_to_attribute = {}
    for product in products:
        name_to_attribute[product["product_name"]] = product

    for idx, item in enumerate(clean_dataset):
        new_item = {}
        new_item["id"] = idx
        new_item["global_id"] = item["global_id"]
        new_item["query"] = item["query"]
        new_item["response"] = item["response"]
        new_item["score"] = item["score"]
        new_item["product_attributes"] = name_to_attribute[item["product"]]
        new_item["selected_attributes"] = item["attributes"]
        new_item["evaluation"] = item["evaluation"]
        new_item["evaluation_reason"] = item["evaluation_reason"]

        final_dataset.append(new_item)
        
    return final_dataset

def aggregate_data(filtered_data, eval_robustness, products_filepath):
    """filter out the similar responses and update the score with the average score of the group

    Args:
        filtered_data (dict): The formatted data with the raw queries grouped by the cluster ids
        eval_robustness (dict): The eval robustness data with the group scores

    Returns:
        list: The filtered and updated data
    """
    tmp_dataset = []

    for key, value in eval_robustness.items():
        queries_group = filtered_data[key]
        value_l = value["l"]
        group_scores = []
        for reduced_id, group_ids in value_l.items():
            found = False
            for item in queries_group["queries"]:
                if item["id"] == reduced_id:
                    picked_item = copy.deepcopy(item)
                    found = True
                    break
            if found:
                for item in queries_group["queries"]:
                    if item["id"] in group_ids:
                        if item["score"] != "None":
                            group_scores.append(float(item["score"]))
                avg_score = np.mean(group_scores)
                # picked_item["score"] = avg_score
                picked_item["global_id"] = key
                tmp_dataset.append(picked_item)

    # re-format the data with needed fields
    final_dataset = reformat(tmp_dataset, products_filepath)
    return final_dataset


def calculate_statistics(filtered_data):
    """calculate the statistics of the filtered dataset

    Args:
        filtered_data (dict): The formatted data with the raw queries grouped by the cluster ids
    """
    total_query_count = 0
    filtered_query_count = 0
    filtered_query_percentage_per_group = []

    high_filtered_query_percentage_group_scores = []
    low_filtered_query_percentage_group_scores = []
    high_filtered_query_percentage_products = []
    low_filtered_query_percentage_products = []
    for group_id, group_data in filtered_data.items():
        score_g = group_data["score"]
        current_filtered_query_count = 0
        for query_data in group_data["queries"]:
            total_query_count += 1
            if query_data["response_is_similar"]:
                filtered_query_count += 1
                current_filtered_query_count += 1
        current_filtered_query_percentage = current_filtered_query_count / len(group_data["queries"])
        filtered_query_percentage_per_group.append(current_filtered_query_percentage)
        if current_filtered_query_percentage > 0.8:
            high_filtered_query_percentage_group_scores.append(score_g)
            high_filtered_query_percentage_products.append(group_data["queries"][0]["product"])
        elif current_filtered_query_percentage < 0.4:
            low_filtered_query_percentage_group_scores.append(score_g)
            low_filtered_query_percentage_products.append(group_data["queries"][0]["product"])
        else:
            continue

    t_stat, p_value = stats.ttest_ind(high_filtered_query_percentage_group_scores, low_filtered_query_percentage_group_scores)
    
    # The score drop between global query and local queries
    score_drop = []
    score_drop_high = []
    score_drop_low = []
    for group_id, group_data in filtered_data.items():
        score_g = float(group_data["score"])
        if score_g == 0:
            score_g = 0.1
        for query_data in group_data["queries"]:
            if query_data["id"] != 0 and not query_data["response_is_similar"]:
                score_drop.append((float(score_g) - float(query_data["score"])))
                if score_g > 1.5: # high score group
                    score_drop_high.append((float(score_g) - float(query_data["score"])))
                else:
                    score_drop_low.append((float(score_g) - float(query_data["score"])))
    print("="*100)
    print("Calculate the statistics of the score drop between global query and local queries, where high score group is defined as score > 1.5 and low score group is defined as score <= 1.5")
    print(f"score_drop mean: {np.mean(score_drop)}, score_drop std: {np.std(score_drop)}")
    print(f"score_drop_high mean: {np.mean(score_drop_high)}, score_drop_high std: {np.std(score_drop_high)}")
    print(f"score_drop_low mean: {np.mean(score_drop_low)}, score_drop_low std: {np.std(score_drop_low)}")

    print("="*100)
    print("Calculate the statistics of the dataset")
    print(f"total_query_count: {total_query_count}, filtered_query_count: {filtered_query_count}, final dataset count: {total_query_count-filtered_query_count}")

    print("="*100)
    print("Calculate the statistics of the filtered query percentage per group, where filtered ")
    print(f"Avg percentage of filtered queries per group: {np.mean(filtered_query_percentage_per_group)}, filtered_query_percentage_per_group std: {np.std(filtered_query_percentage_per_group)}")
    print(f"For the percentage larger than 0.8, what are their corresponding global scores: {high_filtered_query_percentage_group_scores}")
    print(f"For the percentage smaller than 0.4, what are their corresponding global scores: {low_filtered_query_percentage_group_scores}")
    if p_value < 0.05:
        print(f"t-statistic: {t_stat}, p-value: {p_value}, There is significant difference between the high and low filtered query percentage groups based on the global scores")
    else:
        print(f"t-statistic: {t_stat}, p-value: {p_value}, There is no significant difference between the high and low filtered query percentage groups based on the global scores")
    

    print("="*100)
    print(f"Products in the high filtered query percentage groups: {high_filtered_query_percentage_products}")
    print(f"Products in the low filtered query percentage groups: {low_filtered_query_percentage_products}")
    # print("No abvious relationship between the filtered query percentage and the products")

    print("="*100)


def eval_score_robustness(filtered_data, eval_robustness):
    """evaluate the score robustness of the filtered dataset

    Args:
        filtered_data (dict): The filtered data
        eval_robustness (dict): The eval robustness data
    """

    p_values = []

    for key, value in eval_robustness.items():
        queries_group = filtered_data[key]
        value_l = value["l"]
        group_scores = []
        for reduced_id, group_ids in value_l.items():
            found = False
            if reduced_id == 0:
                found = True
                representative_score = float(value["g"])
            # else:
            #     for item in queries_group["queries"]:
            #         if item["id"] == reduced_id:
            #             representative_score = float(item["score"])
            #             found = True
            #             break
            if found:
                for item in queries_group["queries"]:
                    if item["id"] in group_ids:
                        if item["score"] != "None":
                            group_scores.append(float(item["score"]))

                if np.std(group_scores) == 0 and np.mean(group_scores) == representative_score:
                    t_stat = 0
                    p_value = 1
                elif len(group_scores) == 1 and abs(group_scores[0] - representative_score) < 0.25:
                    t_stat = 0
                    p_value = 1
                else:
                    t_stat, p_value = stats.ttest_1samp(group_scores, popmean=representative_score)
                
                if p_value == np.nan:
                    p_value = 1
                p_values.append(p_value)
        # print(f"{key}: t-statistic: {t_stat}, p-value: {p_value}")

    unrobustness_count = 0
    for idx, p_value in enumerate(p_values):
        if p_value < 0.05:
            unrobustness_count += 1
    print(f"unrobustness_count: {unrobustness_count}/{len(p_values)}")

    return p_values

def main(args):

    # 1. Load raw data
    with open(args.input_file, "r") as f:
        raw_data = json.load(f)

    # 2. Remove the data with the similar responses
    filtered_data = {}  # {node_id}_{index}: {query: {query_g}, response: {response_g}, score: {}, queries: []}
    response_edit_distances = {}  # {node_id}_{index}: [response_edit_distance]
    eval_robustness = {}  # {node_id}_{index}: {"g": score, "l": {"{reduced_id}": [group_ids], ...}}
    groups_response = {}

    for node in raw_data:
        node_id = node["id"]
        queries = node["queries"]
        for idx, query_group in enumerate(queries):
            filtered_data[f"{node_id}_{idx}"] = {}
            filtered_data[f"{node_id}_{idx}"]["queries"] = []
            response_edit_distances[f"{node_id}_{idx}"] = []
            eval_robustness[f"{node_id}_{idx}"] = {"g": "", "l": {}}
            for id, query in enumerate(query_group):
                # print(f"query {id} in query group {idx}")
                if query["id"] == 0 and query["parent_id"] is None:
                    query_g_content = query["query"]
                    response_g = query["response"]
                    score_g = query["score"]
                    eval_robustness[f"{node_id}_{idx}"]["g"] = score_g

                    filtered_data[f"{node_id}_{idx}"]["query"] = query_g_content
                    filtered_data[f"{node_id}_{idx}"]["response"] = response_g
                    filtered_data[f"{node_id}_{idx}"]["score"] = score_g

                filtered_data[f"{node_id}_{idx}"]["queries"].append(query)

            groups, reduced = group_strings(filtered_data[f"{node_id}_{idx}"], args.filter_out_threshold)
            for reduced_id, group_ids in zip(reduced, groups):
                eval_robustness[f"{node_id}_{idx}"]["l"][reduced_id] = group_ids
                groups_response[f"{node_id}_{idx}_{reduced_id}"] = []
                if reduced_id == 0:
                    groups_response[f"{node_id}_{idx}_{reduced_id}"].append(response_g)
                for query in filtered_data[f"{node_id}_{idx}"]["queries"]:
                    if query["id"] in group_ids:
                        groups_response[f"{node_id}_{idx}_{reduced_id}"].append(query["response"])
                
            for query in filtered_data[f"{node_id}_{idx}"]["queries"]:
                if query["id"] in reduced:
                    query["response_is_similar"] = False
                else:
                    query["response_is_similar"] = True

    final_dataset = aggregate_data(filtered_data, eval_robustness, args.products_filepath)

    return final_dataset, filtered_data, eval_robustness



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="/home/yunanlu/code/aqd-for-agent/output_data/gpt5-mini_1126/results_seedprompt0.json", help="path to the input file")
    parser.add_argument("--output_dir", type=str, default="/home/yunanlu/code/aqd-for-agent/output_data/gpt5-mini_1126/cleaned_dataset", help="path to the output directory")
    parser.add_argument("--filter_out_threshold", type=float, default=0.2, help="threshold for filtering out similar responses")
    parser.add_argument("--products_filepath", type=str, default="/local/data/shared/aqd/input_data/walmart_products.json", help="path to the products file")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    final_dataset, filtered_data, eval_robustness = main(args)
    with open(os.path.join(args.output_dir, "cleaned_dataset.json"), "w") as f:
        json.dump(final_dataset, f, indent=4)

    calculate_statistics(filtered_data)
    eval_score_robustness(filtered_data, eval_robustness)

'''
python scripts/data/data_preprocess.py \
--input_file /home/yunanlu/code/aqd-for-agent/output_data/gpt5-mini_1126/results_seedprompt0.json \
--output_dir /home/yunanlu/code/aqd-for-agent/output_data/gpt5-mini_1126/cleaned_dataset \
--filter_out_threshold 0.2 \
--products_filepath /local/data/shared/aqd/input_data/walmart_products.json
'''
