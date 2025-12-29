# aqd-for-agent
Adversarial Query Discovery for Objective Violations in AI Agents

## Setup Ollama
Set the ollama server url in the environment variable `OLLAMA_URL`. For example,
```
export OLLAMA_URL="http://localhost:9999"
```

## Our Method
### Run the Method
To run our method, use the command `sh run_ours.sh`

### Evaluation
#### Step 1: Preprocess the data
It will save the cleaned_dataset.json in the output_dir.
```
python scripts/data/data_preprocess.py \
--input_file output_data/ours/results.json \
--output_dir output_data/gpt5-mini_1218/ours \
--filter_out_threshold 0.2 \
--products_filepath input_data/product_info.json \
--task ours
```
#### Step 2: Run the automatic evaluation
It will save the eval_result in the output_dir.
```
python -m scripts.eval.eval \
--input_file output_data/gpt5-mini_1218/ours/cleaned_dataset.json \
--output_dir output_data/gpt5-mini_1218/ours/eval_1_data_per_product \
--product_file input_data/product_info_1.json \
--embedding_model_path Qwen/Qwen3-Embedding-0.6B \
--budget_file output_data/ours/budget.json \
--task ours \
--data_sample 1_data_per_product
```
Results:
- unhelpful_data.json: all the unhelpful data extracted from the data
- extracted_data.json: the extracted data for evaluation
- lexical_diversity.json: the lexical diversity result, including mtld_score, hdd, log_ttr, root_ttr, vocab_size, distincts, ngram_entropy, tokens_per_turn, num_tokens
- semantic_diversity.json: the semantic diversity result, including embedding_diversity, n_clusters, cluster_diversity, dataset_dcscore
- statistics.json: the statistics of the data, including total_query_count, product_count, query_per_product_count, percentage_of_unhelpful_queries, average_completion_token_cost_per_query, average_prompt_token_cost_per_query, average_cached_prompt_token_cost_per_query


## GEPA
### Run the Method
To install gepa, do the following:
```
pip install gepa
```
To run GEPA, use the command `sh run_gepa.sh`

### Evaluation
#### Step 0: preprocess the data to the format of OUR method data
```
python scripts/data/baseline_data_preprocess.py \
--input_file output_data/gepa/full_run/gepa_prompt_log.json \
--output_file output_data/gpt5-mini_1218/gepa/formatted_results.json \
--task gepa
```

All the other steps are the same as OUR method.