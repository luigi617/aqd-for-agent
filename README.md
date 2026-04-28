# PQR: A Framework to Generate Diverse and Realistic User Queries that Elicit QA Agent Failures

## Setup Ollama
We use ollama as the local LLM server.

Follow the instructions [here](https://docs.ollama.com/linux) to install ollama on linux. If you don't have sudo permission, you can install ollama in your home directory and add it to your PATH:
```
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz

mkdir -p ollama

tar -C ollama -xzvf ollama-linux-amd64.tgz

export PATH=$PATH:$HOME/ollama/bin
```

Set the ollama server url in the environment variable `OLLAMA_URL`, which will be used in the code. For example,
```
export OLLAMA_URL="http://localhost:9999"
```

Start the ollama server:
```
OLLAMA_HOST=0.0.0.0:9999 ollama serve
```

## Our Method
### Run the Method
To run our method, use the command `sh main.sh`

### Evaluation
#### Step 1: Preprocess the data
It will save the cleaned_dataset.json in the output_dir.
```
python scripts/data/data_preprocess.py \
--input_file output_data/gpt5/pqr/results.json \
--output_dir output_data/evaluation/gpt5/pqr \
--filter_out_threshold 0.2 \
--products_filepath '<your domain knowledge>' \
--task ours
```
#### Step 2: Run the automatic evaluation
It will save the eval_result in the output_dir.
```
python -m scripts.eval.eval \
--input_file output_data/evaluation/gpt5/pqr/cleaned_dataset.json \
--output_dir output_data/evaluation/gpt5/pqr/eval_1_data_per_product \
--product_file '<your domain knowledge>' \
--embedding_model_path Qwen/Qwen3-Embedding-0.6B \
--budget_file output_data/gpt5/pqr/budget.json \
--task ours \
--data_sample 1_data_per_product
```
Results:
- unhelpful_data.json: all the unhelpful data extracted from the data
- extracted_data.json: the extracted data for evaluation
- lexical_diversity.json: the lexical diversity result, including mtld_score, hdd, log_ttr, root_ttr, vocab_size, distincts, ngram_entropy, tokens_per_turn, num_tokens
- semantic_diversity.json: the semantic diversity result, including embedding_diversity, n_clusters, cluster_diversity, dataset_dcscore
- statistics.json: the statistics of the data, including total_query_count, product_count, query_per_product_count, percentage_of_unhelpful_queries, average_completion_token_cost_per_query, average_prompt_token_cost_per_query, average_cached_prompt_token_cost_per_query


## Baseline Methods
### GEPA
Install GEPA package by running the command:
```
pip install gepa
```
To run GEPA, use the command `sh run_gepa.sh`

### Other Baseline Methods
The code for other baseline methods are adapted from the original code of the methods.

### Evaluation
#### Step 0: preprocess the data to the format of PQR method data (Take GEPA as an example, other baseline methods are similar)
```
python scripts/data/baseline_data_preprocess.py \
--input_file output_data/gpt5/gepa/gepa_prompt_log.json \
--output_file output_data/evaluation/gpt5/gepa/formatted_results.json \
--task gepa
```

#### Step 1: Preprocess the data
It will save the cleaned_dataset.json in the output_dir.
```
python scripts/data/data_preprocess.py \
--input_file output_data/evaluation/gpt5/gepa/formatted_results.json \
--output_dir output_data/evaluation/gpt5/gepa \
--filter_out_threshold 0.2 \
--products_filepath '<your domain knowledge>' \
--task gepa
```

#### Step 2: Run the automatic evaluation
It will save the eval_result in the output_dir.
```
python -m scripts.eval.eval \
--input_file output_data/evaluation/gpt5/gepa/cleaned_dataset.json \
--output_dir output_data/evaluation/gpt5/gepa/eval_1_data_per_product \
--product_file '<your domain knowledge>' \
--embedding_model_path Qwen/Qwen3-Embedding-0.6B \
--budget_file output_data/gpt5/gepa/budget.json \
--task gepa \
--data_sample 1_data_per_product
```