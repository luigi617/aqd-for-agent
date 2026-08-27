#### This file contains four separate PQR runs (main + 2 ablations + gemma variant),
#### run back-to-back if you execute the whole file with `sh main.sh`.
#### In practice, comment out the blocks you don't need and run one at a time.
#### Replace '<your domain knowledge>' with the path to your products JSON in every
#### block below before running (see README.md Section 0.4).

#### Main command to start PQR method by using gpt-5-mini as generation model.
python3 -m src.main run \
    --generate_model 'gpt-5-mini' \
    --target_model 'qwen3:8b' \
    --evaluation_models 'gpt-5-mini' 'gemini-2.5-flash' 'deepseek-reasoner' \
    --preferred_definition 'input_data/preferred_definition.json' \
    --unpreferred_definition 'input_data/unpreferred_definition.json' \
    --filtering_policy 'input_data/realism_definition.txt' \
    --query_refinement_strategies 'input_data/query_refinement_strategies.json' \
    --init_prompt 'input_data/init_prompt.json' \
    --products '<your domain knowledge>' \
    --output_path 'output_data/gpt5/pqr/results.json' \
    --n_queries 20 \
    --n_prompt_refinement_iterations 4 \
    --n_prompt_refinement_beam_search_size 2 \
    --n_query_refinement_iterations 3 \
    --n_query_refinement_beam_search_size 3 \
    --n_threads 64 \
    --budget 1000 \
    --seed 617 \


#### Resume the process if encounter error in the middle.
#### Point --prev_result / --config_path at the partial output + config saved by the
#### interrupted run (both are written automatically alongside --output_path), and
#### --output_path at where the completed run should be written.
python3 -m src.main resume \
    --prev_result 'output_data/test/results_depth0.json' \
    --config_path 'output_data/test/config.json' \
    --output_path 'output_data/test1/results.json'


#### Ablation: query refinement only (--n_prompt_refinement_iterations 0), i.e. no
#### prompt-refinement loop — isolates the query-refinement module's contribution.
python3 -m src.main run \
    --generate_model 'gpt-5-mini' \
    --target_model 'qwen3:8b' \
    --evaluation_models 'gpt-5-mini' 'gemini-2.5-flash' 'deepseek-reasoner' \
    --preferred_definition 'input_data/preferred_definition.json' \
    --unpreferred_definition 'input_data/unpreferred_definition.json' \
    --filtering_policy 'input_data/realism_definition.txt' \
    --query_refinement_strategies 'input_data/query_refinement_strategies.json' \
    --init_prompt 'input_data/init_prompt.json' \
    --products '<your domain knowledge>' \
    --output_path 'output_data/ablation/query_only/results.json' \
    --n_queries 100 \
    --n_prompt_refinement_iterations 0 \
    --n_prompt_refinement_beam_search_size 2 \
    --n_query_refinement_iterations 3 \
    --n_query_refinement_beam_search_size 3 \
    --n_threads 64 \
    --budget 1000 \
    --seed 617 \

#### Ablation: prompt refinement only (--n_query_refinement_iterations 0), i.e. no
#### query-refinement rewrites — isolates the prompt-refinement module's contribution.
python3 -m src.main run \
    --generate_model 'gpt-5-mini' \
    --target_model 'qwen3:8b' \
    --evaluation_models 'gpt-5-mini' 'gemini-2.5-flash' 'deepseek-reasoner' \
    --preferred_definition 'input_data/preferred_definition.json' \
    --unpreferred_definition 'input_data/unpreferred_definition.json' \
    --filtering_policy 'input_data/realism_definition.txt' \
    --query_refinement_strategies 'input_data/query_refinement_strategies.json' \
    --init_prompt 'input_data/init_prompt.json' \
    --products '<your domain knowledge>' \
    --output_path 'output_data/ablation/prompt_only/results.json' \
    --n_queries 20 \
    --n_prompt_refinement_iterations 25 \
    --n_prompt_refinement_beam_search_size 2 \
    --n_query_refinement_iterations 0 \
    --n_query_refinement_beam_search_size 3 \
    --n_threads 64 \
    --budget 1000 \
    --seed 617 \


#### Main command to start PQR method by using gemma3-27b as generation model.
#### Requires the model to be pulled in Ollama first: `ollama pull gemma3:27b`.
python3 -m src.main run \
    --generate_model 'gemma3:27b' \
    --target_model 'qwen3:8b' \
    --evaluation_models 'gpt-5-mini' 'gemini-2.5-flash' 'deepseek-reasoner' \
    --preferred_definition 'input_data/preferred_definition.json' \
    --unpreferred_definition 'input_data/unpreferred_definition.json' \
    --filtering_policy 'input_data/realism_definition.txt' \
    --query_refinement_strategies 'input_data/query_refinement_strategies.json' \
    --init_prompt 'input_data/init_prompt.json' \
    --products '<your domain knowledge>' \
    --output_path 'output_data/gemma/pqr/results.json' \
    --n_queries 20 \
    --n_prompt_refinement_iterations 4 \
    --n_prompt_refinement_beam_search_size 2 \
    --n_query_refinement_iterations 3 \
    --n_query_refinement_beam_search_size 3 \
    --n_threads 64 \
    --budget 1000 \
    --seed 617 \