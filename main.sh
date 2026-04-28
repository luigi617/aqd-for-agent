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


#### Resume the process if encounter error in the middle
python3 -m src.main resume \
    --prev_result 'output_data/test/results_depth0.json' \
    --config_path 'output_data/test/config.json' \
    --output_path 'output_data/test1/results.json'


#### ablation - only query refinement
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

##### ablation - only prompt refinement
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