#### Main command to start PAIR method by using gpt-5-mini as generation model.
python3 -m src.main run \
    --task tap \
    --generate_model 'gpt-5-mini' \
    --target_model 'qwen3:8b' \
    --evaluation_models 'gpt-5-mini' 'gemini-2.5-flash' 'deepseek-reasoner' \
    --preferred_definition 'input_data/preferred_definition.json' \
    --unpreferred_definition 'input_data/unpreferred_definition.json' \
    --filtering_policy 'input_data/realism_definition.txt' \
    --query_refinement_strategies 'input_data/query_refinement_strategies.json' \
    --init_prompt 'input_data/init_prompt.json' \
    --products '<your domain knowledge>' \
    --output_path 'output_data/gpt5/tap/results.json' \
    --n_threads 64 \
    --budget 100 \
    --seed 617 \
    --n_branching_factor 4 \
    --max_width 4 \
    --max_depth 3

# gemma3:27b
python3 -m src.main run \
    --task tap \
    --generate_model 'gemma3:27b' \
    --target_model 'qwen3:8b' \
    --evaluation_models 'gpt-5-mini' 'gemini-2.5-flash' 'deepseek-reasoner' \
    --preferred_definition 'input_data/preferred_definition.json' \
    --unpreferred_definition 'input_data/unpreferred_definition.json' \
    --filtering_policy 'input_data/realism_definition.txt' \
    --query_refinement_strategies 'input_data/query_refinement_strategies.json' \
    --init_prompt 'input_data/init_prompt.json' \
    --products '<your domain knowledge>' \
    --output_path 'output_data/gemma/tap/results.json' \
    --n_threads 64 \
    --budget 100 \
    --seed 617 \
    --n_branching_factor 4 \
    --max_width 4 \
    --max_depth 3

    

