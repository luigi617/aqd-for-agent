#### Main command to start GEPA method by using gpt-5-mini as generation model.
python3 -m src.main run \
    --task "gepa" \
    --generate_model 'gpt-5-mini' \
    --target_model 'qwen3:8b' \
    --evaluation_models 'gpt-5-mini' 'gemini-2.5-flash' 'deepseek-reasoner' \
    --preferred_definition 'input_data/preferred_definition.json' \
    --unpreferred_definition 'input_data/unpreferred_definition.json' \
    --filtering_policy 'input_data/realism_definition.txt' \
    --query_refinement_strategies 'input_data/query_refinement_strategies.json' \
    --init_prompt 'input_data/init_prompt.json' \
    --products '<your domain knowledge>' \
    --output_path 'output_data/gpt5/gepa/results.json' \
    --n_prompt_refinement_iterations 30 \
    --n_threads 64 \
    --budget 1000 \
    --seed 617 \

# gemma3:27b
python3 -m src.main run \
    --task "gepa" \
    --generate_model 'gemma3:27b' \
    --target_model 'qwen3:8b' \
    --evaluation_models 'gpt-5-mini' 'gemini-2.5-flash' 'deepseek-reasoner' \
    --preferred_definition 'input_data/preferred_definition.json' \
    --unpreferred_definition 'input_data/unpreferred_definition.json' \
    --filtering_policy 'input_data/realism_definition.txt' \
    --query_refinement_strategies 'input_data/query_refinement_strategies.json' \
    --init_prompt 'input_data/init_prompt.json' \
    --products '<your domain knowledge>' \
    --output_path 'output_data/gpt5/gepa/results.json' \
    --n_prompt_refinement_iterations 30 \
    --n_threads 64 \
    --budget 1000 \
    --seed 617 \


# if you need to resume from previous iteration, would need to decrease number of iterations
python3 -m src.main resume \
    --task "gepa" \
    --prev_result 'output_data/gpt5/gepa/gepa_prompt_log.json' \
    --config_path 'output_data/gpt5/gepa/config.json'\
    --output_path 'output_data/gpt5/gepa/gepa_prompt_log.json' \