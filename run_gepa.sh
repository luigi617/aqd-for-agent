# query refinement, filtering policy unused (just so program can run)
python3 -m src.main run \
    --task "gepa" \
    --generate_model 'gpt-5-mini' \
    --target_model 'qwen3:8b' \
    --evaluation_models 'gpt-5-mini' 'deepseek-reasoner' \
    --preferred_definition 'input_data/preferred_definition.json' \
    --unpreferred_definition 'input_data/unpreferred_definition.json' \
    --filtering_policy 'input_data/realism_definition.txt' \
    --query_refinement_strategies 'input_data/query_refinement_strategies.json' \
    --init_prompt 'input_data/seed_prompts.json' \
    --products 'input_data/product_info_1.json' \
    --output_path 'output_data/gepa/test_run/results.json' \
    --n_prompt_refinement_iterations 1 \
    --n_threads 50 \
    --budget 1000 \
    --seed 617 \

# python3 -m src.main run \
#     --task "gepa" \
#     --generate_model 'gpt-5-mini' \
#     --target_model 'qwen3:8b' \
#     --evaluation_models 'gpt-5-mini' 'gemini-2.5-flash' 'deepseek-reasoner' \
#     --preferred_definition 'input_data/preferred_definition.json' \
#     --unpreferred_definition 'input_data/unpreferred_definition.json' \
#     --filtering_policy 'input_data/realism_definition.txt' \
#     --query_refinement_strategies 'input_data/query_refinement_strategies.json' \
#     --init_prompt 'input_data/seed_prompts.json' \
#     --products 'input_data/product_info_1.json' \
#     --output_path 'output_data/gepa/full_run/results.json' \
#     --n_prompt_refinement_iterations 30 \
#     --n_threads 50 \
#     --budget 1000 \
#     --seed 617 \

# --evaluation_models 'gpt-5-mini' 'gemini-2.5-flash' 'deepseek-reasoner' \

# if you need  to resume from previous iteration, would need to decrease number of iterations
# python3 -m src.main resume \
#     --task "gepa" \
#     --prev_result 'output_data/full_run/gepa_prompt_log.json' \
#     --config_path 'output_data/full_run/config.json'\
#     --output_path 'output_data/full_run/gepa_prompt_log.json' \