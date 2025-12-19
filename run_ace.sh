python3 -m src.main run \
    --task "ace" \
    --generate_model 'gpt-5-mini' \
    --target_model 'qwen3:8b' \
    --evaluation_models 'gpt-5-mini' 'deepseek-reasoner' \
    --preferred_definition 'input_data/preferred_definition.json' \
    --unpreferred_definition 'input_data/unpreferred_definition.json' \
    --filtering_policy 'input_data/realism_definition.txt' \
    --query_refinement_strategies 'input_data/query_refinement_strategies.json' \
    --init_prompt 'input_data/seed_prompts.json' \
    --products 'input_data/product_info_1.json' \
    --output_path 'output_data/ace/results.json' \
    --n_queries 10 \
    --n_threads 50 \
    --budget 1000 \
    --seed 617 \

# --evaluation_models 'gpt-5-mini' 'gemini-2.5-flash' 'deepseek-reasoner' \