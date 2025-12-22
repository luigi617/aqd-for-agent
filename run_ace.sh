python3 -m src.main run \
    --task "ace" \
    --generate_model 'qwen3:8b' \
    --target_model 'qwen3:8b' \
    --evaluation_models 'qwen3:8b' \
    --preferred_definition 'input_data/preferred_definition.json' \
    --unpreferred_definition 'input_data/unpreferred_definition.json' \
    --filtering_policy 'input_data/realism_definition.txt' \
    --query_refinement_strategies 'input_data/query_refinement_strategies.json' \
    --init_prompt 'input_data/seed_prompts.json' \
    --products 'input_data/product_info_1.json' \
    --output_path 'output_data/ace/results.json' \
    --n_queries 100 \
    --n_threads 20 \
    --budget 1000 \
    --seed 617 \

# --evaluation_models 'gpt-5-mini' 'gemini-2.5-flash' 'deepseek-reasoner' \