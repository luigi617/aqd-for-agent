
python3 src/main.py \
    --generate_model 'gpt-5-nano' \
    --target_model 'qwen3:8b' \
    --preferred_definition 'input_data/preferred_definition.json' \
    --unpreferred_definition 'input_data/unpreferred_definition.json' \
    --filtering_policy 'input_data/realism_definition.txt' \
    --input_pool 'input_data/seed_prompts.json' \
    --products 'input_data/walmart_products.json' \
    --output_path 'output_data/gpt5-nano/results.json' \
    --n_queries 5 \
    --n_prompt_refinemnet_iterations 3 \
    --n_prompt_refinemnet_children 3 \
    --n_query_refiment_iterations 3 \
    --n_query_refiment_children 3 \
    --n_threads 45 \
    --seed 42 \
